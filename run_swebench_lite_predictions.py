"""
Generate predictions for SWE-bench Lite and evaluate real pass rate with the official harness.

This script:
1) Loads SWE-bench Lite dataset (HF or local json/jsonl)
2) Samples N instances deterministically (seed)
3) Checks out each repo at its base_commit
4) Runs THIS repo's workflow to generate a patch
5) Writes a predictions .jsonl file compatible with SWE-bench harness:
   {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}

Notes:
- This script only generates patches; to measure "real pass rate" you must run the
  SWE-bench evaluation harness (Docker-based) on Linux/WSL.
- Default file selection is intentionally conservative (1 file) because the current
  GraphManager pipeline parses a single Python module via AST.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

from src.enhanced_graph_adapter import AnalysisStrategy
from src.state import AgentState


class TokenCounterCallback(BaseCallbackHandler):
    """Best-effort token counter for LangChain ChatOpenAI calls."""

    def __init__(self) -> None:
        super().__init__()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.successful_requests = 0

    def _add_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        self.successful_requests += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        """Record usage from LLMResult/ChatResult llm_output when available."""
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            self._add_usage(usage)
        except Exception:
            return

    def on_chat_model_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        """Record usage; some versions call this instead of on_llm_end."""
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            self._add_usage(usage)
        except Exception:
            return


SWE_BENCH_LITE_DATASET = "SWE-bench/SWE-bench_Lite"


def _detect_file_style(path: Path) -> Tuple[str, bool]:
    data = path.read_bytes()
    newline = "\r\n" if b"\r\n" in data else "\n"
    ends_with_newline = data.endswith(b"\n") or data.endswith(b"\r")
    return newline, ends_with_newline


def _apply_file_style(content: str, *, newline: str, ends_with_newline: bool) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if newline == "\r\n":
        normalized = normalized.replace("\n", "\r\n")
    if ends_with_newline:
        if newline == "\r\n" and not normalized.endswith("\r\n"):
            normalized += "\r\n"
        elif newline == "\n" and not normalized.endswith("\n"):
            normalized += "\n"
    else:
        if newline == "\r\n" and normalized.endswith("\r\n"):
            normalized = normalized[:-2]
        elif newline == "\n" and normalized.endswith("\n"):
            normalized = normalized[:-1]
    return normalized


def _write_text_exact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid Windows text-mode newline translation which can explode diffs (LF→CRLF or CRLF→CRCRLF).
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(content)


def _run(cmd: Sequence[str], *, cwd: Path, timeout_s: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=True,
    )


def _git(repo_dir: Path, args: Sequence[str], *, timeout_s: int = 600) -> str:
    cp = _run(["git", *args], cwd=repo_dir, timeout_s=timeout_s)
    return cp.stdout


def _load_dataset(dataset: str, split: str) -> List[Dict[str, Any]]:
    """
    Load SWE-bench dataset from HF or local json/jsonl.
    We avoid importing swebench.harness on Windows (it imports `resource` unconditionally).
    """
    if dataset.endswith(".json"):
        return json.loads(Path(dataset).read_text(encoding="utf-8"))
    if dataset.endswith(".jsonl"):
        return [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]

    try:
        from datasets import load_dataset as hf_load_dataset
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install it with `python -m pip install datasets` "
            "or pass a local dataset file via `--dataset path/to/SWE-bench_Lite.jsonl`."
        ) from e

    ds = hf_load_dataset(dataset, split=split)
    return [dict(row) for row in ds]


def _sample_instances(instances: List[Dict[str, Any]], *, n: int, seed: int) -> List[Dict[str, Any]]:
    instances_sorted = sorted(instances, key=lambda x: str(x.get("instance_id", "")))
    rng = random.Random(seed)
    rng.shuffle(instances_sorted)
    return instances_sorted[:n]


def _repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


def _ensure_checkout(repo_root: Path, *, repo: str, base_commit: str) -> None:
    repo_url = f"https://github.com/{repo}.git"
    repo_root.mkdir(parents=True, exist_ok=True)

    if not (repo_root / ".git").exists():
        _git(repo_root, ["init"])
        _git(repo_root, ["remote", "add", "origin", repo_url])
        # Prevent CRLF normalization from polluting diffs on Windows.
        _git(repo_root, ["config", "core.autocrlf", "false"])
        _git(repo_root, ["config", "core.eol", "lf"])

    _git(repo_root, ["fetch", "--depth", "1", "origin", base_commit], timeout_s=1200)
    _git(repo_root, ["checkout", "--force", "FETCH_HEAD"])
    _git(repo_root, ["reset", "--hard"])
    _git(repo_root, ["clean", "-fdx"])


_STOPWORDS = {
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "with",
    "a",
    "an",
    "is",
    "are",
    "be",
    "as",
    "on",
    "this",
    "that",
    "it",
    "when",
    "from",
    "by",
    "should",
    "must",
    "not",
    "does",
    "do",
    "will",
    "can",
}


def _extract_candidate_paths(text: str) -> List[str]:
    # crude but effective for "file.py" mentions
    paths = re.findall(r"(?:(?:^|\s|\())([\w./-]+\.py)(?:$|\s|\)|:)", text, flags=re.MULTILINE)
    cleaned = []
    for p in paths:
        p = p.strip()
        if p.startswith("./") or p.startswith(".\\"):
            p = p[2:]
        p = p.replace("\\", "/")
        if p and p not in cleaned:
            cleaned.append(p)
    return cleaned


def _extract_tokens(text: str, *, max_tokens: int = 12) -> List[str]:
    tokens: List[str] = []

    backticked = re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", text)
    tokens.extend(backticked)

    code_ids = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b", text)
    tokens.extend(code_ids)

    filtered = []
    seen = set()
    for t in tokens:
        tl = t.lower()
        if tl in _STOPWORDS:
            continue
        if tl.isdigit():
            continue
        if len(t) > 64:
            continue
        if t in seen:
            continue
        seen.add(t)
        filtered.append(t)

    def _priority(tok: str) -> int:
        if tok.startswith("__") and tok.endswith("__"):
            return 0
        if "_" in tok:
            return 1
        if any(c.isupper() for c in tok[1:]):
            return 2
        return 3

    # Prefer syntactically-informative tokens first (dunder > underscore > CamelCase),
    # then longer / more specific tokens.
    filtered.sort(key=lambda s: (_priority(s), -len(s), s))
    return filtered[:max_tokens]


def _is_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return "/tests/" in p or "/test/" in p or p.startswith("tests/") or p.startswith("test/")


def _rg_list_files(repo_root: Path, *, token: str, include_tests: bool) -> Optional[List[str]]:
    rg_globs = ["--glob", "*.py"]
    if not include_tests:
        rg_globs += ["--glob", "!**/tests/**", "--glob", "!**/test/**"]

    try:
        cp = subprocess.run(
            ["rg", "-l", "-S", *rg_globs, token, "."],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return None

    if cp.returncode not in (0, 1):
        return []

    hits: List[str] = []
    for line in (cp.stdout or "").splitlines():
        path = line.strip()
        if not path:
            continue
        if path.startswith("./") or path.startswith(".\\"):
            path = path[2:]
        hits.append(path.replace("\\", "/"))
    return hits


def _git_grep_list_files(repo_root: Path, *, token: str, include_tests: bool) -> Optional[List[str]]:
    exclude_args: List[str] = []
    if not include_tests:
        exclude_args = [":(exclude)**/tests/**", ":(exclude)**/test/**"]

    try:
        cp = subprocess.run(
            ["git", "grep", "-l", "-I", "-F", token, "--", "*.py", *exclude_args],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return None

    if cp.returncode not in (0, 1):
        return []

    hits: List[str] = []
    for line in (cp.stdout or "").splitlines():
        path = line.strip()
        if not path:
            continue
        hits.append(path.replace("\\", "/"))
    return hits


def _python_scan_list_files(repo_root: Path, *, token: str, include_tests: bool) -> List[str]:
    try:
        cp = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
        )
        candidates = [p.strip() for p in (cp.stdout or "").splitlines() if p.strip()]
    except FileNotFoundError:
        candidates = []

    if not candidates:
        candidates = [
            str(p.relative_to(repo_root)).replace("\\", "/")
            for p in repo_root.rglob("*.py")
            if p.is_file()
        ]

    token_is_lower = token.lower() == token
    token_l = token.lower()

    hits: List[str] = []
    for rel in candidates:
        if not include_tests and _is_test_path(rel):
            continue
        try:
            text = (repo_root / rel).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if token_is_lower:
            if token_l in text.lower():
                hits.append(rel)
        else:
            if token in text:
                hits.append(rel)
    return hits


def _list_files_for_token(repo_root: Path, *, token: str, include_tests: bool) -> List[str]:
    hits = _rg_list_files(repo_root, token=token, include_tests=include_tests)
    if hits is not None:
        return hits

    hits = _git_grep_list_files(repo_root, token=token, include_tests=include_tests)
    if hits is not None:
        return hits

    return _python_scan_list_files(repo_root, token=token, include_tests=include_tests)


def _select_target_file(repo_root: Path, instance: Dict[str, Any]) -> Optional[str]:
    problem = (instance.get("problem_statement") or "").strip()
    hints = (instance.get("hints_text") or "").strip()
    fail_to_pass = "\n".join(instance.get("FAIL_TO_PASS") or [])
    # PASS_TO_PASS is often huge/noisy and hurts file selection; keep FAIL_TO_PASS only.
    combined = f"{problem}\n\n{hints}\n\n{fail_to_pass}".strip()

    for p in _extract_candidate_paths(combined):
        candidate = repo_root / p
        if candidate.exists() and candidate.is_file():
            return p

    tokens = _extract_tokens(combined)
    if not tokens:
        return None

    _COMMON_DUNDERS = {
        "__init__",
        "__repr__",
        "__str__",
        "__call__",
        "__iter__",
        "__len__",
        "__getitem__",
        "__setitem__",
        "__enter__",
        "__exit__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
    }

    def _token_weight(tok: str) -> float:
        if tok.startswith("__") and tok.endswith("__"):
            return 3.0
        if "_" in tok:
            return 2.0
        if any(c.isupper() for c in tok[1:]):
            return 1.5
        return 1.0

    def _score(tokens_to_use: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        # Pass 1: prefer non-test files; Pass 2: allow tests as a last resort.
        for include_tests in (False, True):
            hits_by_token: Dict[str, List[str]] = {}
            for tok in tokens_to_use:
                hits_by_token[tok] = _list_files_for_token(repo_root, token=tok, include_tests=include_tests)

            # Drop extremely-common tokens which cause near-random tie-breaks (e.g., "__init__").
            filtered: List[str] = []
            for tok, hits in hits_by_token.items():
                if not hits:
                    continue
                if tok in _COMMON_DUNDERS:
                    continue
                if len(hits) > 200:
                    continue
                filtered.append(tok)

            # If everything got filtered out, fall back to the original list (still weighted).
            tokens_for_scoring = filtered if filtered else [t for t, h in hits_by_token.items() if h]

            # Prefer more specific tokens first (fewer hits).
            tokens_for_scoring.sort(key=lambda t: (len(hits_by_token.get(t, [])), -len(t), t))

            for tok in tokens_for_scoring[:12]:
                hits = hits_by_token.get(tok) or []
                if not hits:
                    continue
                weight = _token_weight(tok) / max(len(hits), 1)
                for path in hits:
                    if not path:
                        continue
                    scores[path] += weight
            if scores:
                break
        return scores

    # First try with high-signal dunder tokens (but skip ultra-common ones like "__init__").
    dunder_tokens = [
        t for t in tokens if t.startswith("__") and t.endswith("__") and t not in _COMMON_DUNDERS
    ]
    scores = _score(dunder_tokens) if dunder_tokens else {}
    if not scores:
        scores = _score(tokens)

    if not scores:
        return None

    # Tie-break: higher score, prefer non-tests, then shorter path.
    best = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], _is_test_path(kv[0]), len(kv[0]), kv[0]),
    )[0][0]
    best = best.strip()
    if best.startswith("./") or best.startswith(".\\"):
        best = best[2:]
    best = best.replace("\\", "/")
    return best


def _build_requirements(instance: Dict[str, Any]) -> str:
    problem = (instance.get("problem_statement") or "").strip()
    hints = (instance.get("hints_text") or "").strip()
    fail_to_pass = instance.get("FAIL_TO_PASS") or []

    parts: List[str] = []
    if problem:
        parts.append(problem)
    if hints:
        parts.append(f"Hints:\n{hints}")
    if fail_to_pass:
        tests = "\n".join(f"- {t}" for t in list(fail_to_pass)[:25])
        parts.append(f"Fix the following failing tests:\n{tests}")

    return "\n\n".join(parts).strip()


@dataclass
class SolveResult:
    instance_id: str
    repo: str
    base_commit: str
    model: str
    selected_files: List[str]
    patch: str
    success: bool
    error: Optional[str] = None
    duration_s: float = 0.0


def _default_workflow_builder(mode: str) -> str:
    if mode == "integrated":
        return "run_experiment_integrated:build_integrated_workflow"
    return "src.main_enhanced:build_workflow"


def _import_symbol(path: str) -> Any:
    if ":" in path:
        module_name, symbol_name = path.split(":", 1)
    else:
        module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _build_langgraph_app(
    *,
    workflow_builder: str,
    llm_model: str,
    analysis_model: str,
    analysis_strategy: str,
    max_revisions: int,
    callbacks: Sequence[Any],
) -> Any:
    builder = _import_symbol(workflow_builder)
    params = inspect.signature(builder).parameters

    kwargs: Dict[str, Any] = {}
    if "llm_model" in params:
        kwargs["llm_model"] = llm_model
    if "max_revisions" in params:
        kwargs["max_revisions"] = max_revisions
    if "analysis_model" in params:
        kwargs["analysis_model"] = analysis_model
    if "analysis_strategy" in params:
        kwargs["analysis_strategy"] = AnalysisStrategy(analysis_strategy)
    if "callbacks" in params:
        kwargs["callbacks"] = list(callbacks)

    workflow_or_app = builder(**kwargs)
    if hasattr(workflow_or_app, "compile"):
        return workflow_or_app.compile()
    return workflow_or_app


def _run_langgraph_workflow(
    *,
    workflow_builder: str,
    repo_root: Path,
    files: Dict[str, str],
    requirements: str,
    llm_model: str,
    analysis_model: str,
    analysis_strategy: str,
    max_revisions: int,
    recursion_limit: int,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    token_cb = TokenCounterCallback()
    app = _build_langgraph_app(
        workflow_builder=workflow_builder,
        llm_model=llm_model,
        analysis_model=analysis_model,
        analysis_strategy=analysis_strategy,
        max_revisions=max_revisions,
        callbacks=[token_cb],
    )

    initial_state: AgentState = {
        "messages": [],
        "files": files,
        "requirements": requirements,
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
        # Extra context for workflows that can load additional files on demand.
        "repo_root": str(repo_root),
    }

    # Some workflows add extra fields; initialize them to keep logging predictable.
    initial_state.setdefault("advanced_analysis", None)  # type: ignore[typeddict-item]
    initial_state.setdefault("analysis_report", None)  # type: ignore[typeddict-item]

    final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})

    updated_files = dict(final_state.get("files") or {})

    advanced_summary: Optional[Dict[str, Any]] = None
    adv = final_state.get("advanced_analysis")
    if isinstance(adv, dict):
        advanced_summary = {
            "strategy": adv.get("strategy"),
            "confidence": adv.get("confidence"),
            "processing_time": adv.get("processing_time"),
            "findings_count": len(adv.get("findings") or []),
            "recommendations_count": len(adv.get("recommendations") or []),
        }

    meta: Dict[str, Any] = {
        "workflow_builder": workflow_builder,
        "revision_count": int(final_state.get("revision_count", 0) or 0),
        "final_conflict_report": final_state.get("conflict_report"),
        "chat_usage": {
            "prompt_tokens": token_cb.prompt_tokens,
            "completion_tokens": token_cb.completion_tokens,
            "total_tokens": token_cb.total_tokens,
            "successful_requests": token_cb.successful_requests,
        },
    }
    if advanced_summary:
        meta["advanced_summary"] = advanced_summary

    advanced_usage = final_state.get("advanced_usage")
    if isinstance(advanced_usage, dict):
        meta["advanced_usage"] = advanced_usage

    return updated_files, meta
def _generate_patch(repo_root: Path) -> str:
    # Use git diff for a canonical patch that SWE-bench harness can apply.
    return _git(repo_root, ["diff", "--no-color"])


def _write_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SWE-bench Lite predictions (.jsonl).")
    parser.add_argument("--dataset", default=SWE_BENCH_LITE_DATASET, help="HF dataset name or local .json/.jsonl")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--instance-id",
        action="append",
        default=None,
        help="Run specific instance_id(s). Can be passed multiple times; overrides --num/--seed sampling.",
    )
    parser.add_argument("--num", type=int, default=30, help="Number of instances to sample (ignored if --start/--end used)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--start", type=int, default=None, help="Start index (0-based) for instance selection after shuffle")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) for instance selection after shuffle")
    parser.add_argument("--workdir", default="swebench_runs", help="Working directory for repo checkouts/logs")
    parser.add_argument("--output", default="predictions/swebench_lite_sample.jsonl", help="Output predictions jsonl")
    parser.add_argument("--model", default="gpt-5.1-codex-mini", help="LLM model for Developer/Judge (ChatOpenAI)")
    parser.add_argument("--analysis-model", default="gpt-5.1-codex-mini", help="LLM model for Advanced Analysis (LLMInterface)")
    parser.add_argument("--mode", choices=["integrated", "traditional"], default="integrated")
    parser.add_argument(
        "--workflow-builder",
        default=None,
        help=(
            "Optional override for workflow builder as `module:function`. "
            "Defaults to integrated=run_experiment_integrated:build_integrated_workflow "
            "and traditional=src.main_enhanced:build_workflow."
        ),
    )
    parser.add_argument(
        "--analysis-strategy",
        default=AnalysisStrategy.AUTO_SELECT.value,
        choices=[s.value for s in AnalysisStrategy],
        help="Advanced analysis strategy (only used if the selected workflow supports it).",
    )
    parser.add_argument("--max-revisions", type=int, default=1, help="Max judge-driven revisions per instance")
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="LangGraph recursion limit for workflow execution.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls; output empty patches")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv(override=True)
    workflow_builder = args.workflow_builder or _default_workflow_builder(args.mode)

    workdir = Path(args.workdir)
    repos_dir = workdir / "repos"
    logs_dir = workdir / "logs"
    repos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    existing_ids = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing_ids.add(json.loads(line).get("instance_id"))
            except Exception:
                continue

    instances = _load_dataset(args.dataset, args.split)
    if args.instance_id:
        wanted = [str(x) for x in args.instance_id if str(x).strip()]
        wanted_set = set(wanted)
        selected = [inst for inst in instances if str(inst.get("instance_id")) in wanted_set]
        found_set = {str(inst.get("instance_id")) for inst in selected}
        missing = sorted(wanted_set - found_set)
        if missing:
            raise RuntimeError(f"Requested instance_id(s) not found in dataset: {missing}")
        print(f"Selected {len(selected)} specified instance(s) from {args.dataset}:{args.split}")
    else:
        # Apply seed-based shuffle first
        instances_sorted = sorted(instances, key=lambda x: str(x.get("instance_id", "")))
        rng = random.Random(args.seed)
        rng.shuffle(instances_sorted)
        
        # Use --start/--end if provided, otherwise use --num
        if args.start is not None or args.end is not None:
            start_idx = args.start if args.start is not None else 0
            end_idx = args.end if args.end is not None else len(instances_sorted)
            selected = instances_sorted[start_idx:end_idx]
            print(f"Selected instances [{start_idx}:{end_idx}] (seed={args.seed}) from {args.dataset}:{args.split}")
        else:
            selected = instances_sorted[:args.num]
            print(f"Selected {len(selected)} instances (seed={args.seed}) from {args.dataset}:{args.split}")

    print(f"Predictions file: {output_path}")

    total_chat_tokens = 0
    total_adv_tokens = 0

    for inst in selected:
        instance_id = str(inst["instance_id"])
        if instance_id in existing_ids:
            print(f"[skip] {instance_id} already in predictions")
            continue

        start = time.time()
        repo = str(inst["repo"])
        base_commit = str(inst["base_commit"])
        repo_root = repos_dir / _repo_slug(repo)

        meta: Dict[str, Any] = {}
        selected_paths: List[str] = []
        result: SolveResult
        try:
            _ensure_checkout(repo_root, repo=repo, base_commit=base_commit)

            selected_file = _select_target_file(repo_root, inst)
            if not selected_file:
                raise RuntimeError("Could not select a target file (no matches found).")

            requirements = _build_requirements(inst)

            # Read file(s)
            selected_abs = repo_root / selected_file
            newline_style, ends_with_newline = _detect_file_style(selected_abs)
            files = {selected_file: selected_abs.read_text(encoding="utf-8")}
            file_styles = {selected_file: (newline_style, ends_with_newline)}
            selected_paths = [selected_file]

            patch = ""

            if not args.dry_run:
                updated_files, meta = _run_langgraph_workflow(
                    workflow_builder=workflow_builder,
                    repo_root=repo_root,
                    files=files,
                    requirements=requirements,
                    llm_model=args.model,
                    analysis_model=args.analysis_model,
                    analysis_strategy=args.analysis_strategy,
                    max_revisions=args.max_revisions,
                    recursion_limit=args.recursion_limit,
                )

                # Apply updated files to disk
                for rel_path in updated_files.keys():
                    if rel_path in file_styles:
                        continue
                    abs_path = repo_root / rel_path
                    try:
                        file_styles[rel_path] = _detect_file_style(abs_path)
                    except Exception:
                        file_styles[rel_path] = ("\n", True)
                for rel_path, content in updated_files.items():
                    abs_path = repo_root / rel_path
                    nl, eof_nl = file_styles.get(rel_path, ("\n", True))
                    _write_text_exact(
                        abs_path,
                        _apply_file_style(content, newline=nl, ends_with_newline=eof_nl),
                    )

                patch = _generate_patch(repo_root)

            duration = time.time() - start
            result = SolveResult(
                instance_id=instance_id,
                repo=repo,
                base_commit=base_commit,
                model=args.model,
                selected_files=selected_paths,
                patch=patch,
                success=True,
                duration_s=duration,
            )

            # Write per-instance log
            (logs_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": repo,
                        "base_commit": base_commit,
                        "selected_files": selected_paths,
                        "mode": args.mode,
                        "model": args.model,
                        "analysis_model": args.analysis_model,
                        "dry_run": args.dry_run,
                        "duration_s": duration,
                        "meta": meta,
                        "patch_chars": len(patch),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        except Exception as e:
            duration = time.time() - start
            result = SolveResult(
                instance_id=instance_id,
                repo=repo,
                base_commit=base_commit,
                model=args.model,
                selected_files=[],
                patch="",
                success=False,
                error=str(e),
                duration_s=duration,
            )
            try:
                (logs_dir / f"{instance_id}.json").write_text(
                    json.dumps(
                        {
                            "instance_id": instance_id,
                            "repo": repo,
                            "base_commit": base_commit,
                            "selected_files": selected_paths,
                            "mode": args.mode,
                            "model": args.model,
                            "analysis_model": args.analysis_model,
                            "dry_run": args.dry_run,
                            "duration_s": duration,
                            "meta": meta,
                            "error": str(e),
                            "patch_chars": 0,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # Always reset repo to keep workspace clean for next instance.
        try:
            if repo_root.exists() and (repo_root / ".git").exists():
                _git(repo_root, ["reset", "--hard"])
                _git(repo_root, ["clean", "-fdx"])
        except Exception:
            pass

        # Write prediction line (empty patch = fail in harness).
        _write_jsonl_line(
            output_path,
            {
                "instance_id": result.instance_id,
                "model_name_or_path": result.model,
                "model": result.model,
                "model_patch": result.patch,
            },
        )

        chat_tokens = 0
        adv_tokens = 0
        if meta:
            try:
                chat_tokens = int(meta.get("chat_usage", {}).get("total_tokens", 0) or 0)
                adv_tokens = int(meta.get("advanced_usage", {}).get("total_tokens", 0) or 0)
            except Exception:
                chat_tokens = 0
                adv_tokens = 0
        total_chat_tokens += chat_tokens
        total_adv_tokens += adv_tokens

        status = "ok" if result.success else f"error: {result.error}"
        token_info = ""
        if chat_tokens or adv_tokens:
            token_info = f" tokens(chat={chat_tokens}, adv={adv_tokens})"
        print(
            f"[{status}] {instance_id} ({repo}) {result.duration_s:.1f}s "
            f"patch={len(result.patch)} chars{token_info}"
        )

    print("\nDone.")
    if total_chat_tokens or total_adv_tokens:
        print(f"Total tokens: chat={total_chat_tokens}, advanced={total_adv_tokens}, combined={total_chat_tokens + total_adv_tokens}")
    print("Next: run SWE-bench harness on Linux/WSL, e.g.:")
    run_id = output_path.stem
    predictions_path = output_path.as_posix()
    print(
        "  python -m swebench.harness.run_evaluation "
        f"--predictions_path {predictions_path} --run_id {run_id} "
        f"--dataset_name {args.dataset} --split {args.split} --max_workers 1"
    )


if __name__ == "__main__":
    main()
