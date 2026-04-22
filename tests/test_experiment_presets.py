from src.enhanced_graph_adapter import AnalysisStrategy

from run_experiment_integrated import (
    _bounded_fallback_target_set,
    _file_local_neighborhood_target_set,
    _summarize_fallback_usage,
)
from run_swebench_lite_predictions import (
    _extra_context_files_for_preset,
    _resolve_experiment_preset,
    _retrieved_context_files_for_preset,
    _build_requirements,
)


def test_ablation1_preset_maps_to_reproducible_graph_only_builder():
    preset = _resolve_experiment_preset("ablation1", mode="integrated")

    assert preset.name == "ablation1"
    assert preset.workflow_builder == "run_experiment_integrated:build_integrated_workflow_ablation1"
    assert preset.analysis_strategy == AnalysisStrategy.GRAPH_ONLY.value
    assert preset.max_revisions == 1


def test_fault_space_fallback_preset_maps_to_isolated_builder():
    preset = _resolve_experiment_preset("fault_space_fallback", mode="integrated")

    assert preset.name == "fault_space_fallback"
    assert preset.workflow_builder == "run_experiment_integrated:build_integrated_workflow_fault_space_fallback"
    assert preset.analysis_strategy == AnalysisStrategy.GRAPH_ONLY.value
    assert preset.max_revisions == 1


def test_fault_space_neighborhood_preset_maps_to_isolated_builder():
    preset = _resolve_experiment_preset("fault_space_neighborhood", mode="integrated")

    assert preset.name == "fault_space_neighborhood"
    assert preset.workflow_builder == "run_experiment_integrated:build_integrated_workflow_fault_space_neighborhood"
    assert preset.analysis_strategy == AnalysisStrategy.GRAPH_ONLY.value
    assert preset.max_revisions == 1


def test_fault_space_neighborhood_context_preset_maps_to_same_builder():
    preset = _resolve_experiment_preset("fault_space_neighborhood_context", mode="integrated")

    assert preset.name == "fault_space_neighborhood_context"
    assert preset.workflow_builder == "run_experiment_integrated:build_integrated_workflow_fault_space_neighborhood"
    assert preset.analysis_strategy == AnalysisStrategy.GRAPH_ONLY.value
    assert preset.max_revisions == 1


def test_fault_space_neighborhood_retrieval_preset_maps_to_same_builder():
    preset = _resolve_experiment_preset("fault_space_neighborhood_retrieval", mode="integrated")

    assert preset.name == "fault_space_neighborhood_retrieval"
    assert preset.workflow_builder == "run_experiment_integrated:build_integrated_workflow_fault_space_neighborhood"
    assert preset.analysis_strategy == AnalysisStrategy.GRAPH_ONLY.value
    assert preset.max_revisions == 1


def test_semantics_contract_prompt_preset_maps_to_same_builder():
    preset = _resolve_experiment_preset("semantics_contract_prompt", mode="integrated")

    assert preset.name == "semantics_contract_prompt"
    assert preset.workflow_builder == "run_experiment_integrated:build_integrated_workflow_fault_space_neighborhood"
    assert preset.analysis_strategy == AnalysisStrategy.GRAPH_ONLY.value
    assert preset.max_revisions == 1


def test_semantics_contract_rerank_preset_maps_to_dedicated_builder():
    preset = _resolve_experiment_preset("semantics_contract_rerank", mode="integrated")

    assert preset.name == "semantics_contract_rerank"
    assert preset.workflow_builder == "run_experiment_integrated:build_integrated_workflow_semantics_contract_rerank"
    assert preset.analysis_strategy == AnalysisStrategy.GRAPH_ONLY.value
    assert preset.max_revisions == 1


def test_fault_space_fallback_preset_matches_ablation1_except_builder_name():
    baseline = _resolve_experiment_preset("ablation1", mode="integrated")
    fallback = _resolve_experiment_preset("fault_space_fallback", mode="integrated")

    assert baseline.analysis_strategy == fallback.analysis_strategy
    assert baseline.max_revisions == fallback.max_revisions
    assert baseline.workflow_builder != fallback.workflow_builder


def test_fault_space_neighborhood_preset_matches_ablation1_except_builder_name():
    baseline = _resolve_experiment_preset("ablation1", mode="integrated")
    neighborhood = _resolve_experiment_preset("fault_space_neighborhood", mode="integrated")

    assert baseline.analysis_strategy == neighborhood.analysis_strategy
    assert baseline.max_revisions == neighborhood.max_revisions
    assert baseline.workflow_builder != neighborhood.workflow_builder


def test_extra_context_files_are_defined_for_stagec_cases_only():
    assert _extra_context_files_for_preset("fault_space_neighborhood_context", "django__django-15320") == [
        "django/db/models/expressions.py"
    ]
    assert _extra_context_files_for_preset("fault_space_neighborhood_context", "matplotlib__matplotlib-18869") == [
        "lib/matplotlib/__init__.py"
    ]
    assert _extra_context_files_for_preset("fault_space_neighborhood_context", "pallets__flask-4045") == [
        "src/flask/app.py"
    ]
    assert _extra_context_files_for_preset("fault_space_neighborhood_context", "psf__requests-3362") == [
        "requests/__init__.py"
    ]
    assert _extra_context_files_for_preset("fault_space_neighborhood_context", "sympy__sympy-13773") == [
        "sympy/matrices/matrices.py"
    ]
    assert _extra_context_files_for_preset("fault_space_neighborhood_context", "unknown") == []
    assert _extra_context_files_for_preset("fault_space_neighborhood", "django__django-15320") == []


def test_retrieved_context_files_follow_explicit_python_mentions(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    explicit = repo_root / "tests" / "test_blueprints.py"
    explicit.parent.mkdir(parents=True)
    explicit.write_text("print('ok')\n", encoding="utf-8")

    instance = {
        "problem_statement": "See tests/test_blueprints.py for the regression.",
        "hints_text": "",
        "FAIL_TO_PASS": [],
    }

    assert _retrieved_context_files_for_preset(
        "fault_space_neighborhood_retrieval",
        instance,
        repo_root,
    ) == ["tests/test_blueprints.py"]
    assert _retrieved_context_files_for_preset("fault_space_neighborhood_context", instance, repo_root) == []


def test_build_requirements_includes_semantic_contract_for_preset():
    instance = {
        "instance_id": "sympy__sympy-13773",
        "problem_statement": "Fix matrix multiplication semantics.",
        "hints_text": "",
        "FAIL_TO_PASS": ["test_matmul"],
    }

    requirements = _build_requirements(instance, preset_name="semantics_contract_prompt")

    assert "Semantic repair contract:" in requirements
    assert "NotImplemented" in requirements
    assert "test_matmul" in requirements


def test_build_requirements_includes_added_semantic_contract_cases():
    django_instance = {
        "instance_id": "django__django-11620",
        "problem_statement": "Fix technical 404 converter behavior.",
        "hints_text": "",
        "FAIL_TO_PASS": ["test_technical_404_converter_raise_404"],
    }
    sympy_instance = {
        "instance_id": "sympy__sympy-11897",
        "problem_statement": "Fix Piecewise latex rendering.",
        "hints_text": "",
        "FAIL_TO_PASS": ["test_latex_Piecewise"],
    }

    django_requirements = _build_requirements(django_instance, preset_name="semantics_contract_prompt")
    sympy_requirements = _build_requirements(sympy_instance, preset_name="semantics_contract_prompt")

    assert "technical 404 response" in django_requirements
    assert "test_latex_Piecewise" in sympy_requirements
    assert "Piecewise formatting semantics" in sympy_requirements


def test_build_requirements_omits_semantic_contract_for_default_preset():
    instance = {
        "instance_id": "sympy__sympy-13773",
        "problem_statement": "Fix matrix multiplication semantics.",
        "hints_text": "",
        "FAIL_TO_PASS": ["test_matmul"],
    }

    requirements = _build_requirements(instance, preset_name="default")

    assert "Semantic repair contract:" not in requirements


def test_non_default_preset_requires_integrated_mode():
    try:
        _resolve_experiment_preset("ablation1", mode="traditional")
    except ValueError as exc:
        assert "requires --mode integrated" in str(exc)
    else:
        raise AssertionError("expected integrated-mode validation error")


def test_bounded_fallback_target_set_keeps_primary_targets_when_disabled():
    targets, meta = _bounded_fallback_target_set(
        ["ForeignKey.check"],
        repair_brief={
            "related_symbols": [
                "ManyToManyField._check_relationship_model",
                "ForeignKey.check",
                "Model._meta",
            ]
        },
        enabled=False,
    )

    assert targets == ["ForeignKey.check"]
    assert meta["entered_fallback"] is False
    assert meta["fallback_reason"] == "fallback_disabled"
    assert meta["alternative_targets"] == [
        "ManyToManyField._check_relationship_model",
        "ForeignKey.check",
        "Model._meta",
    ]


def test_bounded_fallback_target_set_appends_related_symbols_with_cap():
    targets, meta = _bounded_fallback_target_set(
        ["ForeignKey.check"],
        repair_brief={
            "related_symbols": [
                "ManyToManyField._check_relationship_model",
                "Model._meta",
                "Field.clean",
            ]
        },
        enabled=True,
    )

    assert targets == [
        "ForeignKey.check",
        "ManyToManyField._check_relationship_model",
        "Model._meta",
    ]
    assert meta["entered_fallback"] is True
    assert meta["fallback_added_targets"] == [
        "ManyToManyField._check_relationship_model",
        "Model._meta",
    ]
    assert meta["alternative_targets"] == [
        "ManyToManyField._check_relationship_model",
        "Model._meta",
        "Field.clean",
    ]


def test_summarize_fallback_usage_reports_fallback_hit():
    usage = _summarize_fallback_usage(
        effective_change=True,
        target_hit_info={"target_hit": True},
        primary_target_hit_info={"target_hit": False},
        fallback_target_hit_info={
            "target_hit": True,
            "target_hit_rate": 1.0,
            "target_symbols_total": 1,
            "target_symbols_hit": 1,
        },
        relocalization_meta={
            "fallback_enabled": True,
            "entered_fallback": True,
            "primary_targets": ["EnumSerializer.serialize"],
            "expanded_targets": ["EnumSerializer.serialize", "items"],
            "fallback_added_targets": ["items"],
        },
    )

    assert usage["selected_target_source"] == "fallback"
    assert usage["fallback_target_hit"] is True
    assert usage["fallback_would_have_triggered_but_not_used_reason"] is None


def test_summarize_fallback_usage_reports_non_usage_reason():
    usage = _summarize_fallback_usage(
        effective_change=True,
        target_hit_info={"target_hit": False},
        primary_target_hit_info={"target_hit": False},
        fallback_target_hit_info={
            "target_hit": False,
            "target_hit_rate": 0.0,
            "target_symbols_total": 1,
            "target_symbols_hit": 0,
        },
        relocalization_meta={
            "fallback_enabled": True,
            "entered_fallback": True,
            "primary_targets": ["EnumSerializer.serialize"],
            "expanded_targets": ["EnumSerializer.serialize", "items"],
            "fallback_added_targets": ["items"],
        },
    )

    assert usage["selected_target_source"] == "none"
    assert usage["fallback_would_have_triggered_but_not_used_reason"] == "fallback_targets_not_hit"


def test_file_local_neighborhood_target_set_uses_anchor_symbol_from_same_file():
    files = {
        "django/db/migrations/serializer.py": """
class EnumSerializer:
    def serialize(self):
        return self.value.name

    def deserialize(self):
        return self.value

def helper_function():
    return None
"""
    }

    targets, meta = _file_local_neighborhood_target_set(
        files,
        ["items"],
        repair_brief={
            "target_symbol": "EnumSerializer.serialize",
            "related_symbols": [],
        },
        enabled=True,
    )

    assert meta["target_expansion_mode"] == "file_local_neighborhood"
    assert meta["entered_fallback"] is True
    assert meta["expansion_anchor_file"] == "django/db/migrations/serializer.py"
    assert meta["expansion_anchor_symbol"] == "EnumSerializer.serialize"
    assert "EnumSerializer.serialize" in targets
    assert "EnumSerializer.serialize" in meta["fallback_added_targets"]
