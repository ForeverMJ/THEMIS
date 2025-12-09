# THEMIS: Spec-Change Consistency Agents

LangGraph を使ったマルチエージェントで、コードと要件の矛盾を検出・修正する実験用プロジェクトです。NetworkX で AST 由来の構造グラフを作り、LLM で要件ノードを付与し、Judge が矛盾をレポートします。ベースラインとして単純プロンプトと Reflexion も用意しています。

## ディレクトリ
- `src/state.py` : AgentState 定義
- `src/graph_manager.py` : AST 解析＋要件ノード付与
- `src/agents/developer.py` : LLM でコード修正
- `src/agents/judge.py` : KG から矛盾検出
- `src/main.py` : LangGraph ワークフロー
- `src/baselines/` : vanilla / reflexion ベースライン
- `run_experiment.py` : 提案手法の実行スクリプト
- `run_baseline.py` : vanilla 実行
- `run_baseline_reflexion.py` : reflexion 実行
- `tests/` : GraphManager のテスト

## セットアップ
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

環境変数に OpenAI API キーを設定してください:
```bash
setx OPENAI_API_KEY "your_key_here"  # Windows 永続
# または一時的に
set OPENAI_API_KEY=your_key_here
```

## 実行例
```bash
python run_experiment.py          # 提案手法（KG + Judge ループ）
python run_baseline.py            # ベースライン1: 直接プロンプト
python run_baseline_reflexion.py  # ベースライン2: Reflexion ループ
```

## テスト
```bash
pytest -q
```

## メモ
- LLM モデル名は `src/main.py` や `src/baselines/*` の `build_workflow`/`build_app` で変更できます。
- Judge はベースライン KG と修正後 KG の両方を参照し、VIOLATES エッジがあればハードチェックで即レポートします。ソフトチェックは情報不足の場合に保守的な提案を返すことがあるので、必要に応じてプロンプトや閾値を調整してください。
