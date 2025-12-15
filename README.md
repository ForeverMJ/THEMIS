# THEMIS: Advanced Code Analysis and Spec-Change Consistency System

A comprehensive code analysis system that combines LLM-driven semantic understanding with structural graph analysis. The system features an Enhanced GraphManager for requirement-code consistency checking and an Advanced Code Analysis system for intelligent bug detection and classification.

## 🚀 New: Advanced Code Analysis System

The latest addition to THEMIS is a sophisticated LLM-powered code analysis system that provides:

- 🧠 **Intelligent Bug Classification** - Automatically categorizes issues and selects optimal analysis strategies
- 📚 **Context Enhancement** - Provides rich code context and domain knowledge to LLMs
- 🎯 **Pattern Learning** - Learns from successful cases to improve future analysis
- 🔄 **Multi-Round Reasoning** - Uses iterative verification to improve accuracy
- 📊 **Confidence Scoring** - Provides reliability metrics for all analysis results

### Quick Start with Advanced Analysis

```bash
# Demo mode (no API key required)
python run_demo_mode.py

# Quick test (requires OpenAI API key)
python run_quick_test.py

# Comprehensive analysis
python run_experiment_advanced.py
```

## System Architecture

The system now includes two complementary analysis approaches:

1. **Enhanced GraphManager** - Structural analysis using AST-derived graphs with requirement mapping
2. **Advanced Code Analysis** - LLM-driven semantic understanding with intelligent reasoning

Both systems can work independently or in integrated mode for maximum effectiveness.

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

### Advanced Code Analysis (New)
```bash
python run_demo_mode.py           # Demo mode - no API key needed
python run_quick_test.py          # Quick test of advanced analysis
python run_experiment_advanced.py # Comprehensive advanced analysis
python demo_integration.py        # System integration demo
```

### Traditional Analysis
```bash
python run_experiment.py          # 提案手法（KG + Judge ループ）
python run_experiment_enhanced.py # Enhanced GraphManager analysis
python run_baseline.py            # ベースライン1: 直接プロンプト
python run_baseline_reflexion.py  # ベースライン2: Reflexion ループ
```

## テスト
```bash
pytest -q
```

## Configuration

### API Keys Setup
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
OPENAI_API_KEY=your_actual_api_key_here
```

### Analysis Strategies

The Advanced Code Analysis system supports multiple strategies:

- **AUTO_SELECT**: Automatically chooses the best strategy based on problem type
- **ADVANCED_ONLY**: Pure LLM-driven semantic analysis
- **GRAPH_ONLY**: Structure-based analysis using Enhanced GraphManager
- **INTEGRATED**: Combines both LLM and graph analysis for comprehensive results

## Documentation

- 📖 [Advanced Analysis Guide](ADVANCED_ANALYSIS_GUIDE.md) - Comprehensive usage guide
- 🧪 [Experiment Usage](EXPERIMENT_USAGE.md) - Detailed experiment instructions
- 🔧 [Integration Summary](INTEGRATION_SUMMARY.md) - System integration details

## メモ
- LLM モデル名は `src/main.py` や `src/baselines/*` の `build_workflow`/`build_app` で変更できます。
- Judge はベースライン KG と修正後 KG の両方を参照し、VIOLATES エッジがあればハードチェックで即レポートします。ソフトチェックは情報不足の場合に保守的な提案を返すことがあるので、必要に応じてプロンプトや閾値を調整してください。
- The Advanced Code Analysis system provides intelligent semantic understanding and can work with or without API keys (demo mode available).
