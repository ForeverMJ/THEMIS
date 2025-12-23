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

# Three analysis modes:

# 1. Traditional workflow (KG → Developer → Judge)
python run_experiment_enhanced.py

# 2. Advanced LLM analysis only (semantic understanding)
python run_experiment_advanced.py

# 3. Integrated workflow (Advanced Analysis → KG → Developer → Judge) ⭐ Recommended
python run_experiment_integrated.py

# Compare all three modes side-by-side
python compare_workflows.py
```

**Analysis Modes Explained:**

1. **Traditional Enhanced** (`run_experiment_enhanced.py`)
   - Uses Enhanced GraphManager for structural analysis
   - KG construction → Developer revision → Judge evaluation
   - Fast, rule-based approach

2. **Advanced Analysis** (`run_experiment_advanced.py`)
   - LLM-driven semantic understanding
   - Bug classification, concept mapping, pattern learning
   - Provides insights and recommendations (no automatic code revision)

3. **Integrated Workflow** (`run_experiment_integrated.py`) ⭐
   - **Best of both worlds**
   - Step 1: Advanced LLM analysis for semantic understanding
   - Step 2: KG construction enriched with LLM insights
   - Step 3: Developer uses both semantic and structural insights
   - Step 4: Judge validates the revised code
   - Combines intelligent understanding with automated revision

### 🔄 Model Switching (New!)

Easily switch between different LLM models with a single command:

```bash
# View available models
python switch_model.py --list

# Switch to GPT-4o (recommended)
python switch_model.py gpt-4o

# Switch to Claude 3.5 Sonnet
python switch_model.py claude-3.5-sonnet

# Switch to GPT-3.5 (cost-effective)
python switch_model.py gpt-3.5-turbo
```

**Supported Models:**
- OpenAI: `gpt-4`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- Anthropic: `claude-3.5-sonnet`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

**Documentation:**
- Quick Reference: `QUICK_MODEL_SWITCH.md`
- Detailed Guide: `MODEL_SWITCHING_GUIDE.md`
- 中文说明: `模型切换使用说明.md`

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


## 🔧 Troubleshooting

### Empty LLM Response

If you see:
```
Empty LLM response for conceptual matching; skipping.
```

**Cause**: Using `gpt-5-mini` which has unstable Responses API, or API key has quotes.

**Solution**:

1. **Check .env file** - Remove quotes from API key:
   ```bash
   # Wrong ❌
   OPENAI_API_KEY="sk-proj-..."
   
   # Correct ✅
   OPENAI_API_KEY=sk-proj-...
   ```

2. **Switch to stable model** (recommended):
   ```bash
   python switch_model.py gpt-4o-mini
   ```

See `GPT5_ISSUES.md` for detailed information.

### JSON Parsing Errors

If you encounter errors like:
```
Could not parse LLM response for conceptual matching: Expecting ',' delimiter
```

**Solution**: The system now has robust JSON parsing with automatic cleanup. If errors persist:

1. **Switch to a more stable model**:
   ```bash
   python switch_model.py gpt-4o
   ```

2. **Use traditional mode** (doesn't rely on LLM JSON parsing):
   ```bash
   python run_experiment_enhanced.py
   ```

3. **Check logs**: Errors are logged but won't interrupt the workflow

See `修复说明.md` for detailed fix information.

### API Key Issues

If you see "API key not found" warnings:

1. **Check your .env file**:
   ```bash
   # For OpenAI
   OPENAI_API_KEY=sk-proj-...
   
   # For Anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. **Verify the key is loaded**:
   ```bash
   python test_model_switch.py
   ```

### Model Not Working

If a specific model isn't working:

1. **List available models**:
   ```bash
   python switch_model.py --list
   ```

2. **Try a different model**:
   ```bash
   python switch_model.py gpt-3.5-turbo
   ```

3. **Check model compatibility**: Some models may not support all features

### Slow Performance

If analysis is taking too long:

1. **Use faster models**:
   ```bash
   python switch_model.py gpt-3.5-turbo  # Fastest
   python switch_model.py gpt-4o-mini    # Fast and good quality
   ```

2. **Use traditional mode** (faster than integrated):
   ```bash
   python run_experiment_enhanced.py
   ```

3. **Reduce context size**: Edit `src/advanced_code_analysis/config.py`:
   ```python
   max_context_tokens = 4000  # Reduce from 8000
   ```

### Comparison

For detailed workflow comparison and recommendations, see:
- `WORKFLOW_COMPARISON.md` - Detailed comparison of all three modes
- `运行指南.txt` - Quick reference guide (Chinese)
- `compare_workflows.py` - Run all modes and compare results

## 📚 Additional Resources

- **Model Switching**: `MODEL_SWITCHING_GUIDE.md`, `QUICK_MODEL_SWITCH.md`
- **Workflow Comparison**: `WORKFLOW_COMPARISON.md`
- **Fix Documentation**: `修复说明.md`, `最终修复总结.md`
- **Quick Reference**: `运行指南.txt`, `快速切换模型.txt`
- **Troubleshooting**: `GPT5_ISSUES.md`
- **Environment Variables**: `ENV_VARIABLES_GUIDE.md`, `ENV_FLOW_DIAGRAM.txt`, `CONFIG_PRIORITY.md`
