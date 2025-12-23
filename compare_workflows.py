#!/usr/bin/env python3
"""
工作流对比脚本

一次运行所有三种分析模式，对比结果：
1. 传统增强模式 (Traditional Enhanced)
2. 高级分析模式 (Advanced Analysis)
3. 集成工作流 (Integrated Workflow)

使用方法:
    python compare_workflows.py
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any

import networkx as nx

# Import all three workflows
from src.main_enhanced import build_workflow as build_traditional_workflow
from src.state import AgentState


def load_text(path: Path) -> str:
    """Load text file with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def run_traditional_enhanced(requirements: str, source_code: str, target_filename: str) -> Dict[str, Any]:
    """运行传统增强模式"""
    print("\n" + "="*80)
    print("🔄 模式 1: 传统增强模式 (Traditional Enhanced)")
    print("="*80)
    
    workflow = build_traditional_workflow()
    app = workflow.compile()

    initial_state: AgentState = {
        "messages": [],
        "files": {target_filename: source_code},
        "requirements": requirements,
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
    }

    start_time = time.time()
    try:
        final_state = app.invoke(initial_state, config={"recursion_limit": 50})
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "processing_time": processing_time,
            "final_state": final_state,
            "conflict_report": final_state.get("conflict_report"),
            "final_code": final_state.get("files", {}).get(target_filename, ""),
            "revision_count": final_state.get("revision_count", 0),
            "analysis_report": final_state.get("analysis_report", {}),
        }
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ 错误: {e}")
        return {
            "success": False,
            "processing_time": processing_time,
            "error": str(e)
        }


async def run_advanced_analysis(requirements: str, source_code: str, target_filename: str) -> Dict[str, Any]:
    """运行高级分析模式"""
    print("\n" + "="*80)
    print("🧠 模式 2: 高级分析模式 (Advanced Analysis)")
    print("="*80)
    
    try:
        from src.enhanced_graph_adapter import EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions
        
        adapter = EnhancedGraphAdapter()
        
        # Create temporary file
        temp_file = Path(target_filename)
        temp_file.write_text(source_code, encoding='utf-8')
        
        try:
            options = AnalysisOptions(
                strategy=AnalysisStrategy.AUTO_SELECT,
                confidence_threshold=0.6,
                include_requirements=True,
                debug_mode=False,
                max_context_tokens=8000
            )
            
            start_time = time.time()
            result = await adapter.analyze(
                issue_text=requirements,
                target_files=[target_filename],
                requirements_text=None,
                options=options
            )
            processing_time = time.time() - start_time
            
            return {
                "success": result.success,
                "processing_time": processing_time,
                "result": result,
                "findings": result.primary_findings,
                "recommendations": result.recommendations,
                "confidence": result.confidence_score,
                "strategy": result.strategy_used.value,
            }
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        return {
            "success": False,
            "processing_time": 0,
            "error": str(e)
        }


def run_integrated_workflow(requirements: str, source_code: str, target_filename: str) -> Dict[str, Any]:
    """运行集成工作流"""
    print("\n" + "="*80)
    print("✨ 模式 3: 集成工作流 (Integrated Workflow)")
    print("="*80)
    
    try:
        # Import here to avoid circular dependencies
        import sys
        import importlib.util
        
        # Load run_experiment_integrated module
        spec = importlib.util.spec_from_file_location(
            "run_experiment_integrated",
            Path(__file__).parent / "run_experiment_integrated.py"
        )
        integrated_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integrated_module)
        
        workflow = integrated_module.build_integrated_workflow()
        app = workflow.compile()

        initial_state: AgentState = {
            "messages": [],
            "files": {target_filename: source_code},
            "requirements": requirements,
            "knowledge_graph": nx.DiGraph(),
            "baseline_graph": None,
            "conflict_report": None,
            "revision_count": 0,
            "advanced_analysis": None,
            "analysis_report": None,
        }

        start_time = time.time()
        final_state = app.invoke(initial_state, config={"recursion_limit": 50})
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "processing_time": processing_time,
            "final_state": final_state,
            "conflict_report": final_state.get("conflict_report"),
            "final_code": final_state.get("files", {}).get(target_filename, ""),
            "revision_count": final_state.get("revision_count", 0),
            "advanced_analysis": final_state.get("advanced_analysis", {}),
            "analysis_report": final_state.get("analysis_report", {}),
        }
    except Exception as e:
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "processing_time": processing_time,
            "error": str(e)
        }


def print_comparison_table(results: Dict[str, Dict[str, Any]], original_code: str):
    """打印对比表格"""
    print("\n" + "="*80)
    print("📊 工作流对比结果")
    print("="*80)
    
    # Success status
    print(f"\n✅ 执行状态:")
    print(f"   传统增强:  {'✅ 成功' if results['traditional']['success'] else '❌ 失败'}")
    print(f"   高级分析:  {'✅ 成功' if results['advanced']['success'] else '❌ 失败'}")
    print(f"   集成工作流: {'✅ 成功' if results['integrated']['success'] else '❌ 失败'}")
    
    # Processing time
    print(f"\n⏱️  处理时间:")
    print(f"   传统增强:  {results['traditional']['processing_time']:.2f}秒")
    print(f"   高级分析:  {results['advanced']['processing_time']:.2f}秒")
    print(f"   集成工作流: {results['integrated']['processing_time']:.2f}秒")
    
    # Code changes
    print(f"\n📝 代码修改:")
    trad_changed = results['traditional'].get('final_code', '') != original_code
    integ_changed = results['integrated'].get('final_code', '') != original_code
    
    print(f"   传统增强:  {'✅ 已修改' if trad_changed else '❌ 未修改'}")
    print(f"   高级分析:  ⚠️  仅提供建议 (不修改代码)")
    print(f"   集成工作流: {'✅ 已修改' if integ_changed else '❌ 未修改'}")
    
    # Analysis depth
    print(f"\n🔍 分析深度:")
    
    # Traditional
    trad_report = results['traditional'].get('analysis_report', {})
    if trad_report:
        stats = trad_report.get('graph_statistics', {})
        violations = trad_report.get('violation_report', {})
        print(f"   传统增强:")
        print(f"      • 图节点: {stats.get('total_nodes', 0)}")
        print(f"      • 违规数: {violations.get('total_violations', 0)}")
    
    # Advanced
    if results['advanced']['success']:
        adv_result = results['advanced']
        print(f"   高级分析:")
        print(f"      • 策略: {adv_result.get('strategy', 'N/A')}")
        print(f"      • 置信度: {adv_result.get('confidence', 0):.2f}")
        print(f"      • 发现数: {len(adv_result.get('findings', []))}")
        print(f"      • 建议数: {len(adv_result.get('recommendations', []))}")
    
    # Integrated
    if results['integrated']['success']:
        integ_adv = results['integrated'].get('advanced_analysis', {})
        integ_report = results['integrated'].get('analysis_report', {})
        
        print(f"   集成工作流:")
        if integ_adv:
            print(f"      • LLM策略: {integ_adv.get('strategy', 'N/A')}")
            print(f"      • LLM置信度: {integ_adv.get('confidence', 0):.2f}")
            print(f"      • LLM发现: {len(integ_adv.get('findings', []))}")
        if integ_report:
            stats = integ_report.get('graph_statistics', {})
            violations = integ_report.get('violation_report', {})
            print(f"      • 图节点: {stats.get('total_nodes', 0)}")
            print(f"      • 违规数: {violations.get('total_violations', 0)}")
    
    # Conflicts
    print(f"\n⚖️  冲突检测:")
    trad_conflict = results['traditional'].get('conflict_report')
    integ_conflict = results['integrated'].get('conflict_report')
    
    print(f"   传统增强:  {'⚠️ 有冲突' if trad_conflict else '✅ 无冲突'}")
    print(f"   高级分析:  N/A (不执行验证)")
    print(f"   集成工作流: {'⚠️ 有冲突' if integ_conflict else '✅ 无冲突'}")
    
    # Recommendations
    print(f"\n💡 推荐使用:")
    print(f"   • 快速开发: 传统增强 (最快)")
    print(f"   • 深入分析: 高级分析 (最详细)")
    print(f"   • 生产环境: 集成工作流 (最全面) ⭐")


async def main():
    """主函数"""
    print("🚀 工作流对比测试")
    print("="*80)
    print("将运行三种分析模式并对比结果:")
    print("  1. 传统增强模式")
    print("  2. 高级分析模式")
    print("  3. 集成工作流")
    print("="*80)
    
    # Load experiment data
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"

    requirements = load_text(req_path)
    source_code = load_text(code_path)
    target_filename = "target_file.py"

    print(f"\n📋 实验数据:")
    print(f"   需求长度: {len(requirements)} 字符")
    print(f"   代码长度: {len(source_code)} 字符")
    
    # Run all three workflows
    results = {}
    
    # 1. Traditional Enhanced
    results['traditional'] = run_traditional_enhanced(requirements, source_code, target_filename)
    
    # 2. Advanced Analysis
    results['advanced'] = await run_advanced_analysis(requirements, source_code, target_filename)
    
    # 3. Integrated Workflow
    results['integrated'] = run_integrated_workflow(requirements, source_code, target_filename)
    
    # Print comparison
    print_comparison_table(results, source_code)
    
    # Detailed findings
    print(f"\n" + "="*80)
    print("📋 详细发现")
    print("="*80)
    
    if results['advanced']['success']:
        print(f"\n🧠 高级分析发现:")
        for i, finding in enumerate(results['advanced'].get('findings', [])[:5], 1):
            print(f"   {i}. {finding}")
        
        print(f"\n💡 高级分析建议:")
        for i, rec in enumerate(results['advanced'].get('recommendations', [])[:5], 1):
            print(f"   {i}. {rec}")
    
    if results['integrated']['success']:
        integ_adv = results['integrated'].get('advanced_analysis', {})
        if integ_adv:
            print(f"\n✨ 集成工作流 - LLM 洞察:")
            for i, finding in enumerate(integ_adv.get('findings', [])[:3], 1):
                print(f"   {i}. {finding}")
    
    print(f"\n" + "="*80)
    print("✅ 对比测试完成!")
    print("="*80)
    print(f"\n查看 WORKFLOW_COMPARISON.md 了解更多详情")


if __name__ == "__main__":
    asyncio.run(main())
