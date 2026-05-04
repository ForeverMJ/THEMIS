# SWE-bench Lite `cases1_100_work3` 评测记录

日期：2026-05-04

## 1. 评测对象

- predictions 文件：`predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases1_100_work3.jsonl`
- 模型：`gpt-5.1-codex-mini`
- 数据集：`SWE-bench/SWE-bench_Lite`

## 2. 实际使用命令

本机已安装 harness 的 CLI 与旧记录略有不同，因此按本机 `--help` 实际可用参数执行：

```bash
.venv/bin/python -m swebench.harness.run_evaluation \
  -p predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases1_100_work3.jsonl \
  -id cases1_100_work3_eval \
  -d SWE-bench/SWE-bench_Lite \
  -s test \
  --max_workers 4 \
  -t 900 \
  --report_dir work3
```

## 3. 关键产物

- harness 控制台日志：`work3/cases1_100_work3_eval.console.log`
- harness 汇总报告：`gpt-5.1-codex-mini.cases1_100_work3_eval.json`
- per-case 评测目录：`logs/run_evaluation/cases1_100_work3_eval/gpt-5.1-codex-mini/`

## 4. 最终结果摘要

从 `work3/cases1_100_work3_eval.console.log` 与 `gpt-5.1-codex-mini.cases1_100_work3_eval.json` 核实：

| 指标 | 数值 |
|---|---:|
| predictions 总数 | 100 |
| 非空 patch case | 89 |
| 空 patch case | 11 |
| 已完成官方评测的 case | 89 |
| 通过（resolved） | 19 |
| 未通过（unresolved） | 70 |
| harness 错误 | 0 |
| 总耗时 | 1:52:49 |

### 通过率

- 以 100 个 predictions 全部计：`19 / 100 = 19.0%`
- 只看 89 个实际进入 harness 的非空 patch case：`19 / 89 ≈ 21.35%`

## 5. 通过的 case（19）

1. `django__django-10914`
2. `django__django-11620`
3. `django__django-11848`
4. `django__django-12589`
5. `django__django-12856`
6. `django__django-13265`
7. `django__django-13447`
8. `django__django-13933`
9. `django__django-13964`
10. `django__django-15061`
11. `django__django-15814`
12. `django__django-16046`
13. `django__django-16595`
14. `psf__requests-2674`
15. `sympy__sympy-15346`
16. `sympy__sympy-18189`
17. `sympy__sympy-18698`
18. `sympy__sympy-20049`
19. `sympy__sympy-20154`

## 6. 空 patch case（11）

这 11 个 case 在 predictions 中 `model_patch` 为空，因此没有进入 89 个实际执行的 harness case：

1. `django__django-15320`
2. `sympy__sympy-24152`
3. `sympy__sympy-14396`
4. `matplotlib__matplotlib-23299`
5. `sympy__sympy-15609`
6. `sphinx-doc__sphinx-10451`
7. `sympy__sympy-18621`
8. `matplotlib__matplotlib-25442`
9. `sphinx-doc__sphinx-10325`
10. `sphinx-doc__sphinx-7686`
11. `sympy__sympy-13915`

## 7. 失败模式简析

对 89 个已完成评测的 case 做本地归类：

- 普通测试未通过：68
- 超时/挂起：1
  - `pytest-dev__pytest-5495`
- 安装或构建失败：1
  - `sympy__sympy-13031`
- patch apply 失败：0

也就是说，这一批未通过 case 主要不是 harness 本身报错，而是 patch 通过了应用阶段，但测试结果没有满足 benchmark 通过条件。

## 8. 通过 case 的仓库分布

- `django`: 13
- `sympy`: 5
- `psf/requests`: 1

## 9. 备注

1. 本次 harness 最终输出中出现的：
   - `Total instances: 300`
   - `Instances submitted: 100`
   是因为 SWE-bench Lite 全测试集大小为 300，而本次 predictions 只提交了其中 100 个 case。
2. `Instances completed: 89` 与 `Instances with empty patches: 11` 相加正好等于 100，和本次 predictions 范围一致。
3. 后续如果要逐个检查失败原因，优先看：
   - `logs/run_evaluation/cases1_100_work3_eval/gpt-5.1-codex-mini/<instance_id>/report.json`
   - `logs/run_evaluation/cases1_100_work3_eval/gpt-5.1-codex-mini/<instance_id>/test_output.txt`
   - `logs/run_evaluation/cases1_100_work3_eval/gpt-5.1-codex-mini/<instance_id>/run_instance.log`
