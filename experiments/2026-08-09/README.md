# 2026-08-09 TextVQA / GQA / POPE 实验交付

先阅读：[SLoFo_08_09_TextVQA_GQA_POPE_实验报告.md](SLoFo_08_09_TextVQA_GQA_POPE_实验报告.md)

- `manifests/`：固定问题清单和抽样校验。
- `results/`：三个 benchmark 的逐题聚合答案与批次耗时。
- `analysis/`：逐题 CSV、统计 JSON 和 SVG 图表。
- `metadata/`：论文对照数字、运行环境和 SHA-256。

GitHub 精简版未包含样例大图、压缩包、数据集原图、模型权重或服务器日志；
逐题聚合答案与固定 manifest 足以复核汇总统计。

这些分数来自固定开发子集，不是全量官方提交分数。服务器原始数据与完整运行目录位于：

`/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo/benchmarks/official/`
