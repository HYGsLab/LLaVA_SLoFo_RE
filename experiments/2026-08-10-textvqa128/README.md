# 08-10 TextVQA显存消融

本目录保存2026-08-10完成的TextVQA固定128题实验。

- 主报告：`reports/SLoFo_08_10_TextVQA128_显存消融与论文默认基线报告.md`
- 因子拆分报告：`reports/SLoFo_08_10_TextVQA128_因子拆分消融报告.md`
- 总对照：`analysis/ablation_comparison.json`
- 因子汇总：`analysis/factorial/textvqa_factor_ablation.json`
- 低显存8-Token结果：`results/optimized_8token_minmax_topk128/`
- 论文公式直译单Token结果：`results/paper_1token_raw_single128/`
- 固定问题清单：`manifests/textvqa_fixed_128.json`
- 本轮代码：仓库根目录的 `slofo/`、`scripts/` 与 `tests/`
- 环境与哈希：`metadata/`
- 图表：`figures/`

为控制仓库体积，这里只提交固定 manifest、逐题聚合答案、批次汇总、统计 CSV/JSON
和 SVG 图表；逐题可视化大图、NumPy 热图、模型权重、数据集原图及服务器日志未提交。

核心结论：低显存路径最大Scan allocated显存为`19,086.9 MiB`，相对旧路径`24,665.4 MiB`降低`22.62%`，且逐题回答、bbox和候选rank均完全一致。

因子拆分进一步表明：raw融合下1/8 Token完全等价；min-max会激活语义分支但在本128题上降低双图得分；固定`8 Token + min-max`时，Top-k带来`+1.33`个百分点、3题改善且0题退化。
