# TextVQA validation 5,000 题全量验证

本实验承接 1,024 题四配置因子消融，只运行两条必要路径：

- A：1 Token + raw + 单框，作为论文公式直译基线；
- E：8 Token + min-max + top-k=5，作为当前完整工程方案。

固定清单 `manifests/textvqa_full_5000.json` 包含 TextVQA 0.5.1 validation 的全部 5,000 条问答和 3,166 张唯一图片；前 1,024 条与上一轮清单完全一致。仓库规范化 LF 文件的 SHA-256 为 `af133e3dcc2e4096f122dcdff4560fdfe95e745c2837f60f3f375cbcad499b90`。

为保持嵌套清单不变，前序 512 题沿用 LLaVA TextVQA OCR question file 的
`Reference OCR token` 辅助提示，另外 4,488 题使用“原始问题 + 单词或短语作答”
提示。两组 A/E 对同一道题使用相同提示，因此配对差值是公平的；但混合提示下的
绝对准确率不应直接当作统一纯问题提示的官方排行榜成绩。全量分析会同时报告
`ocr_assisted_prompt` 与 `plain_question_prompt` 两个分层结果。

本阶段仍是当前工程实现的官方 split 全量验证。由于目标论文未公开代码，不能把结果表述为作者实现的完全复现。

## Slurm 运行

服务器入口脚本为 `slurm/submit_textvqa5000_compare.sh`。脚本先提交 A，再以
`afterok` 依赖提交 E，保证两组作业串行占用一张真实 24 GiB RTX 3090。计算节点
内部还会运行 `slurm/preflight_empty_gpu.sh`；如果分配到的显卡并非空卡，作业会
主动退出。

批处理程序以 `question_id` 检查已有结果，因此中断后可以安全重提：已经完成的
题目会跳过，只补跑缺失题目。A 在首次运行到 203 题时按计划暂停，第二次提交从
第 204 题继续。报告总耗时需要同时注明这是一次断点续跑，不能把第二次作业的
wall time 当成 A 的完整一次性运行时间。

## 全量对照分析

两组各完成 5,000 题后，在已激活的 LLaVA 环境运行：

```bash
python scripts/analyze_textvqa_full_compare.py \
  --manifest experiments/2026-08-13-textvqa5000/manifests/textvqa_full_5000.json \
  --baseline /path/to/A_1token_raw_single/benchmark_answers.jsonl \
  --optimized /path/to/E_8token_minmax_topk5/benchmark_answers.jsonl \
  --output /path/to/textvqa5000_AE_compare.json \
  --case-output /path/to/textvqa5000_AE_cases.csv
```

这里的 E−A 是“完整工程方案相对公式直译基线”的端到端比较，三个因素同时改变：

1. 语义 rollout 从 1 Token 增加到 8 Token；
2. 融合归一化从 raw 改为 min-max；
3. 候选框从单框改为 top-k=5 重排。

因此 E−A 不能解释成某一个单独因子的因果贡献；单因素结论应引用 1,024 题的
A/B/D/E 配对消融。

## 完成状态与核心结果

两组作业均在真实 24 GiB RTX 3090 空卡上完成，各输出 5,000 条结果：

| 分支 | A | E | E−A |
|---|---:|---:|---:|
| 原图 | 47.152% | 47.152% | 0.000 pp |
| 裁切图 | 41.234% | 50.092% | +8.858 pp |
| 双图联合 | 52.920% | 56.406% | +3.486 pp |
| Focus | 53.004% | 56.536% | +3.532 pp |

Focus 的配对 95% CI 为 `[+2.684, +4.380]`，400 题改善、212 题退化；E 的
最大 Scan 峰值为 19.27 GiB。Top-k 在 E 内部把双图得分从 55.628% 提高到
56.406%（+0.778 pp），实际选择 rank>1 候选框 577 次。

完整解释、收敛趋势、提示词分层和资源分析见
`SLoFo_TextVQA5000_全量对照实验报告_2026-08-14.md`。仓库同时保留：

- A/E 原始 `benchmark_answers.jsonl` 与 `batch_summary.json`；
- 5,000 题配对统计 JSON 和逐题 CSV；
- 改善、退化、同分改写和 top-k 重排复查清单；
- 四张报告 SVG 及再生成脚本；
- A/E 前 1,024 题与独立历史实验的一致性检查。
