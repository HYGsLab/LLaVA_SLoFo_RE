# TextVQA 1,024 题固定清单因子实验

本实验用于复核 128 题消融中观察到的趋势，且不把 1,024 题结果冒充官方全量 benchmark。

## 固定数据

- 数据源：TextVQA 0.5.1 validation split；
- 清单：`manifests/textvqa_nested_1024.json`；
- 数量：1,024 条问答，942 张唯一图片；
- 嵌套关系：前 512 条与 `experiments/2026-08-09/manifests/textvqa_subset_512.json` 完全一致；
- 新增抽样 seed：`20260813`；
- 清单 SHA-256：`60987130ae7dd4d274a4240d08b45ea8e28d4ad5fb1d035c7a4d216e670e353e`。

## 四组配置

| 配置 | 语义 rollout | 融合归一化 | 候选框 | 目的 |
|---|---:|---|---:|---|
| A | 1 Token | raw / none | 单框 | 论文公式直译基线 |
| B | 1 Token | min-max | 单框 | 单独观察 min-max |
| D | 8 Token | min-max | 单框 | 单独观察多 Token |
| E | 8 Token | min-max | top-k=5 | 观察候选框重排 |

四组均固定 `selective_hook`、原图坐标系、mean Token 聚合、`max_new_tokens=16` 和 LLaVA-v1.5-7B FP16。任务通过 Slurm 串行提交，每次只申请一张经过空卡预检的 24 GiB RTX 4090。

## 运行与验收

- 单组脚本：`slurm/textvqa1024_factor.sbatch`；
- 串行提交：`slurm/submit_textvqa1024_factors.sh`；
- 调度输出会记录代码提交、LLaVA 提交、清单哈希、节点、GPU 空卡预检和起止时间；
- 每组结果必须包含 1,024 条逐题结果和对应的 `batch_summary.json`；
- 四组结束后用仓库分析脚本做配对比较，再决定是否扩大到官方全量 benchmark。
