# LLaVA SLoFo RE

这是一个面向代码学习与论文复现的**非官方实验仓库**。项目在
LLaVA-v1.5-7B 上独立重建 SLoFo 的
`Scan → Locate → 原图/裁切图双图输入 → Focus 四阶段剪枝`，并保留可重复的
实验清单、逐题答案和汇总统计。

> SLoFo 作者未公开代码。本仓库的“论文公式直译基线”来自论文公式与文字，
> 不代表作者官方实现；min-max、多 Token 与 Top-k 均明确作为实现补充或实验扩展。

## 当前进度

- Scan：梯度加权语义显著性、PCA 结构显著性与 SSIM 融合；
- Locate：论文式多尺度单框定位，以及可选的 Top-k 候选生成、NMS、重排和安全回退；
- 双图输入：原图和裁切图作为两个独立图像张量参与同一次生成；
- Focus：32 层划分为四阶段，原图视觉 Token 按 `576→288→144→72` 真实缩短；
- 数值修复：仅将 `attention × gradient` 转为 FP32，避免 FP16 下溢；
- 低显存 Scan：选择性 Hook 只捕获语义层 14 的注意力与结构层 7 的隐藏状态；
- 评测：完成 TextVQA、GQA、POPE 共 1,624 题固定开发子集，以及 TextVQA 128 题五组因子消融。

## 关键结果

### 标准任务固定开发子集（2026-08-09）

| Benchmark | 原图 | Top-k 双图 | Focus |
|---|---:|---:|---:|
| TextVQA soft accuracy，512 题 | 56.76 | **62.36** | 61.60 |
| GQA exact match，512 题 | 60.55 | 60.35 | **61.13** |
| POPE Accuracy，600 题 | 87.17 | **89.33** | 89.00 |

这些是固定开发子集结果，不是官方全量提交成绩。完整口径、置信区间和论文对照见
[`2026-08-09` 实验报告](experiments/2026-08-09/SLoFo_08_09_TextVQA_GQA_POPE_实验报告.md)。

### TextVQA 128 显存与因子消融（2026-08-10）

- selective hook 将 Scan 最大 PyTorch 分配峰值从 `24,665.4 MiB` 降至
  `19,086.9 MiB`，五路答案、候选框和排名与旧实现均为 `0/128` 差异；
- 论文公式直译 `1 Token + raw + 单框`：原图 `52.97`、双图 `57.50`、Focus `57.42`；
- raw 下 1/8 Token 的框、答案和分数完全一致，表明当前 raw 融合中结构分支淹没语义变化；
- min-max 会改变 `87/128` 个框并改善仅裁切分支，但当前 8 Token mean 对双图/Focus 有负趋势；
- Top-k 相对对应单框配置使双图和 Focus 各提高 `1.33` 个百分点（改善/退化 `3/0`），
  但平均耗时约增加 `67%`。N=128 的配对区间均跨 0，现阶段只作为工程趋势。

详见 [`2026-08-10` 实验目录](experiments/2026-08-10-textvqa128/README.md)。

## 仓库结构

```text
LLaVA_SLoFo_RE/
├── slofo/                   Scan-Locate 与 Focus 的张量核心
├── scripts/                 LLaVA 适配、批处理、空卡守卫、数据准备与分析
│   └── analysis/            早期 Focus/Top-k/图表分析脚本
├── tests/                   不依赖模型权重的 CPU 单元测试
├── config/                  最小推理依赖版本
├── images/test_08_06/       早期 10 图功能验证输入
├── experiments/
│   ├── 2026-08-06/          Scan-Locate 与坐标映射
│   ├── 2026-08-07/          Focus、随机对照与 Top-k 初测的精简产物
│   ├── 2026-08-09/          三个标准任务的清单、逐题答案、统计与图表
│   └── 2026-08-10-textvqa128/ 显存优化、论文基线与因子拆分
└── docs/SERVER_WORKFLOW.md  服务器与 VS Code 工作流
```

逐题大图、模型权重、数据集原图、缓存与服务器日志未提交。实验目录保留的是复核结论所需的
manifest、`benchmark_answers.jsonl`、`batch_summary.json`、CSV、统计 JSON 和小型图表。

## 固定实验版本

- LLaVA 源码：[`haotian-liu/LLaVA`](https://github.com/haotian-liu/LLaVA)，commit
  `c121f0432da27facab705978f83c4ada465e46fd`；
- checkpoint：`liuhaotian/llava-v1.5-7b`，revision
  `4481d270cc22fd5c4d1bb5df129622006ccd9234`；
- vision tower：`openai/clip-vit-large-patch14-336`，revision
  `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`；
- Python / PyTorch / Transformers：3.10.20 / 2.1.2+cu121 / 4.37.2；
- 模型精度：FP16；仅语义显著性乘法转 FP32；
- 语义层 / 结构层 / PCA：14 / 7 / 20；视觉 Token：`24×24=576`。

上游 LLaVA 源码、checkpoint 与数据集是外部依赖，不复制进本仓库。服务器路径和安装步骤见
[`docs/SERVER_WORKFLOW.md`](docs/SERVER_WORKFLOW.md)。

## 测试

张量核心测试不需要下载 LLaVA 权重：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

当前共 19 项 CPU 单元测试。服务器批处理的主要入口为：

```bash
source scripts/activate_project.sh
python scripts/run_slofo_vqa_benchmark_batch.py --help
```

`run_on_empty_gpu.sh` / `run_on_specific_empty_gpu.sh` 仅根据显存与利用率选择空闲卡，
不读取或操作其他用户的项目。

## 当前边界

已完成的是核心链路的功能性复现和初步标准数据集验证，尚不能称为论文完整复现。仍需完成：

- 真实 24 GB 显卡端到端验证；
- 高分辨率分块 Scan 与全局 SSIM 拼接；
- TextVQA、GQA、POPE 全量官方评测；
- 语义锚点、融合尺度与门控 Top-k 的进一步优化；
- Focus 的实际墙钟加速，而不只是序列和 token-layer 工作量下降。

## 参考

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [MLLMs Know Where to Look reference code](https://github.com/saccharomycetes/mllms_know)
