# LLaVA SLoFo RE

这是一个面向代码学习与论文复现的非官方实验仓库，记录我们在
LLaVA-1.5-7B 上实现和验证 SLoFo **Scan-Locate** 核心流程的代码与结果。

> 当前仓库不是论文作者发布的官方实现。SLoFo 后续 Focus 四阶段 token
> 剪枝尚未实现；当前完成的是扫描、定位、裁切和原图+裁切图联合推理。

## 已完成

- 从 LLaVA 指定层提取 576 个视觉 token 的注意力与梯度；
- 计算梯度加权语义重要性图；
- 从结构层隐状态计算 PCA 重构误差；
- 融合语义/结构分支并进行多尺度滑窗定位；
- 比较 raw/min-max 融合以及 original/padded 坐标映射；
- 在同一个 prompt 中真正输入原图和裁切图两个图像张量；
- 修复 FP16 下 `attention * gradient` 的显著性下溢；
- 在 10 张测试图上完成可复现实验并保存报告。

## 关于 FP32 和 576 个非零 token

LLaVA 模型仍以 FP16 运行。这里只把语义显著性计算中的 attention 和
gradient 临时转为 FP32 后再相乘，避免约 `1e-8` 的小正数在 FP16 中被舍入
为零。

`18/576 -> 576/576` 表示显著性图中 576 个位置都保留了数值信息，**不表示
最终保留了 576 个 token，也不表示没有裁切**。Scan-Locate 需要先观察全部
视觉 token 才能选择裁切框；当前裁切结果记录在每个实验案例的
`selected_bbox.png` 和 `crop.png` 中。真正减少后续语言模型视觉 token 的
Focus 阶段仍未实现。

## 仓库结构

```text
LLaVA_SLoFo_RE/
|-- slofo/                 独立 Scan-Locate 张量实现
|-- scripts/               LLaVA 适配、双图推理、空卡守卫与批量实验
|-- tests/                 单元测试
|-- config/                最小推理依赖版本记录
|-- images/test_08_06/     10 张测试输入与问题清单
|-- experiments/2026-08-06/
|   |-- SLoFo_08_06_实验报告.md
|   `-- SLoFo_08_06_实验结果/  精简且可核验的结果
`-- docs/SERVER_WORKFLOW.md  服务器复现工作流
```

## 固定实验版本

- LLaVA 源码：[`haotian-liu/LLaVA`](https://github.com/haotian-liu/LLaVA)，
  commit `c121f0432da27facab705978f83c4ada465e46fd`
- LLaVA checkpoint：`liuhaotian/llava-v1.5-7b`，revision
  `4481d270cc22fd5c4d1bb5df129622006ccd9234`
- Vision tower：`openai/clip-vit-large-patch14-336`，revision
  `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`
- 模型精度：FP16；语义显著性乘法使用 FP32
- 语义层 / 结构层：14 / 7
- PCA 维度：20
- 视觉 token：24 × 24 = 576
- 实验 GPU：空闲的 NVIDIA GeForce RTX 4090

上游 LLaVA 源码和模型权重没有复制进本仓库。服务器运行前需要按
[`docs/SERVER_WORKFLOW.md`](docs/SERVER_WORKFLOW.md) 准备它们，并根据自己的
目录修改 `scripts/activate_project.sh` 与批量脚本中的项目路径。

## 测试

张量核心只需要可用的 PyTorch，不需要下载 LLaVA 权重：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

当前服务器结果：11/11 项通过。

完整 LLaVA 批量实验入口：

```bash
source scripts/activate_project.sh
scripts/run_slofo_08_06_batch.sh
```

脚本会通过 `run_on_empty_gpu.sh` 读取聚合显存/利用率，只选择空闲卡，不读取
其他用户的项目或进程命令行。

## 实验结果

结论、逐图 bbox、回答和失败原因见
[`SLoFo_08_06_实验报告.md`](experiments/2026-08-06/SLoFo_08_06_实验报告.md)。

当前 10 图结果：

- `minmax + original` 人工定位约 8/10；
- 严格颜色回答为 6 个正确、2 个部分/近似正确、2 个错误；
- 03、05 的失败来自扫描阶段语义峰值选错，而不是统一的坐标偏移。

## 参考

- LLaVA: <https://github.com/haotian-liu/LLaVA>
- MLLMs Know Where to Look reference code:
  <https://github.com/saccharomycetes/mllms_know>
