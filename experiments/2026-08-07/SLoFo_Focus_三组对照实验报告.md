# SLoFo Focus 三组对照实验报告

- 实验日期：2026-08-07（Asia/Shanghai）
- 报告整理时间：2026-08-07 19:10
- 服务器项目：`/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo`
- 测试集：`test_08_06_01.jpg` 至 `test_08_06_10.jpg`
- 结果目录：`08_07/Focus_三组对照实验结果/`
- 目的：解释“注意力 Focus 10/10 保持未剪枝双图回答”究竟来自有效 token 选择、裁切图主导，还是任务本身对任意剪枝都不敏感。

## 1. 实验设计

每张图先用同一套 Scan-Locate 配置得到唯一裁切框，然后在完全相同的问题、模型、FP16 精度、贪心解码和最大输出长度下运行：

| 路径 | 输入 | 原图 token 处理 | 重复次数 |
|---|---|---|---:|
| 未剪枝双图基线 | 原图 + 裁切图 | 保留 576 | 1 |
| 注意力 Focus | 原图 + 裁切图 | `576→288→144→72`，规划注意力排序 | 1 |
| 随机 Focus | 原图 + 裁切图 | 相同数量，随机排序 | seed 0/1/2 |
| Crop-only | 仅裁切图 | 完全移除原图 | 1 |

随机 Focus 与注意力 Focus 使用相同的层边界 7/15/23、相同裁切图 576 token、相同序列缩短和 KV cache 逻辑。唯一变化是原图 token 的保留依据。三个随机重复使用独立随机流：`seed × 1,000,003 + phase`。

所有运行均经过 `scripts/run_on_empty_gpu.sh`。绝大部分使用物理 GPU 1；守卫在一次运行中自动选择了当时空闲的物理 GPU 2。没有检查其他人的项目或进程命令行。

## 2. 记录指标

除最终字符串外，每个生成步还记录：

- 实际生成 token；
- top-1 与 top-2 token；
- top-1 logit margin；
- 生成 token 的 log probability；
- 未剪枝分布到对照分布的 KL；
- 对照分布到未剪枝分布的反向 KL；
- 两个方向平均的 symmetric KL；
- top-1 token 是否一致。

若两个路径中途生成了不同 token，后续 decode history 已不同，不能再做严格的同历史比较。因此汇总的“对齐 KL”只平均：

1. 生成历史仍相同的所有步骤；
2. 加上第一次产生不同 token 的决策步骤。

完整词表 logits 仅在内存中用于计算，JSON 保存逐步汇总，避免产物体积失控。

## 3. 总体结果

![三种对照相对未剪枝双图的分布偏移](Focus_三组对照实验结果/focus_controls_kl.png)

| 对照 | 与基线字符串一致 | 颜色一致 | 平均对齐 symmetric KL | 中位数 KL | 平均 top-1 一致率 |
|---|---:|---:|---:|---:|---:|
| 注意力 Focus | **10/10** | **10/10** | **0.002730** | **0.000868** | **100.0%** |
| 随机 Focus | 26/30 | 28/30 | 0.005737 | 0.001416 | 94.21% |
| Crop-only | 7/10 | 7/10 | 0.016685 | 0.008271 | 90.58% |

与随机 Focus 相比，注意力 Focus 的平均分布偏移低 **52.4%**；与 Crop-only 相比低 **83.6%**。随机 Focus 的平均 KL 是注意力 Focus 的 2.10 倍，Crop-only 是 6.11 倍。

## 4. 逐图 KL

| 序号 | Crop-only 与基线同文本 | 注意力 Focus KL | 随机 Focus 平均 KL | Crop-only KL | 随机同文本次数 |
|---|---:|---:|---:|---:|---:|
| 01 | 否 | 0.000980 | 0.002032 | 0.009665 | 3/3 |
| 02 | 是 | 0.000527 | 0.000955 | 0.002756 | 3/3 |
| 03 | 是 | 0.000272 | 0.000663 | 0.002449 | 3/3 |
| 04 | 否 | 0.001700 | 0.006091 | 0.025667 | 2/3 |
| 05 | 是 | 0.008923 | 0.004606 | 0.032864 | 3/3 |
| 06 | 是 | 0.000756 | 0.000679 | 0.006878 | 3/3 |
| 07 | 否 | 0.004341 | 0.035076 | 0.070517 | 0/3 |
| 08 | 是 | 0.000373 | 0.000935 | 0.003785 | 3/3 |
| 09 | 是 | 0.009226 | 0.005261 | 0.009789 | 3/3 |
| 10 | 是 | 0.000201 | 0.001072 | 0.002486 | 3/3 |

注意力 Focus 的 KL 在 7/10 张图上低于该图三个随机种子的平均值；在 30 个逐种子配对中低于随机 Focus 22/30 次。它并非每张都最小：05、06、09 的随机均值更低，但没有改变这些图的 top-1 输出。

## 5. 随机剪枝改变答案的案例

### 样例 04

未剪枝双图与注意力 Focus：

```text
The person is wearing black clothes.
```

随机 Focus seed 0：

```text
The person is wearing blue clothes.
```

另外两个随机种子仍为 black。说明相同剪枝数量下，随机保留集合可以跨过颜色决策边界，而注意力选择没有。

### 样例 07

未剪枝双图与注意力 Focus：

```text
The person standing on the sidewalk is wearing blue clothes.
```

随机 Focus：

```text
seed 0: The person standing on the sidewalk is wearing black clothes.
seed 1: The person is wearing blue clothes.
seed 2: The person is wearing blue clothes.
```

三个随机结果都没有逐字复现基线；seed 0 还把颜色从 blue 改成 black。这个样例同时是 Crop-only 失败样例，说明原图上下文对它较重要。

随机剪枝总计 30 次中，4 次改变完整文本，2 次改变颜色。注意力 Focus 10 次均未改变完整文本或颜色。

## 6. Crop-only 对照说明了什么

Crop-only 有 7/10 与双图基线完全一致，支持此前的主要推测：在多数衣服颜色问题中，未剪枝的 576 个裁切图 token 已提供主要局部证据。因此即使原图从 576 剪到 72，答案也常能保持。

但 01、04、07 三张的 Crop-only 与双图基线不同：

| 序号 | 双图基线 | Crop-only |
|---|---|---|
| 01 | blue | gray |
| 04 | black | blue |
| 07 | blue | black |

这证明模型并非完全忽略原图。特别是样例 07，随机剪枝与 Crop-only 都容易失去基线的 blue 判断，而注意力 Focus 保持 blue，符合“规划注意力保留了更有用的原图证据”的解释。

需要注意：这里的“保持基线”不等于“更接近人工答案”。例如样例 01 的人工参考是灰色，Crop-only 的 gray 反而比双图基线的 blue 更正确。对照实验衡量的是信息保真，不是准确率提升。

## 7. Focus 不是 no-op

注意力 Focus 的最终文本虽然 10/10 相同，但平均 symmetric KL 为 0.002730，并非 0。这说明剪枝确实改变了每步的输出概率分布，只是没有改变概率最高的 token。

另外：

- hidden states 的序列长度确实按阶段缩短；
- 每层 KV cache 长度随阶段变化；
- 峰值显存较未剪枝路径下降；
- 注意力 Focus 与随机 Focus 最终 72 个原图 token 的平均交集只有 8.3 个；
- 72 个 token 从 576 中随机抽取时，理论期望交集是 9 个。

因此注意力 Focus 选择的 patch 集合与随机集合基本处于随机重合水平，不是随机对照恰好选中了同一批 token。

## 8. 结论强度

当前证据支持以下结论：

1. **裁切图主导是 10/10 文本稳定的重要原因，但不是唯一原因。** Crop-only 可复现 7/10，另外 3/10 仍依赖原图。
2. **规划注意力选择比同数量随机剪枝更稳定。** 注意力 Focus 保持 10/10；随机剪枝只有 26/30 完整文本一致，且平均 KL 高 2.10 倍。
3. **Focus 改变了内部概率，但多数变化未跨过 top-1 决策边界。** 这解释了“KL 非零、文本仍完全一致”。
4. **不能据此宣称已经证明普适优势。** 只有 10 张同类型颜色问题；注意力 KL 仅在 7/10 个样例上低于随机均值，按图计算的样本量不足以给出强统计结论。

更严谨的下一步应扩大到至少 50～100 个带人工 bbox 和答案标注的样例，并包含计数、OCR、空间关系和多区域推理任务；随机剪枝至少保留 3～5 个种子。

## 9. 文件与复现

### 本地

- 汇总 JSON：`08_07/Focus_三组对照实验结果/focus_controls_summary.json`
- KL 图：`08_07/Focus_三组对照实验结果/focus_controls_kl.png`
- 10 组原始结果：`08_07/Focus_三组对照实验结果/experiments/slofo-08-07/focus-controls/`
- 运行日志：`08_07/Focus_三组对照实验结果/logs/slofo-08-07-focus-controls/`
- 分析脚本：`08_07/analyze_focus_controls.py`
- 批处理脚本：`research/remote_runtime/scripts/run_slofo_08_07_focus_controls.sh`

### 服务器

```bash
cd /data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo
source scripts/activate_project.sh
FORCE=1 scripts/run_slofo_08_07_focus_controls.sh
```

服务器结果位于 `experiments/slofo-08-07/focus-controls/`，日志位于 `logs/slofo-08-07-focus-controls/`。

## 10. 关键文件 SHA-256

```text
E11916D9DDF432DC74996292E3F6A258CF25A8A67A089F8E553F7C197318E765  focus.py
B73124275E378E075E03B3248CDD92AB6E9CBBD967C96AE44646AB4F3D03732A  slofo_focus_runtime.py
0C2B957E9333C60C53F5E3A6DF06CC9D58A98B40058733056F8F69ED9B6C426E  run_slofo_scan_locate.py
6DAF48A36344F4C8B9A45FF5BFF3FBE874E0D8CB0A55D582634ECC7072158BE8  run_slofo_08_07_focus_controls.sh
059FDB184FFE2EF45A6CF90FBA15FB88EEAA788FFFD935BF1AFA24793C1EAAF3  analyze_focus_controls.py
```
