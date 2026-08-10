# SLoFo Focus 四阶段剪枝与多 token 语义锚点实验报告

> 补充验证：注意力剪枝、三种随机剪枝、Crop-only 与逐 token KL 的配对实验见 [SLoFo_Focus_三组对照实验报告.md](SLoFo_Focus_三组对照实验报告.md)。

- 实验日期：2026-08-07（Asia/Shanghai）
- 报告整理时间：2026-08-07 18:36
- 服务器项目：`/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo`
- 本地结果：`08_07/SLoFo_08_07_实验结果/`
- 论文依据：[SLoFo.pdf](../Paper/SLoFo.pdf) 第 2.3 节

## 1. 本轮结论

1. **Focus 四阶段剪枝已真实接入 LLaVA-1.5-7B。** 32 个语言层按 4 个等长阶段划分，在第 7、15、23 层后执行三次剪枝；原图视觉 token 严格按 `576 → 288 → 144 → 72` 变化，裁切图的 576 个 token 和文本 token 全部保留。
2. **剪枝改变了后续层的真实输入长度。** 不是只生成一个 mask 或事后统计：hidden states 会缩短，KV cache 也按各层实际长度维护，生成阶段的 position id 按每层 cache 长度继续。
3. **10 张图的 Focus 保真测试通过。** Focus 双图回答与未剪枝双图回答逐字一致 **10/10**。这证明当前剪枝没有进一步改变这组答案，但不代表上游裁切和答案本身 10/10 正确。
4. **估算 prefill token-layer 工作量平均减少 25.15%。** 范围为 25.08%～25.23%；平均显存峰值从 18629.7 MiB 降至 18258.2 MiB，减少约 371.5 MiB。
5. **当前实现尚未获得墙钟加速。** 未剪枝双图平均生成 0.5667 s，Focus 平均 0.6641 s，约慢 16.5%。原因是当前验证版需要 eager attention、逐阶段取注意力和 Python 运行时调度；因此现阶段能证明“剪枝逻辑完整、计算 token 数减少”，不能宣称已经实现实际推理加速。
6. **多 token 语义锚点已实现，但效果有正有负。** 8-token rollout 修复了昨天第 3 张图的错人问题，使双图回答从黑色变为正确的白色；但第 5 张仍失败，并且部分原本较好的样例出现裁切漂移。它是一个可控的实验能力，还不是稳定优于单锚点的最终方案。

## 2. 环境与固定模型版本

| 项目 | 配置 |
|---|---|
| 基础模型 | LLaVA-1.5-7B |
| 本地 checkpoint | `models/llava-v1.5-7b` |
| LLaVA revision | `4481d270cc22fd5c4d1bb5df129622006ccd9234` |
| Vision Tower | CLIP ViT-L/14-336 |
| Vision Tower revision | `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| 模型精度 | FP16；语义显著性乘法使用 FP32 |
| 语言层数 | 32 |
| 原图 / 裁切图 token | 576 / 576 |
| Scan-Locate | 语义层 14、结构层 7、PCA 20、beta=0.7 |
| 工作配置 | 两分支 min-max、原图坐标、8-token rollout |
| GPU | 空闲的物理 GPU 1，NVIDIA GeForce RTX 4090 |
| 选卡方式 | `scripts/run_on_empty_gpu.sh`；未查看他人项目或进程命令行 |
| 单元测试 | 16/16 通过 |

本轮使用的是论文兼容的 FP16 LLaVA-1.5-7B，不是 4-bit 模型。所有模型文件均从项目自己的本地目录离线加载。

## 3. Focus 的实现

### 3.1 四阶段结构

按照论文第 2.3 节，语言模型被划分为 `K=4` 个阶段。对于 32 层 LLaVA，阶段为：

```text
阶段 1：layer 0–7   -> 剪枝 1：576 -> 288
阶段 2：layer 8–15  -> 剪枝 2：288 -> 144
阶段 3：layer 16–23 -> 剪枝 3：144 -> 72
阶段 4：layer 24–31 -> 保留最终 72 个原图 token
```

每个剪枝边界使用该层最后一个规划锚点对“当前剩余原图 token”的注意力，删除注意力最低的 50%。token 类型在多模态展开时被明确记录为：文本、原图视觉 token、裁切图视觉 token。剪枝函数只允许删除原图 token。

### 3.2 运行时完整性

- 每次剪枝后，hidden states、position ids、token 类型和稳定的原图 patch ID 同步缩短；
- position ids 在剪枝后重新压紧，避免稀疏旧位置超出 RoPE cache；
- 每层 KV cache 长度可能不同，decode 时按该层真实 cache 长度生成 position id；
- 裁切图 token 始终为 576；
- 每阶段保存保留/删除的原图 patch ID、序列长度、注意力统计和可视化文件。

样例 01 的序列长度变化是：

```text
1216 -> 928 -> 784 -> 712
```

其中最终 712 包含 72 个原图 token、576 个裁切图 token和文本 token。

## 4. Focus 的 10 图验证

![10 图裁切框](SLoFo_08_07_实验结果/experiments/slofo-08-07/batch-rollout8-focus4/selected_bboxes_contact.png)

![Focus 第三次剪枝后保留的原图 token](SLoFo_08_07_实验结果/experiments/slofo-08-07/batch-rollout8-focus4/focus_phase_3_contact.png)

| 序号 | bbox | 未剪枝双图回答 | Focus 回答 | 人工核对 |
|---|---|---|---|---|
| 01 | `[792,432,1128,768]` | blue | blue | 框到目标；颜色错误，应为灰色 |
| 02 | `[318,545,721,948]` | black | black | 框含雪人和目标一部分；答案正确 |
| 03 | `[525,1255,861,1591]` | white | white | 定位和答案均修正为正确 |
| 04 | `[272,739,608,1075]` | black | black | 框偏向同一人物下半身；上衣应为深灰，近似 |
| 05 | `[472,1325,808,1661]` | black | black | 框落在台球区域；错误，应为浅灰蓝 |
| 06 | `[1031,465,1434,868]` | white | white | 正确 |
| 07 | `[472,1183,808,1519]` | blue | blue | 框偏脚部，但原图分支保住正确答案 |
| 08 | `[472,756,808,1092]` | green | green | 正确 |
| 09 | `[419,232,755,568]` | white | white | 框到白衣人物；错误，目标是棕绿色格纹人物 |
| 10 | `[352,312,688,648]` | green | green | 正确 |

Focus 与未剪枝回答完全一致 10/10。按颜色严格核对，本轮为 **6 正确、1 近似、3 错误**。错误来自 Focus 之前的 Scan-Locate 或基础模型判断；Focus 没有修复它们，也没有额外造成答案变化。

## 5. 计算量、显存与时间

| 指标 | 未剪枝双图 | Focus | 变化 |
|---|---:|---:|---:|
| 原图视觉 token（最终阶段） | 576 | 72 | -87.5% |
| 裁切图视觉 token | 576 | 576 | 不变 |
| 平均估算 prefill token-layer 工作量 | 100% | 74.85% | -25.15% |
| 平均峰值显存 | 18629.7 MiB | 18258.2 MiB | -371.5 MiB |
| 平均生成时间 | 0.5667 s | 0.6641 s | +0.0974 s |

原图最终只剩 12.5%，但总工作量不会减少 87.5%：前三个阶段仍要处理更多原图 token，且裁切图 576 token 与文本 token 从不剪枝。因此本组约 25% 的理论工作量减少是合理的。

当前 wall-clock 变慢不等同于剪枝无效。验证版为了获得每个边界的 attention，强制 eager attention 并在 Python 中动态改序列和 cache；这些额外开销盖过了本组短回答中的矩阵计算节省。后续若追求实际加速，需要把剪枝路径并入模型 forward、减少 Python 分支、使用更高效的 attention kernel，并在更长输出上重新 benchmark。

## 6. 多 token 语义锚点优化与消融

### 6.1 新增能力

原来的语义分支只使用首个预测 token（通常是高置信度的 `The`）。现在可以：

1. 先贪心生成 N 个响应 token；
2. teacher-force 前 N-1 个 token；
3. 为每个目标 token 建立对应的规划锚点；
4. 对这些 token 的平均 log probability 求 attention 梯度；
5. 选择跨 token `mean` 或 `max` 聚合；
6. 用 `semantic_anchor_start_index` 跳过 `The person`，单独研究动作词锚点。

语义显著性仍在 FP32 中计算，避免 `attention × gradient` 的 FP16 下溢。

### 6.2 两个昨日失败样例

| 样例 | 配置 | rollout | bbox | 双图回答 | 结论 |
|---|---|---|---|---|---|
| 03 | 昨日单锚点 | `The` | `[365,899,701,1235]` | black | 错人 |
| 03 | 8-token mean | `The person standing in front and holding a` | `[525,1255,861,1591]` | white | 修复成功 |
| 05 | 昨日单锚点 | `The` | `[152,258,488,594]` | black | 错误 |
| 05 | 8-token mean | `The person leaning forward against the pool` | `[472,1325,808,1661]` | black | 仍错误 |
| 05 | 12-token mean | `The person leaning forward against the pool table is wearing` | `[472,1325,808,1661]` | black | 无进一步变化 |
| 05 | 12-token max | 同上 | `[472,1325,808,1661]` | black | 通用锚点仍占优 |
| 05 | 跳过前 2 token + max | `leaning ... wearing` | `[579,1325,915,1661]` | black | 横向靠近目标，纵向仍被台球吸引 |

第 5 张的失败证明“延长 rollout”本身不足以保证定位。动作锚点已经改变了语义选择，但融合后的最高滑窗仍被台球和前景结构吸引。下一步更合理的是 top-k 候选重排或显式目标短语—视觉区域对齐，而不是继续增加 token 数或手调固定坐标。

### 6.3 全集上的权衡

8-token 方案成功修复样例 03，但样例 01、04、07、09 的裁切或回答出现不同程度漂移。因此本次结果不支持“8-token 全局优于单锚点”的结论。代码保留单锚点、mean、max 和动作锚点筛选，后续可在更大标注集上选择，而不是把当前 10 图的偶然结果写死为默认真理。

## 7. 文件与复现入口

### 本地代码

- Focus 张量核心：`research/slofo/focus.py`
- Scan-Locate 与多 token 聚合：`research/slofo/scan_locate.py`
- Focus LLaVA 运行时：`research/remote_runtime/scripts/slofo_focus_runtime.py`
- 完整实验入口：`research/remote_runtime/scripts/run_slofo_scan_locate.py`
- 08-07 批处理：`research/remote_runtime/scripts/run_slofo_08_07_batch.sh`
- 回归测试：`research/tests/test_slofo_scan_locate.py`

### 本地结果

- 汇总 JSON：`08_07/SLoFo_08_07_实验结果/experiments/slofo-08-07/batch-rollout8-focus4/summary.json`
- 10 组 result、map、重建图：`08_07/SLoFo_08_07_实验结果/experiments/slofo-08-07/batch-rollout8-focus4/`
- 服务器运行日志：`08_07/SLoFo_08_07_实验结果/logs/slofo-08-07/`
- 轻量原始包：`08_07/SLoFo_08_07_实验结果/slofo-08-07-metadata.tar.gz`
- 可视化重建脚本：`08_07/build_report_artifacts.py`

服务器保留了包含全部原始 PNG 的完整压缩包：`tmp/slofo-08-07-results.tar.gz`。本地版本下载 JSON、NPY 和日志后，从原始图片无损重建关键 PNG，避免重复传输 194 MiB。

### 服务器复现

```bash
cd /data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo
source scripts/activate_project.sh
scripts/run_slofo_08_07_batch.sh
```

脚本逐张调用空卡守卫；已有 `result.json` 时默认跳过，可设置 `FORCE=1` 重跑。

## 8. 关键文件 SHA-256

```text
5C6905C21BD9761FB9620B4436B7897596D03F9FDF308E6D3D93859F422AE399  focus.py
708097A634705EB0922C16F1C715E87F3CEF3623CBB48F5ED4BF8D1B9B480ECB  scan_locate.py
BFCB082C3DDD42B694442944DBBFD564F24FF1FE1B29A200B9C5EFA018405205  slofo_focus_runtime.py
CBFFC3D5508798B42DC0B276B89AF681EDAC1189094CAF0AF8F6F9744C283269  run_slofo_scan_locate.py
DB0DE7814656C94A84907BA58767752A18482332F4F648EA40ECEDAABA3C9F8D  run_slofo_08_07_batch.sh
54A5FE4E93333BA7785536097B9A5EB399AE0A774E36F09AEA1C0A913DECAFEE  test_slofo_scan_locate.py
```

## 9. 当前完成度与后续问题

今天要求的两项均已完成：Focus 四阶段剪枝已实现并通过 10 图完整链路验证；动作短语/多 token 语义评分已实现、做了消融并明确了有效与失败边界。

下一阶段最值得处理的是：

1. 对 Scan-Locate 的 top-k 窗口做目标短语一致性重排，优先解决第 5、9 张；
2. 将 Focus 从 Python monkey-patch 整合进正式 model forward，验证是否能获得真实 wall-clock 加速；
3. 扩大带人工 bbox/答案标注的测试集，客观选择单锚点、mean、max 或动作锚点策略。
