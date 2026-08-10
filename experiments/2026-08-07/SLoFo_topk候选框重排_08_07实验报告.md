# SLoFo top-k 候选框重排实验报告（08_07）

- 报告时间：2026-08-07 21:10（Asia/Shanghai）
- 实验项目：`HYG_LLaVA_SLoFo`
- 模型：LLaVA-v1.5-7B，FP16
- 服务器：校内共享 GPU 服务器（账号与地址不写入公开仓库）
- GPU：由空卡守卫选择物理 GPU 1；实验开始时显存占用 15 MiB、利用率 0%
- 本轮验证次数：top-k 原始冒烟 1 次、修正后冒烟 1 次、20 图批量验证 1 轮

## 1. 本轮完成内容

1. 从 Ultralytics 官方 Release 下载并保留完整 COCO128 数据包，没有筛掉任何图片。
2. 将 128 张图片和标签统一命名为 `coco128_08_07_001` 至 `coco128_08_07_128`，并保存源 COCO ID、尺寸、类别和校验信息。
3. 从其中 61 张带人物标签的图片中人工核验 20 张，建立“人物姿势 + 衣服颜色”问答和人物 bbox。其余 108 张仍完整保留，供后续实验使用。
4. 实现多尺度候选框 top-k、IoU-NMS、LLaVA 双图目标存在性验证、裁切答案一致性评分和保守回退。
5. 验证 top-k 与四阶段 Focus 可以同时运行；Focus 的原图 token 调度仍为 `576 → 288 → 144 → 72`。
6. 在空闲 GPU 上完成 20 张图片的 top-k 定位对照，并将完整 JSON、日志、候选裁切和可视化下载到本地。

数据来源：

- [Ultralytics COCO128 官方文档](https://docs.ultralytics.com/datasets/detect/coco128/)：COCO128 是 COCO train2017 的前 128 张图。
- [Ultralytics 官方 COCO128 压缩包](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip)
- [COCO 官方网站](https://cocodataset.org/)

下载压缩包 SHA256：

```text
61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e
```

## 2. 全量数据校验

| 项目 | 数量/结果 |
|---|---:|
| 下载图片 | 128 |
| 统一命名后的图片 | 128 |
| 统一命名后的标签 | 128 |
| 带 `person` 标签的图片 | 61 |
| 本轮人工核验 QA+bbox | 20 |

官方压缩包中存在源 ID 对应不一致：`000000000250.jpg`、`000000000508.jpg` 没有同名标签；标签目录则多出 `000000000656.txt`、`000000000659.txt`，但没有同名图片。为满足“128 张全量保留”，前两张图片生成了标准 YOLO 空标签；不把后两个孤立标签错误配给其他图片。该情况已写入 `source_manifest.json`。

中文问题遵循用户要求的格式，例如：

```text
图中正在篮筐附近投篮的人物的衣服颜色是？
```

由于 LLaVA-v1.5 的英文问答更稳定，清单同时保存等价英文 `model_query`，服务器实际推理使用英文；中文问题、英文问题、人工颜色答案和人物 bbox 均保留在同一条记录中。人工 bbox 和颜色答案仅用于推理结束后的评估，没有提供给候选生成或重排模块。

## 3. top-k 重排实现

### 3.1 候选生成

- 候选 1 固定为旧版 Scan-Locate 的单框结果，保证可以做严格基线对照。
- 每个滑窗尺度从 SSIM 图中取若干高证据位置，不再只保留每个尺度的一个位置。
- 按局部对比度排序，并用原图坐标下的 IoU-NMS 去重。
- 配置：`top_k=5`、`pre_nms_per_scale=12`、`nms_iou=0.55`。
- 因为 NMS 会合并高度重叠的框，实际平均候选数为 2.35；`top_k=5` 是上限，不保证每张图一定得到 5 个有效框。

### 3.2 无标注泄漏的重排

每个候选使用两类模型信号：

1. 原图 + 候选裁切：询问候选中是否包含问题指向的特定人物，计算 `Yes` 对 `No` 的首 token 对数几率。
2. 候选裁切单图：回答原衣服颜色问题，并与原图单图产生的“伪答案”比较颜色词一致性。

最终分数：

```text
score = Yes/No log-odds
      + 0.15 × normalized scan contrast
      + 1.00 × answer color consistency
```

若最优候选相对旧框的优势小于 0.2，则保留旧框。这一保守回退防止 Yes/No 验证器在多人画面中因微小分差错误换框。

### 3.3 首次失败与修正

第 13 张是小目标案例：远处红衣人物在篮筐旁投篮，前景还有一名黑衣人物。

- 原始 Yes/No 重排把候选 2 排第一，裁切和未剪枝双图均错误回答“黑色”。
- 四阶段 Focus 仍回答“红色”，证明 Focus 与 top-k 链路兼容，但不能掩盖上游选框错误。
- 加入裁切答案与原图伪答案的颜色一致性后，候选 3 被选中，覆盖红衣小目标；裁切、未剪枝双图和 Focus 均回答“红色”。

候选框总览：[`topk_candidates_overview.png`](./topk_rerank_experiment/experiments/slofo-08-07/topk-rerank20-v2/topk_08_07_13/topk_candidates_overview.png)

修正后的框：[`selected_bbox.png`](./topk_rerank_experiment/experiments/slofo-08-07/topk-rerank20-v2/topk_08_07_13/selected_bbox.png)

Focus 第三次裁剪后的 token：[`focus_phase_3_kept_tokens.png`](./topk_rerank_experiment/experiments/slofo-08-07/topk-smoke-focus-v2/topk_08_07_13/focus_phase_3_kept_tokens.png)

## 4. 20 图实验结果

| 指标 | 旧版单框 | 保守 top-k 重排 | top-k 事后最优上限 |
|---|---:|---:|---:|
| 目标中心命中 | 19/20（95%） | 20/20（100%） | 20/20（100%） |
| 平均目标覆盖率 | 0.776 | 0.826 | 0.955 |
| 平均 IoU | 0.448 | 0.450 | 0.498 |

补充统计：

- 实际换框：1/20。
- 不加安全阈值时，非旧框的最高分候选：4/20。
- 被 0.2 阈值回退为旧框：3/20。
- 旧框双图回答与重排双图回答发生变化：0/20。
- 人工严格判定：17 正确、2 部分正确、1 错误；严格准确率 85%，半分制 90%，宽松准确率 95%。
- 最高 Scan 显存：21,379.6 MiB。
- 最高生成阶段显存：18,639.5 MiB。

指标图：[`topk_localization_comparison.png`](./topk_rerank_experiment/topk_localization_comparison.png)

逐样本表：[`topk_rerank_cases.csv`](./topk_rerank_experiment/topk_rerank_cases.csv)

机器可读汇总：[`topk_rerank_summary.json`](./topk_rerank_experiment/topk_rerank_summary.json)

## 5. 如何解释“回答 0/20 变化”

这不代表 top-k 没有执行。每张图都实际生成了候选、目标存在性分数和裁切单图答案；只是 19 张图保守地保留旧框，唯一换框的第 13 张在换框前后双图答案本来都已经是“红色”。

本轮直接收益主要在定位：第 13 张由“不包含目标中心、目标覆盖率 0”改善为“包含目标中心、目标覆盖率 1”。但平均 IoU 只提高 0.002，说明新框虽然覆盖目标，却仍偏大、包含前景干扰人物；它不是一个紧致人物检测框。

top-k 事后最优的平均覆盖率 0.955，明显高于当前重排的 0.826。这说明候选生成已经提供更好的框，下一步瓶颈主要在候选评分，而不是候选数量。

## 6. 当前局限与后续任务

1. 原图伪答案一致性可避免已知退化，但会继承原图回答的错误，属于“安全增强”而非独立定位证据。
2. Yes/No 分数在多人画面上区分度不足；首次第 13 张的三个候选 `Yes` 概率仅约为 0.669、0.696、0.672。
3. 当前候选框是固定方形滑窗，无法像目标检测框一样紧贴人物，因此目标覆盖率提升不一定带来明显 IoU 提升。
4. 20 条衣服颜色问答中有多色穿搭，报告同时给出严格、半分和宽松评分，避免把合理但不完整的颜色回答简单记为全错。
5. 批处理脚本当前每张图重新加载模型，结果正确但效率不高；可改为模型只加载一次的批量运行入口。

建议下一步按优先级完成：

1. 实现候选 A/B 成对比较，让模型直接判断哪个裁切更符合动作短语，而不是分别回答 Yes/No。
2. 将动作短语对应的 planning anchor 得分直接加入候选分数，降低原图颜色伪答案的循环依赖。
3. 在同一 20 图集合上重新比较安全重排与 A/B 重排，目标是接近 top-k oracle 的 0.955 覆盖率，同时不降低回答准确率。
4. 候选评分稳定后，再对 20 张图全量启用四阶段 Focus；当前只用第 13 张验证了 top-k 与 Focus 的完整兼容性。

## 7. 主要文件

- 完整 128 图数据：[`image/topk_08_07_coco128`](../image/topk_08_07_coco128)
- 20 条 QA+bbox：[`qa_bbox_20.json`](./topk_qa_dataset/qa_bbox_20.json)
- 数据来源清单：[`source_manifest.json`](../image/topk_08_07_coco128/source_manifest.json)
- top-k 核心：[`scan_locate.py`](../research/slofo/scan_locate.py)
- LLaVA 推理与重排：[`run_slofo_scan_locate.py`](../research/remote_runtime/scripts/run_slofo_scan_locate.py)
- 20 图批处理：[`run_slofo_topk_08_07_batch.py`](../research/remote_runtime/scripts/run_slofo_topk_08_07_batch.py)
- CPU 测试：[`test_slofo_scan_locate.py`](../research/tests/test_slofo_scan_locate.py)
- 完整结果目录：[`topk_rerank_experiment`](./topk_rerank_experiment)
- 原始服务器结果包：[`topk-rerank20-v2-full.tar.gz`](./topk-rerank20-v2-full.tar.gz)

## 8. 可复现实验命令

服务器项目目录：

```bash
cd /data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo
source scripts/activate_project.sh
```

CPU 单元测试：

```bash
python -m unittest tests.test_slofo_scan_locate
```

在空卡上运行 20 图 top-k 定位对照：

```bash
scripts/run_on_empty_gpu.sh \
  python scripts/run_slofo_topk_08_07_batch.py \
  --manifest manifests/08_07/qa_bbox_20.json \
  --image-root images/topk_08_07_coco128/images \
  --output-root experiments/slofo-08-07/topk-rerank20-v2 \
  --log-root logs/slofo-08-07/topk-rerank20-v2 \
  --top-k 5
```

在空卡上复现第 13 张的 top-k + Focus 完整链路：

```bash
scripts/run_on_empty_gpu.sh \
  python scripts/run_slofo_topk_08_07_batch.py \
  --manifest manifests/08_07/qa_bbox_20.json \
  --image-root images/topk_08_07_coco128/images \
  --output-root experiments/slofo-08-07/topk-smoke-focus-v2 \
  --log-root logs/slofo-08-07/topk-smoke-focus-v2 \
  --top-k 5 --enable-focus --case-ids topk_08_07_13
```
