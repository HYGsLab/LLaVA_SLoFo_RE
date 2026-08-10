# SLoFo 在 TextVQA、GQA、POPE 上的阶段性评测报告

> 实验日期：2026-08-09（Asia/Shanghai）  
> 模型：LLaVA-v1.5-7B，FP16，CLIP ViT-L/14-336  
> 结论性质：固定、可复现的开发子集配对实验；不是三个 benchmark 的全量官方提交分数。

## 1. 结论摘要

当前模型已经可以完整执行 TextVQA、GQA、POPE 三种任务格式，并对每题稳定产出以下五条推理分支：原图、仅裁切、旧单框双图、top-k 重排双图、四阶段 Focus。共完成 1,624 题，未出现缺图、模型崩溃或中途 OOM。

最主要结果如下：

- TextVQA：原图 56.76%，top-k 双图 62.36%，Focus 61.60%。top-k 相对原图提高 5.61 个百分点，Focus 提高 4.84 个百分点。
- GQA：原图 60.55%，top-k 双图 60.35%，Focus 61.13%。Focus 是当前最优分支，但只提高 0.59 个百分点，配对检验不显著。
- POPE：原图 Accuracy/F1 为 87.17%/86.13%，top-k 双图为 89.33%/88.81%，Focus 为 89.00%/88.42%。top-k 的 +2.17 个百分点具有显著的配对证据（McNemar p≈0.019）。
- top-k 重排在 TextVQA 和 POPE 上有效，在 GQA 上略有负收益；Focus 在 GQA 上进一步改善，但在 TextVQA、POPE 上相对 top-k 略有损失。
- 当前 TextVQA Scan 峰值达到 25,938 MiB。服务器实际报告的是约 48 GB 显存的 RTX 4090；该峰值不能保证在标准 24 GB 4090 上直接运行，是下一阶段必须解决的问题。

![[analysis/figures/01_branch_scores.svg]]

![[analysis/figures/02_delta_from_original.svg]]

## 2. 评测范围与数据口径

本轮目标是确认当前 SLoFo 实现能否接入标准 VQA benchmark，并在可控成本下比较所有已实现分支。完整数据规模为 TextVQA 5,000 题、GQA testdev-balanced 12,578 题、LLaVA 格式 POPE 8,910 题；以当前完整 pipeline 的速度，在单卡全量运行需要较长时间。因此先采用固定子集：

| Benchmark | 本轮题数 | 来源总题数 | 抽样方式 |
|---|---:|---:|---|
| TextVQA | 512 | 5,000 | seed=20260809 的确定性均匀抽样；保留官方 `Reference OCR token` 提示 |
| GQA | 512 | 12,578 | seed=20260809 的 testdev-balanced 确定性均匀抽样 |
| POPE | 600 | 8,910 | adversarial/popular/random × yes/no 六个格子各 100 题 |

所有清单都逐条校验了图片路径，缺失数均为 0。完整问题、答案和抽样顺序见：

- [[manifests/textvqa_subset_512.json]]
- [[manifests/gqa_subset_512.json]]
- [[manifests/pope_stratified_600.json]]
- [[manifests/manifest_summary.json]]

TextVQA 使用 LLaVA 官方格式的 OCR 辅助提示，因此测量的是论文采用的 TextVQA/LLaVA 口径，不是“完全不提供 OCR token 的纯视觉 OCR”。

## 3. 模型与 SLoFo 配置

| 项目 | 配置 |
|---|---|
| 基础模型 | LLaVA-v1.5-7B |
| 精度 | FP16，未使用 4-bit/8-bit |
| 视觉塔 | CLIP ViT-L/14-336 |
| Semantic 层 | 14 |
| Structure 层 | 7 |
| PCA 维度 | 20 |
| 融合 | min-max，语义权重 0.7 |
| 语义规划 | 最多 8 个 rollout token，多 token mean 聚合 |
| 定位坐标 | original image space |
| top-k | k=5；generic-evidence verifier；保守回退阈值 0.2 |
| Focus | 4 阶段；每次保留原图剩余 token 的 50%；计划为 576→288→144→72 |
| 生成 | greedy；TextVQA/GQA 最多 16 token，POPE 最多 8 token |

大规模模式只为每题保留 `result.json`；每个 benchmark 的前 3 题额外保留热力图、SSIM、top-k 候选框、裁切图和三阶段 token 可视化。该模式已经通过单题冒烟测试，数值推理路径与完整可视化模式一致。

## 4. 主要得分

TextVQA 使用官方 VQA soft accuracy；GQA 使用与 LLaVA 输出转换一致的短答案 exact match；POPE 使用官方 yes/no 归一化，并报告 Accuracy/F1。

| Benchmark | 原图 | 仅裁切 | 旧双图 | top-k 双图 | Focus |
|---|---:|---:|---:|---:|---:|
| TextVQA soft accuracy | 56.76 | 59.34 | 61.05 | **62.36** | 61.60 |
| GQA exact match | 60.55 | 58.79 | 60.74 | 60.35 | **61.13** |
| POPE accuracy | 87.17 | 88.17 | 88.33 | **89.33** | 89.00 |
| POPE F1 | 86.13 | 87.57 | 87.63 | **88.81** | 88.42 |

相对原图的配对变化：

| Benchmark / 分支 | 变化（百分点） | 95% 配对区间 | 纠正 / 退化 | 说明 |
|---|---:|---:|---:|---|
| TextVQA top-k | **+5.61** | [+2.52, +8.69] | 66 / 29 | 明确正收益 |
| TextVQA Focus | **+4.84** | [+1.72, +7.97] | 64 / 32 | 明确正收益 |
| GQA top-k | -0.20 | [-2.40, +2.01] | 16 / 17 | 无正收益 |
| GQA Focus | +0.59 | [-1.55, +2.72] | 17 / 14 | p≈0.720，不显著 |
| POPE top-k | **+2.17** | [+0.48, +3.86] | 20 / 7 | McNemar p≈0.019 |
| POPE Focus | **+1.83** | [+0.21, +3.46] | 18 / 7 | McNemar p≈0.043 |

这里的 95% 区间基于逐题得分差的正态近似。TextVQA 得分可以是分数值，因此不使用二元 McNemar 检验。

## 5. 与 SLoFo 原论文的对照

原论文在 LLaVA-v1.5-7B、完整 benchmark 上报告：

| Benchmark | 论文 Baseline | 论文 SLoFo | 论文增益 | 我们原图子集 | 我们 Focus 子集 | 我们增益 |
|---|---:|---:|---:|---:|---:|---:|
| TextVQA | 58.23 | 63.02 | +4.79 | 56.76 | 61.60 | **+4.84** |
| GQA | 60.32 | 61.78 | +1.46 | 60.55 | 61.13 | +0.59 |
| POPE MSCOCO 平均 | 85.31 | 88.19 | +2.88 | 87.17 | 89.00 | +1.83 |

注意：左侧是论文全量结果，右侧是本轮固定子集，不能把绝对分数直接当作复现误差。可比较的更合理对象是“同一批题上 SLoFo 相对原图的配对变化”。

值得特别澄清：论文摘要中的 GQA `+2.58` 指 high-res 的 62.90 相对 baseline 60.32；默认分辨率 SLoFo 是 61.78，即 +1.46。本轮实现是默认 336 输入，不应以 +2.58 作为目标。

本轮 TextVQA 的 +4.84 与论文 +4.79 非常接近；GQA 增益偏小；POPE 有明确正收益，但主要来自 popular/random，adversarial 改善不足。

## 6. top-k 候选框重排分析

| Benchmark | 改变旧框 | 选择 rank 分布 | 相对旧双图的答案变化 | 得分改善 / 退化 |
|---|---:|---|---:|---:|
| TextVQA | 53/512（10.35%） | r1=459, r2=28, r3=16, r4=7, r5=2 | 17 | 12 / 2 |
| GQA | 32/512（6.25%） | r1=480, r2=30, r3=2 | 5 | 1 / 3 |
| POPE | 61/600（10.17%） | r1=539, r2=43, r3=11, r4=7 | 6 | 6 / 0 |

结论：

1. 保守阈值确实避免了频繁改框，约 90% 题目维持旧框。
2. TextVQA 中重排有效：top-k 比旧双图再提高约 1.31 个百分点。
3. POPE 的 6 次答案变化全部是改善，说明候选验证对目标存在性问题较匹配。
4. GQA 包含关系、比较、计数和多对象推理，单一裁切框不一定覆盖全部证据；当前 generic-evidence verifier 在这里不能稳定判断哪个框更好。

一个 TextVQA 纠正案例是问题 34622：“What is one of the brands being advertised?”。10 个人工答案中 9 个是 `yamaha`、1 个是 `peugeot`；原图回答 `Peugeot`，重排选择 rank 2 后，top-k 与 Focus 都回答 `Yamaha`。

## 7. Focus 四阶段剪枝分析

Focus 相对 top-k 双图的变化：

| Benchmark | 答案变化 | Focus 改善 | Focus 退化 | 相对 top-k 分数变化 |
|---|---:|---:|---:|---:|
| TextVQA | 24 | 4 | 9 | -0.76 个百分点 |
| GQA | 18 | 7 | 3 | +0.78 个百分点 |
| POPE | 6 | 2 | 4 | -0.33 个百分点 |

因此，Focus 不是“始终无损压缩”。它在 GQA 上对关系推理有一定帮助，但在 TextVQA 与 POPE 上会丢掉少量 top-k 已获得的证据。当前固定 `50% × 3` 剪枝策略需要改成按题自适应，或者增加“不剪枝回退”。

GQA 案例 201770690 的答案是 `yes`：原图回答 `No`，top-k 选择 rank 2 后回答 `Yes`，Focus 仍保持 `Yes`。另一个 20226566 案例中，top-k 仍回答 `No`，Focus 才纠正为 `Yes`，说明剪枝有时确实改善信噪比。

## 8. POPE 分类结果

| 设置 | 原图 | 旧双图 | top-k 双图 | Focus |
|---|---:|---:|---:|---:|
| adversarial | 83.50 | 82.50 | **84.50** | 84.00 |
| popular | 86.00 | 89.00 | **89.50** | **89.50** |
| random | 92.00 | 93.50 | **94.00** | 93.50 |

![[analysis/figures/03_pope_categories.svg]]

原论文在 MSCOCO adversarial 上报告 baseline 81.83、SLoFo 85.97（+4.14）。本轮 adversarial 只有 +0.50，说明当前实现对最难的“高共现干扰物”仍明显不足；总体 POPE 增益更多来自 popular/random。

## 9. 时间与显存

| Benchmark | 总时间 | 平均/题 | 中位数 | P95 | Scan 峰值 | Generation 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| TextVQA 512 | 1,632.7 s | 3.060 s | 2.992 s | 3.629 s | **25,938 MiB** | 20,929 MiB |
| GQA 512 | 1,095.3 s | 2.019 s | 1.977 s | 2.536 s | 21,471 MiB | 18,678 MiB |
| POPE 600 | 1,327.5 s | 2.094 s | 2.005 s | 2.627 s | 21,349 MiB | 18,608 MiB |

这些时间是“完整逐题 pipeline”的墙钟时间，包括 Scan、5 个候选验证、原图/裁切/双图/Focus 多条回答和结果写盘。论文所说的 0.57 s 是 SLoFo 相对基础推理的 Scan-Locate + Focus 额外开销，两者不能直接比较。

服务器的 `nvidia-smi` 在 2026-08-09 21:33 报告 7 张 `NVIDIA GeForce RTX 4090`、每张 `49140 MiB`，驱动 595.84。它与此前口头描述的“7 张 3090、24 GB”不一致，应以本次机器实际报告为准。本轮 GQA/TextVQA 使用启动时为空的 GPU 1，POPE 使用 GPU 2；GPU 0 原有任务未被触碰。结束后 GPU 1、2 均恢复为 15 MiB。

## 10. 可视化样例与功能完整性

以下目录保留了每个 benchmark 首题的完整输出，可检查裁切、三张显著性图、top-k 候选框和 Focus 三阶段 token：

- [[samples/runs/textvqa_subset_512/cases/34607/result.json]]
- [[samples/runs/gqa_subset_512/cases/201497576/result.json]]
- [[samples/runs/pope_stratified_600/cases/7/result.json]]

TextVQA 样例 34607：

![[samples/runs/textvqa_subset_512/cases/34607/selected_bbox.png]]

![[samples/runs/textvqa_subset_512/cases/34607/ssim_overlay.png]]

![[samples/runs/textvqa_subset_512/cases/34607/topk_candidates_overview.png]]

该题问题为球衣号码，OCR reference 是 `22`；原图、裁切、旧双图、top-k 与 Focus 均回答 `22`。它用于确认整个可视化和多分支链路完整，不代表性能提升案例。

## 11. 尚未完成的问题

1. **全量官方评测尚未执行。** 本轮结果是开发阶段配对子集，下一次稳定版本应跑完 5,000 + 12,578 + 8,910 题。
2. **标准 24 GB 卡兼容性不足。** TextVQA 当前峰值 25.9 GiB，主要与长 OCR prompt、`output_attentions=True` 和多 token 语义梯度有关。需要分 token 回传、及时释放全层 attention，或恢复论文单 planning-anchor 作为低显存模式。
3. **GQA 的 top-k verifier 不可靠。** 需要针对关系、多对象与计数问题设计多区域证据评分，或在低置信度时回退旧框。
4. **Focus 应改为自适应。** 固定 50% 剪枝在 TextVQA/POPE 相对 top-k 有退化；建议依据 attention margin、候选置信度或答案熵调整剪枝率。
5. **没有定位真值。** 三个 benchmark 的标准指标评回答，不直接评裁切框。本轮只能通过答案变化和少量人工可视化判断定位质量。
6. **服务器项目根目录不是 Git 工作树。** 环境记录无法绑定 commit；本轮已用 [[metadata/used_files_sha256.txt]] 固定实际代码、数据和清单版本。

## 12. 建议的下一步

按优先级建议：

1. 先做 TextVQA 128 题显存消融：论文单 token、当前 8-token、分 token 梯度三种模式，目标把峰值压到 22 GiB 以下。
2. 对 GQA 的失败题按 `choose/compare/logical/query/verify` 分类，增加多框或原图回退规则。
3. 对 Focus 做 `25%/37.5%/50%` 剪枝率与动态不剪枝对照，重点观察 top-k 正确但 Focus 退化的题。
4. 固定代码哈希后，先跑三个 benchmark 的 1,024 题复验；结论稳定后再申请连续空卡跑全量官方分数。

## 13. 文件说明

- [[analysis/textvqa/textvqa_summary.json]]、[[analysis/gqa/gqa_summary.json]]、[[analysis/pope/pope_summary.json]]：机器可读统计。
- `analysis/*/*_cases.csv`：逐题问题、答案、分支得分、bbox 与 rank。
- `results/*/benchmark_answers.jsonl`：逐题聚合推理结果。
- `results/*/batch_summary.json`：逐题耗时和完成状态。
- [[metadata/environment.json]]：软件、模型和 GPU 环境。
- [[metadata/used_files_sha256.txt]]：本轮数据、清单和核心代码哈希。
- [[packages/slofo_vqa_benchmarks_2026-08-09.tar.gz]]：服务器完整轻量结果包的本地副本，包含全部逐题 JSON、日志及每个 benchmark 前 3 题完整可视化。
- 服务器原始目录：`/data/workspace/Gexuri_Project/HYG_LLaVA_SLoFo/benchmarks/official/`。

## 14. 参考来源

- 本地 SLoFo 论文：[[../Paper/SLoFo.pdf]]
- LLaVA 官方评测说明：https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
- TextVQA：https://textvqa.org/
- GQA：https://cs.stanford.edu/people/dorarad/gqa/
- POPE：https://github.com/AoiDragon/POPE
