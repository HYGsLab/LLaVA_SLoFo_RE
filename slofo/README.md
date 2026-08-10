# SLoFo Scan-Locate（独立学习版）

这一目录包含论文的 **Scan-Locate** 张量算法和 **Focus** 四阶段剪枝核心；LLaVA 运行时适配器放在 `../remote_runtime/scripts/`，方便把算法、模型接线和实验入口分开学习。

## 已实现的计算链

```text
第 14 层：规划锚点 → 视觉 token 注意力 ─┐
                         × ReLU(梯度) ─┤→ 语义重要性 A_v ─┐
第 7 层：视觉 token 隐状态 → PCA → 重构误差 E_rec ─────┤
                                                       ├→ SSIM → 多尺度滑窗 → 裁切框
                     A_SSIM = 0.7 A_v + 0.3 E_rec ─────┘
```

这里的 SSIM 是论文定义的 **Semantic-Structural Importance Map（语义-结构重要性图）**，不是常见的图像结构相似度指标。

核心入口是 `scan_locate_from_tensors(...)`：

```python
from slofo import ScanLocateConfig, scan_locate_from_tensors

result = scan_locate_from_tensors(
    planning_attention=attention_to_576_visual_tokens,
    attention_gradient=gradient_with_the_same_shape,
    structure_hidden_states=layer_7_visual_states,  # [576, hidden_size]
    image_size=(image_width, image_height),
    config=ScanLocateConfig(),
)

print(result.crop.bbox)
print(result.ssim_map.shape)  # torch.Size([24, 24])
```

如果手中有可微的伪 token 标量得分，也可以不传 `attention_gradient`，改传 `planning_score`，模块会用 `torch.autograd.grad` 求梯度。

## 哪些是论文默认值

- 语义分支层：14
- 结构分支层：7
- PCA 主成分数：20
- 语义/结构融合系数：0.7 / 0.3
- LLaVA-1.5 视觉 token 网格：24 × 24（576 个 token）
- 基础裁切边长：336

论文正文只说扫描“若干窗口尺寸”，没有列出具体集合。本实现的默认比例 `(1.0, 1.2, ..., 2.0)` 沿用 SLoFo 所基于的 ViCrop 官方公开代码，并保留为可配置项。

论文公式也没有声明在融合前归一化两个分支，所以默认 `fusion_normalization="none"`，严格执行公式。为了后续消融验证，代码显式提供 `"minmax"` 选项，但它不是这一版默认行为。

## LLaVA 接线所需张量

运行时适配器从一次 LLaVA 前向传播中提取：

1. 第 14 层最后一个输入 token（规划锚点）指向 576 个视觉 token 的注意力；
2. 预测出的第一个伪 token 的标量 log-prob，以及它对上述注意力的梯度；
3. 第 7 层对应 576 个视觉 token 的隐状态。

适配器再把这三个张量交给本目录代码即可。这样模型修改、4-bit 加载和本算法不会混在同一个文件里。

## Focus 四阶段剪枝

`focus.py` 实现与模型无关的 token 分类、阶段边界和稳定剪枝。对于
32 层 LLaVA-1.5-7B，四个等长阶段的前三个边界是 7、15、23（从 0
开始计层）。每次删除剩余原图 token 中注意力最低的 50%，裁切图和文本
token 不删除，所以原图 token 数量严格遵循：

```text
576 -> 288 -> 144 -> 72
```

`../remote_runtime/scripts/slofo_focus_runtime.py` 把该逻辑接入 LlamaModel：
各层真实接收缩短后的 hidden states，生成时按每层实际 KV cache 长度继续
position id。实验入口可用 `--enable-focus --focus-phases 4
--focus-prune-ratio 0.5` 开启，并保存每阶段保留 token 的可视化。

为验证注意力排序是否真的优于“任意删掉同样数量”，`FocusConfig` 还提供
`selection_method="random"` 与独立 `random_seed`。随机模式只用于配对
对照，阶段、数量、裁切图和 KV cache 路径都与注意力 Focus 相同。

## 多 token 语义锚点实验

运行时支持多个贪心生成 token 的 teacher-forced 规划锚点。`mean` 聚合全部
锚点；`max` 保留任一锚点的最强响应；`--semantic-anchor-start-index 2`
可跳过常见的 `The person` 前缀，专门研究动作/内容 token。它们是为复杂
动作短语增加的显式实验选项，不冒充论文默认的单规划锚点设定。
## 2026-08-06 integration note

The LLaVA adapter in `../remote_runtime/scripts/run_slofo_scan_locate.py` now
supports a true original-plus-crop prompt with two image placeholders and two
processed image tensors.  It also saves original-only, crop-only, and joint
answers for comparison.

When LLaVA runs in float16, attention and gradient are converted to float32
*before* their element-wise product.  This is necessary because otherwise
small positive saliency values underflow to zero.  The change is protected by
`../tests/test_slofo_scan_locate.py`.

Raw semantic and structural maps still have very different numerical scales;
the optional `fusion_normalization="minmax"` remains an explicit ablation
rather than an asserted paper default.  A ten-image comparison currently uses
`minmax + original-coordinate mapping` as the working baseline.
