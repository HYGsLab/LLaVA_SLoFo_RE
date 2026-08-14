# TextVQA 5000：A/E 断点与复现一致性检查

- 检查日期：2026-08-14
- 当前输出：5000 题 A、E 各自的前 1024 条
- 参照输出：2026-08-13 已完成的独立 A/1024 与 E/1024 实验
- 对齐键：`question_id`

## A 基线

两份输出均为 1024 条，题目顺序完全一致。以下功能字段的差异题数均为 0：

- `original_answer`
- `crop_answer`
- `legacy_joint_answer`
- `topk_joint_answer`
- `focus_answer`
- `selected_bbox`
- `legacy_bbox`
- `selected_rank`
- `selection_changed`

运行时测量字段存在轻微浮动：

- `scan_peak_allocated_mib`：74/1024 条数值不同；
- `generation_peak_allocated_mib`：206/1024 条数值不同。

## E 完整工程方案

E 的上述全部功能字段同样为 `0/1024` 差异；生成显存峰值也完全一致。只有：

- `scan_peak_allocated_mib`：185/1024 条数值不同。

## 结论

CUDA 显存峰值会受到进程重启、缓存状态和分配器状态影响，不参与答案或定位框计算。
A 从第 204 题恢复后，功能输出与独立 A/1024 完全一致；E 前 1024 题也与独立
E/1024 完全一致。因此当前 5000 题 A/E 输出可以继续用于全量配对分析。
