# 2026-08-06 experiment artifacts

本目录保存本轮 SLoFo Scan-Locate 实验的可核验精简结果。

## 保存内容

- 完整中文实验报告；
- 4 张 raw/min-max × original/padded 联系表；
- 10 个案例的 `result.json`、最终 `selected_bbox.png` 和 `crop.png`；
- 失败案例 03、05 的 semantic/structure/fusion overlay；
- 样例 01 的语义数值诊断结果；
- 10 个服务器运行日志。

原始结果目录约 140 MiB，另有一个 139 MiB 的重复压缩包。为了避免把重复
PNG、NumPy 中间数组和压缩包永久写进 Git 历史，本仓库没有上传这些机械性
中间产物。结果 JSON 保留了四种消融配置的 bbox、回答、张量形状、显存峰值
及数值统计；联系表和逐图结果足以人工核对本报告中的主要结论。

测试输入与中文/英文问题清单位于
[`../../images/test_08_06/`](../../images/test_08_06/)。
