# TextVQA validation 5,000 题全量验证

本实验承接 1,024 题四配置因子消融，只运行两条必要路径：

- A：1 Token + raw + 单框，作为论文公式直译基线；
- E：8 Token + min-max + top-k=5，作为当前完整工程方案。

固定清单 `manifests/textvqa_full_5000.json` 包含 TextVQA 0.5.1 validation 的全部 5,000 条问答和 3,166 张唯一图片；前 1,024 条与上一轮清单完全一致。清单 SHA-256 为 `bd2062154b990d88f8bf5e0dfce0e4897009e667844c682365a1092981388c60`。

本阶段仍是当前工程实现的官方 split 全量验证。由于目标论文未公开代码，不能把结果表述为作者实现的完全复现。
