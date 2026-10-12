# SemanticSpan Selector 候选方案（已归档）

> 状态：已归档的替代设计；当前实现使用固定 16-token Page，不使用 semantic span。

日期：2026-09-04
状态：已归档；未作为正式 runtime 或实验结论。

早期讨论曾考虑以 128–256 token `SemanticSpan` 作为语义检索层，再展开成 16-token physical Pages。
该中间层会让检索粒度、预算归属和 Oracle 对齐变得含混，因此当前正式方案不采用它。

当前设计见
[`Selector_Prefill_Reused_Page_Design_2026-09-04.zh-CN.md`](../../current/Selector_Prefill_Reused_Page_Design_2026-09-04.zh-CN.md)：
主模型 prefill/decode 时直接为每个 16-token physical Page 产生 compact vector，Selector、retention、
预算和恢复始终保持 Page 粒度。

本文件只用于保留设计演进记录。后续若为 ANN 增加 coarse grouping，它只能是无语义权威的加速索引，
不能改变 Page 级最终分数、预算或 `SelectionPlan`。
