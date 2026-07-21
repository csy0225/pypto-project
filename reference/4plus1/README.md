# 4+1 视图 4plus1 Views

pypto runtime 框架的 4+1 视图（framework-level）。源材料来自
`pypto_top_level_documents/pypto-runtime-arch-docs/`；本目录是对原文的学习总结
+ 脑图，**权威规格以原文档为准**。

## 视图

| 文档 | 视图 | 回答什么 |
|------|------|---------|
| [`02-logical-view.md`](02-logical-view.md) | Logical | 抽象机器由哪些概念组成、怎么分层、谁依赖谁。 |
| [`03-development-view.md`](03-development-view.md) | Development | 源码组织、构建、依赖、sub-repo 边界。 |
| [`04-process-view.md`](04-process-view.md) | Process | 进程/线程/通信、chip_process、worker、orchestrator。 |
| [`05-physical-view.md`](05-physical-view.md) | Physical | 8 卡 NPU 拓扑、HBM / DDR / IPC、driver/firmware 映射。 |

## 读法

- **先建立全局**：从 [`02-logical-view.md`](02-logical-view.md) 的脑图开始。
- **再落到部署**：[`05-physical-view.md`](05-physical-view.md) ↔ [`../../deployment/`](../../deployment/README.md)。
- **需细节时**跳回原文档对应 §。

## 相关

- pypto 编程 API：[`../pypto-programming-api.md`](../pypto-programming-api.md)
- canonical 验收：[`../canonical-test.md`](../canonical-test.md)
- 整网设计：[`../../design/whole-net/`](../../design/whole-net/README.md)
