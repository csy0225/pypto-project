# 整网集成设计 Whole-Net Design

单 `@pl.program` 全 45 层 step3p5 整网的设计文档。N=1 standalone 已关 gate（Phase 27）；
N≥6 整网仍有随机 stall（见 `../../postmortems/07-whole-net-scheduler-timeout.md`）。

## 内容

| 文档 | 层级 | 用途 |
|------|------|------|
| [`01-system-design.md`](01-system-design.md) | HLD | 模块组成、8 卡数据流、collective 时序、comm domain 划分。 |
| [`02-detailed-design.md`](02-detailed-design.md) | LLD | file:line、kernel 接口、per-layer window / signal buffer layout、不变量。 |
| [`03-integration-axes.md`](03-integration-axes.md) | — | per-layer / per-block / 整网三轴分析 + N≥6 墙 + DeepSeek 对照（历史背景）。 |

## 读法

- **新人**：从 [`01-system-design.md`](01-system-design.md) §1 开始 → [`../00-context-and-goals.md`](../00-context-and-goals.md)。
- **改 kernel**：先查 [`02-detailed-design.md`](02-detailed-design.md) 的接口/不变量 → 再去 sub-repo `pypto-lib/models/step3p5/`。
- **查"为什么整网卡"**：[`03-integration-axes.md`](03-integration-axes.md) + `../../postmortems/07-whole-net-scheduler-timeout.md` + `../../postmortems/08-multiprogram-coprepare-deadlock.md`。

## 相关

- vLLM 集成侧：[`../vllm-pypto/`](../vllm-pypto/README.md)
- N=1 验收金标准：[`../../reference/canonical-test.md`](../../reference/canonical-test.md)
- Phase 27（N=1 fusion）：[`../../planning/phases/27-n1-whole-net-fusion.md`](../../planning/phases/27-n1-whole-net-fusion.md)
