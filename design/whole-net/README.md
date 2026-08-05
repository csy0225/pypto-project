# 整网集成设计 Whole-Net Design

单 `@pl.program` 全 45 层 step3p5 整网的设计文档。Phase 27 是已归档的历史
single-program bring-up；当前产品入口和准出状态以根 `STATUS.md` 与
`reference/canonical-test.md` 为准。

## 内容

| 文档 | 层级 | 用途 |
|------|------|------|
| [`01-system-design.md`](01-system-design.md) | HLD | 模块组成、8 卡数据流、collective 时序、comm domain 划分。 |
| [`02-detailed-design.md`](02-detailed-design.md) | LLD | file:line、kernel 接口、per-layer window / signal buffer layout、不变量。 |

## 读法

- **新人**：从 [`01-system-design.md`](01-system-design.md) §1 开始 → [`../00-context-and-goals.md`](../00-context-and-goals.md)。
- **改 kernel**：先查 [`02-detailed-design.md`](02-detailed-design.md) 的接口/不变量 → 再去 sub-repo `pypto-lib/models/step3p5/`。
- **查"为什么整网卡"**：`../../postmortems/07-whole-net-scheduler-timeout.md` + `../../postmortems/08-multiprogram-coprepare-deadlock.md`。

## 相关

- vLLM 集成侧：[`../vllm-pypto/`](../vllm-pypto/README.md)
- 当前验收金标准：[`../../reference/canonical-test.md`](../../reference/canonical-test.md)
- Phase 27（历史单程序整网融合）：
  [`../../archive/completed-phases/27-single-program-whole-net-fusion.md`](../../archive/completed-phases/27-single-program-whole-net-fusion.md)
