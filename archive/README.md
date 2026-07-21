# 归档 Archive

归档截止时间冻结的历史内容。一旦写入就**只追加，不重写**。

## 内容

| 文档 / 目录 | 范围 | 用途 |
|------|------|------|
| [`milestones-2026-Q2.md`](milestones-2026-Q2.md) | 2026-Q2 各 session | **每日 session 流水 SSOT** + pin snapshot 历史 + 已解 blocker。 |
| [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md) | Phase 01-19（2026-05 至 2026-06-22） | pypto kernel 原型开发的压缩历史。 |
| [`completed-phases/`](completed-phases/) | Phase 20 / 21 / 22（×3）/ 23 / 24 / 25 | 已完成并归档的 phase docs。 |
| [`deliveries/`](deliveries/) | Step3p5 W8A8 交付快照 ×2 | 2026-06-26 prefill + e2e 精度交付报告与 tar 包。 |

## 子目录索引

- **`completed-phases/`**：`20-vllm-backend-monkey-patch.md`、`21-precision-validation.md`、
  `22-device-shared-inprocess.md`(+`-p1-env-0234.md`)、`22-perf-baseline.md`、
  `23-zero-copy-kv-ipc-validation.md`、`24-live-layer-replacement.md`、
  `25-whole-model-orchestration.md`。
- **`deliveries/`**：`step3p5-w8a8-e2e-delivery-20260626.md`、
  `step3p5-w8a8-prefill-delivery-20260626.md`。

## 什么内容归哪里

| 触发 | 文档 |
|------|------|
| Session 末尾 milestone 总结 | 追加 entry 到 `milestones-2026-Q2.md` |
| Blocker 解决 | "已解 blocker" entry 加到 `milestones-2026-Q2.md` + 从 `../blockers.md` 删掉 + 在 `../postmortems/` 落复盘 |
| Pin snapshot 移动（任意 fork push） | "Pin snapshot history" 追加一行到 `milestones-2026-Q2.md` |
| Phase 完成（从 `../planning/phases/` 移走） | 移入 `completed-phases/` |
| 精度交付 / release 快照 | `deliveries/` |

## dev host 上的散 phase docs（**不**在本仓）

`<dev-host>/data/chensiyu/hw_project/pypto/docs/step3p5/phases/01-19*.md`
那 26 个细节 phase doc **故意未迁入**本仓。它们含 session-specific 细节，仅有
历史参考价值；[`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
的压缩 summary 已经够用。

## 相关

- 活跃 phase：[`../planning/phases/`](../planning/phases/README.md)
- 路线图：[`../planning/roadmap.md`](../planning/roadmap.md)
- 此刻状态：[`../STATUS.md`](../STATUS.md)
- 已解事故复盘：[`../postmortems/`](../postmortems/README.md)
