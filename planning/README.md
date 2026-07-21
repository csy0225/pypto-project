# 规划 Planning

整体规划（durable）。**与 session 流水分开**——每日进展在
[`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)，
此刻状态在 [`../STATUS.md`](../STATUS.md)。

## 内容

| 文档 | 范围 | 用途 |
|------|------|------|
| [`roadmap.md`](roadmap.md) | 全项目 | 路线图 / 里程碑 / 阶段定义。durable。 |
| [`handoff.md`](handoff.md) | 当前工作面 | ephemeral 接力上下文，给"接着干"的人一页纸。 |
| [`phases/`](phases/README.md) | 活跃 phase | Phase 27 / 28 任务清单与进度。 |

## 什么内容归哪里

| 触发 | 去向 |
|------|------|
| 整体路线 / 里程碑定义 | `roadmap.md` |
| 当前 session 的接力上下文 | `handoff.md`（覆盖写） |
| 单个 phase 的任务 / 进度 | `phases/NN-*.md` |
| Session 末尾 milestone 总结 | `../archive/milestones-2026-Q2.md`（追加） |
| 当前阻塞 / pin snapshot | `../STATUS.md` + `../blockers.md` |
| 已完成的 phase | `../archive/completed-phases/` |

## 相关

- 此刻状态：[`../STATUS.md`](../STATUS.md)
- 阻塞清单：[`../blockers.md`](../blockers.md)
- 每日流水：[`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
- 设计文档：[`../design/`](../design/README.md)
