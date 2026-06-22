# 归档 Archive

归档截止时间冻结的历史内容。一旦写入就**只追加，不重写**。

## 内容

| 文档 | 范围 | 用途 |
|------|------|------|
| [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md) | Phase 01-19（2026-05 至 2026-06-22） | pypto kernel 原型开发的压缩历史 |
| [`milestones-2026-Q2.md`](milestones-2026-Q2.md) | 2026-Q2 各 session | session-by-session milestone 日志 + pin snapshot 历史 + 已解 blocker |

## 什么内容归哪里

| 触发 | 文档 |
|------|------|
| Phase 完成（phase 从 `phases/` 移到历史） | 追加段到 `prototype-phase-01-19-summary.md` 或它的后继 |
| Session 末尾 milestone 总结 | 追加 entry 到 `milestones-2026-Q2.md` |
| Blocker 解决 | "已解 blocker" entry 加到 `milestones-2026-Q2.md` + 从 `../blockers.md` 删掉 |
| Pin snapshot 移动（任意 push 到 fork） | "Pin snapshot history" 追加一行到 `milestones-2026-Q2.md` |

## dev host 上的散 phase docs（**不**在本仓）

`<dev-host>/data/chensiyu/hw_project/pypto/docs/step3p5/phases/01-19*.md`
那 26 个细节 phase doc **故意未迁入**本仓。它们含 session-specific 细
节，仅有历史参考价值；[`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
的压缩 summary 已经够用。

将来如果有需要（如 contributor 想追溯某个设计决策为何这么做），那些
文件在 dev host 上还可读。可以按需逐个迁入本归档，不影响实时跟踪器。
