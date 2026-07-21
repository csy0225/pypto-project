# 参考资料 Reference

canonical 测试、pypto runtime 框架视图、编程 API、kernel 约束的参考资料。
**不是设计**（设计在 [`../design/`](../design/README.md)）；**不是 runbook**
（runbook 在 [`../deployment/`](../deployment/README.md)）。

## 内容

| 文档 | 用途 |
|------|------|
| [`canonical-test.md`](canonical-test.md) | **N=1 whole-net 唯一验收金标准 SSOT**（P42 → token 6127 → argmax 303）。任何"跑通/精度"结论只能由本测试给出。 |
| [`4plus1/`](4plus1/README.md) | pypto runtime 框架 4+1 视图（02 逻辑 / 03 开发 / 04 过程 / 05 物理）。 |
| [`pypto-programming-api.md`](pypto-programming-api.md) | pypto frontend 编程 API 速查（`pl.range` / `pl.slice` / `pl.program` / collective 等）。 |
| [`cache-line-signal-isolation.md`](cache-line-signal-isolation.md) | 跨卡 signal / cache-line 隔离探针与诊断方法。 |
| [`aclgraph-vs-pypto.md`](aclgraph-vs-pypto.md) | aclGraph 与 pypto 编译模型对照。 |
| [`moe-constraints.md`](moe-constraints.md) | MoE kernel / dispatch / combine 的硬约束清单。 |
| [`moe-routed-live-wiring.md`](moe-routed-live-wiring.md) | routed expert live 接线路径（vLLM ↔ pypto）。 |
| [`zero-copy-ipc-route.md`](zero-copy-ipc-route.md) | 跨卡 zero-copy IPC（weight / KV）路径设计。 |

## 什么内容归哪里

| 内容 | 去向 |
|------|------|
| canonical 验收口径 / golden token / 命令 | `canonical-test.md` |
| pypto runtime 框架概念 | `4plus1/` |
| pypto frontend API | `pypto-programming-api.md` |
| 跨仓架构 / 模块图 | `../design/` |
| 部署版本绑定 | `../deployment/` |
| 事故复盘 | `../postmortems/` |

## 相关

- 设计文档：[`../design/`](../design/README.md)
- 部署：[`../deployment/`](../deployment/README.md)
- 事故复盘：[`../postmortems/`](../postmortems/README.md)
- 术语表：[`../GLOSSARY.md`](../GLOSSARY.md)
