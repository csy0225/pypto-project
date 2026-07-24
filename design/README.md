# 设计 Design

软件工程意义上的设计文档（HLD + LLD）。**写代码前先在这里落设计**。
与 session 流水（`../archive/`）、规划（`../planning/`）、部署（`../deployment/`）分开。

## 内容

| 文档 | 层级 | 用途 |
|------|------|------|
| [`00-context-and-goals.md`](00-context-and-goals.md) | — | 项目背景 / 目标 / 5 仓 + vLLM 全景。对外介绍第一份。 |
| [`step3p5-model-architecture.md`](step3p5-model-architecture.md) | — | **step3p5 模型本身**：config 参数 + 完整 48 层流程图（模型视角，与实现解耦）。 |
| [`whole-net/`](whole-net/README.md) | HLD+LLD | 整网集成：单 `@pl.program` 全 45 层、per-layer/block/整网三轴、N≥6 墙、DeepSeek 对照。 |
| [`vllm-pypto/`](vllm-pypto/README.md) | HLD+LLD | vLLM 与 pypto 同卡共驻：monkey-patch / sidecar / KV·weight IPC / co-tenancy。 |
| [`performance/`](performance/README.md) | HLD+LLD+跟踪 | step3p5 decode 性能优化专项：对照 v4-flash mega-kernel，12 个独立可并行子任务 + 跟踪表。 |

## HLD vs LLD 约定

| 层级 | 回答什么 | 不回答什么 |
|------|---------|-----------|
| **HLD**（System Design） | 模块边界、数据流、时序、8 卡协同、接口形状 | file:line、算法细节、不变量 |
| **LLD**（Detailed Design） | file:line、接口签名、算法步骤、不变量、数据 layout | 为什么这么分（那是 HLD 的事） |

每个子系统两个层级成对出现（`01-system-design.md` + `02-detailed-design.md`），
LLD 引用 HLD 的模块名，HLD 不引用 LLD 的实现细节。

## 什么内容归哪里

| 内容 | 去向 |
|------|------|
| 跨仓架构 / 模块图 / 数据流 | 本目录 |
| 单 phase 的任务清单 / 进度 | `../planning/phases/` |
| 部署 runbook（driver/firmware/CANN） | `../deployment/` |
| 事故复盘（error signature 定位） | `../postmortems/` |
| pypto kernel 编码坑 | sub-repo `pypto-lib/docs/known-pypto-pitfalls.md` |

## 相关

- 规划与里程碑：[`../planning/`](../planning/README.md)
- 当前阻塞：[`../blockers.md`](../blockers.md)
- 事故复盘：[`../postmortems/`](../postmortems/README.md)
- 参考资料与金标准：[`../reference/`](../reference/README.md)
