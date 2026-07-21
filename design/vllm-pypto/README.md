# vLLM + pypto 集成设计 vLLM-PyPTO Design

vLLM 与 pypto worker 同 8 卡共驻的设计文档。live single-handoff 集成在 Phase 28 🟡。

## 内容

| 文档 | 层级 | 用途 |
|------|------|------|
| [`01-system-design.md`](01-system-design.md) | HLD | vLLM / pypto 分工、同卡共驻拓扑、请求链路、co-tenancy 开关。 |
| [`02-detailed-design.md`](02-detailed-design.md) | LLD | monkey-patch 点 file:line、sidecar 协议、KV·weight IPC 接口、不变量。 |
| [`03-vllm-op-mapping.md`](03-vllm-op-mapping.md) | — | vLLM op → pypto kernel 映射表（attention / rmsnorm / moe / rope 等）。 |

## 读法

- **新人**：[`01-system-design.md`](01-system-design.md) → [`../00-context-and-goals.md`](../00-context-and-goals.md)。
- **接线**：[`02-detailed-design.md`](02-detailed-design.md) 给 monkey-patch hook 点 + sidecar socket 协议。
- **查 op 对照**：[`03-vllm-op-mapping.md`](03-vllm-op-mapping.md)。
- **撞到 live 运维事故**：`../../postmortems/11-8001-bridge-live-ops.md` + `../../postmortems/03-hccl-cotenancy.md`。

## 相关

- 整网侧：[`../whole-net/`](../whole-net/README.md)
- Phase 28（live 集成）：[`../../planning/phases/28-n1-live-integration.md`](../../planning/phases/28-n1-live-integration.md)
- KV-IPC 路径参考：[`../../reference/zero-copy-ipc-route.md`](../../reference/zero-copy-ipc-route.md)
- 部署（co-tenancy 运行时）：[`../../deployment/`](../../deployment/README.md)
