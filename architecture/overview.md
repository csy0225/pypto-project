# 系统架构概览

pypto step3p5 栈把 5 个代码仓 + 1 个集成目标拼成一个端到端 serving
系统。本 doc 展示各部件做什么、数据怎么流。

## 大图

```
                          ┌─────────────────────────────┐
                          │  vLLM stepcast fork          │
                          │  (Phase 2 集成目标 ——        │
                          │   公司内部 gitlab fork)      │
                          │                              │
                          │  • LLM 引擎 + scheduler      │
                          │  • Continuous batching       │
                          │  • Sampler + tokenizer       │
                          │  • Paged KV cache 管理       │
                          │  • Step3p5Model.forward      │
                          │                              │
                          │  通过 monkey-patch           │
                          │  pypto.step3p5.vllm_backend  │
                          │  （在 pypto-lib）            │
                          └──────────────┬──────────────┘
                                         │ 调用
                                         ▼
                          ┌─────────────────────────────┐
   pypto-lib              │  decode_fwd                  │
   "模型 + kernel"        │  （一个融合 45 层 kernel）   │
                          └──────────────┬──────────────┘
                                         │ 由 ... 编译
                                         ▼
   pypto                  ┌─────────────────────────────┐
   "框架"                 │  pypto.ir.compile            │
                          │  • multi-level IR            │
                          │  • codegen passes            │
                          │  • 出 .so + .bin             │
                          └──────────────┬──────────────┘
                                         │ 用
                                   ┌─────┴─────┐
                                   ▼           ▼
   pto-isa                  ┌──────────┐  ┌──────────────────────┐
   "tile 库"                │ pto-isa  │  │      PTOAS            │
                            │ 虚拟     │  │   字节码 assembler    │
                            │ tile ISA │  │      (= ptoas-bin)   │
                            └──────────┘  └──────────────────────┘
                                         ▲
                                         │ 产字节码给 ...
                                         │
                          ┌─────────────────────────────┐
   simpler                │  PTO runtime                 │
   "执行层"               │  • AICPU + AICore dispatcher │
   （pypto submodule）    │  • 跨卡 IPC（shmem）         │
                          │  • collective                │
                          └──────────────┬──────────────┘
                                         │ 跑在 ...
                                         ▼
                          ┌─────────────────────────────┐
                          │  Ascend 910B / A2A3          │
                          │  • driver 25.5.2             │
                          │  • firmware 7.8.0.7.220      │
                          │  • CANN 9.0.0-beta.1         │
                          └─────────────────────────────┘
```

## 仓库角色

### pypto

编程框架。提供 Python DSL（`pypto.language`，
`pypto.language.distributed`），multi-level IR 和 codegen pass。把
`@pl.program` 编成 PTOAS 字节码 + host 侧 dispatch `.so`。

Runtime：`pypto/runtime/` 是 git submodule，指向 **simpler**。

### pypto-lib

Tensor 级 kernel 实现和端到端 LLM 模型。承载 step3p5 家族：

- `models/step3p5/decode_fwd.py` —— 融合 45 层 decode + lm_head
- `models/step3p5/decode_layer.py` —— 每层 dispatcher（dense vs MoE）
- `models/step3p5/{attention_full,attention_swa}.py` —— attention 变体
- `models/step3p5/moe.py` + 5 个 MoE 组件文件 —— MoE block
- `models/step3p5/weight_loader.py` —— HF safetensors → 每 rank bundle

Phase 2 集成代码会落在 `models/step3p5/vllm_backend/`（Phase 20 任务
1.1+）。

### pto-isa

Tile-ISA 虚拟实现。定义 tile 操作（matmul、reduce、broadcast 等），
pypto codegen 下沉到这些 op。硬件特定（我们这里是 Ascend 910B）。

### PTOAS

基于 LLVM/MLIR 的字节码 assembler。把 pypto 出的 MLIR 转成设备字节码
+ dispatch metadata。

二进制发布：`ptoas-bin`（当前 v0.45）—— 我们实际跑的 assembler；
PTOAS source 仓为参考 / 从源代码 build。

### simpler（pypto submodule）

PTO runtime。管 AICPU + AICore 的任务 dispatch、跨卡 shmem window
IPC、collective primitive。最 platform-touchy 的组件 —— Phase 16 三
剑合璧绑定主要就是为了让 simpler 跑得起来。

### vLLM stepcast fork（Phase 2 目标）

公司内部 vLLM fork，含 step3p5 模型实现
（`vllm/model_executor/models/step3p5.py`）。提供 decoder 之外所有部件：
tokenizer、sampler、KV cache 管理、请求调度、continuous batching。

Phase 2 集成：monkey-patch `Step3p5Model.forward` 调用 pypto-编译的
`decode_fwd`，替代 torch eager。详见
[`vllm-step3p5-mapping.md`](vllm-step3p5-mapping.md)。

## Decode 时数据流（Phase 2 v0.1 之后）

```
用户 prompt
    │
    ▼
vLLM tokenizer ─────────────► token_ids
    │
    ▼
vLLM scheduler ─────────────► batch（B 请求）
    │
    ▼
vLLM Step3p5Model.forward（已 monkey-patch）
    │
    ▼
pypto decode_fwd（编出的 .so）
    │
    ├─► 45 层（dense / MoE mixed-mode）
    │       │
    │       ├─► attention: QKV + RMS norm + RoPE + paged KV cache update + flash
    │       └─► MoE 或 dense MLP
    │
    └─► lm_head + rms norm
    │
    ▼
Logits [B, VOCAB]
    │
    ▼
vLLM Sampler ─────────────► 每 batch 元素的 next_token_id
    │
    ▼
vLLM 续 seq_lens、KV cache slot_mapping 推进，循环
```

KV cache 在 HBM，layout 在 vLLM 侧分配 + pypto kernel 访问之间通过
零拷贝 view 共享（详见 Phase 20 任务 1.3 `kv_bridge.py`）。

## Build 依赖顺序

从源代码 rebuild 时：

1. `pypto`（框架 —— 提供 pypto-lib import 的 Python DSL）
2. `pto-isa`（codegen 时用）
3. `PTOAS`（codegen 时用；通常被 `ptoas-bin` 替代）
4. `simpler`（pypto/runtime 下的 submodule）
5. `pypto-lib`（依赖上面所有）

部署机通常只要：
- `pypto-lib` 源码
- `pypto` 装好（`pip install -e <workspace>/pypto`）
- `simpler` build 好装上（通过 pypto submodule build）
- `pto-isa` 源码（被 `$PTO_ISA_ROOT` 引用）
- `ptoas-bin`（在 `$PATH` 和 `$LD_LIBRARY_PATH`）

参见 [`../deployment/version-matrix.md`](../deployment/version-matrix.md)
看锁定的版本。

## 相关文档

- [`vllm-step3p5-mapping.md`](vllm-step3p5-mapping.md) —— vLLM ↔ pypto
  op 映射，Phase 20 monkey-patch 用
- [`../phases/20-vllm-backend-monkey-patch.md`](../phases/20-vllm-backend-monkey-patch.md)
  —— 消费本 overview 的 Phase 20 设计
- [`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md)
  —— 硬件平台绑定
