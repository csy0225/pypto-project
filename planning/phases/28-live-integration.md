# Phase 28 — 单程序整网 → vLLM live 集成

> **状态：🟡 进行中。最后更新：2026-08-05。**
> 本文只保留当前 live 集成工作面；历史 N1 bring-up 已归档到
> [`../../archive/completed-phases/27-single-program-whole-net-fusion.md`](../../archive/completed-phases/27-single-program-whole-net-fusion.md)。

## 1. 当前对象

```text
current pypto-lib  91c7f46ee949045e2fce807276412b48d8121763
current pypto      8e92b46808f9f7c09b6431ad4691503f09c12ee5
current Main       models.step3p5.decode_fwd:whole_decode_step3p5
vLLM overlay       1b3e538c35999e62b6d24e0651b3a85b7d16c826
last qualified     stepfun-develop-20260803-attn-final-wave5
manifest           sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32
```

当前源码 tip 与最后 release-qualified 镜像必须分开。R1 已废弃；R2 build 暂停且
未发布/验证，不能继承 Wave5 的准出结论。

## 2. Goal

让真实 vLLM online request 的 decode forward 进入 PyPTO 单程序整网 Main，同时保持：

- vLLM 负责 request scheduling、paged KV、sampling 和 serving 生命周期；
- PyPTO 负责 45 层 decoder forward；
- native W8A8，不用 BF16-dequant fallback；
- dynamic batch/context metadata 正确映射；
- Main→MTP 使用同代 absolute oracle；
- 同卡共驻时 HBM 预算可持续。

## 3. 已完成

- 默认 Main 已统一为 `whole_decode_step3p5`，旧 unroll/opt/rollback 入口已删除。
- standalone Main、batch16、MTP focused gate 和 Wave5 immutable release gate 已完成。
- Attention/Vec、TP all-reduce source publication/lifetime 已在 Wave5 对 0162 准出。
- sidecar/holder、权重 IPC、KV metadata schema 和 vLLM overlay 已有实现基础。

这些证据只证明 standalone/sidecar 组件，不等价于真实 production request 已无条件平替。

## 4. 当前未闭环项

### 4.1 独立 live front 接管

必须证明真实 online request 的 decoder forward 实际进入 PyPTO runner，而不是
vanilla fallback、shadow 或离线 replay。

### 4.2 Paged-KV 与 dynamic batch

从真实 vLLM KV pool 导入 per-layer K/V slice，并正确传递：

```text
positions
seq_lens
block_table
slot_mapping
active batch / active token count
```

必须验证历史 row 不被覆盖、inactive row 不参与计算、跨 step KV 可见性正确。

### 4.3 Main → MTP 同代 absolute gate

MTP 输入必须来自本次 Main 的配对 hidden；禁止拿旧 N1 artifact 或不同代 fixture
作为 absolute oracle。报告 token、hidden、finite、TP spread 和 acceptance state。

### 4.4 3-way HBM / redundant weights

同时常驻 vLLM 权重、exporter IPC 权重和 PyPTO working set 会放大 HBM。需要通过
共享/复用或明确生命周期消除重复权重，不能靠调小 ring heap 掩盖真实预算问题。

## 5. 准出条件

1. live request 路径证据明确，fallback/shadow 状态可观测。
2. paged-KV/dynamic batch 在 bs1～16（必要时 32）与多步 decode 下正确。
3. vanilla raw alignment、revision equivalence、MTP absolute gate 分开报告。
4. 无 stall、无残留 exporter/chip 进程、TP spread=0。
5. HBM 峰值和三类权重 ownership 有可复核账本。
6. 最终结论来自 immutable digest，不使用宿主源码挂载。

验收标准：[`../../reference/canonical-test.md`](../../reference/canonical-test.md)。

## 6. 与当前 Attention R2 的关系

Attention R2 的 bs1/64K 两层 DFX和整网 ITL是当前更高优先级交付；build 已按用户要求
暂停。Phase 28 不得抢占或改写 R2 pin。恢复顺序见
[`../handoff.md`](../handoff.md)。

## 7. 明确废弃的输入

- 历史 N1 branch/pin/stable-env 不能作为当前 checkout。
- 旧 `whole_decode_faithful_real*`、多 Main selector、`models/step3p5_opt` 不得恢复。
- 首 token `argmax=303` 只能作为 smoke，不能代替多步 precision。
- 旧 0234 stall 记录不再列为当前 active blocker。
