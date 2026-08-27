# Phase 28 — 单程序整网 → vLLM live 集成

> **状态：🟡 进行中。最后更新：2026-08-27。**
> 本文只保留当前 live 集成工作面；历史 N1 bring-up 已归档到
> [`../../archive/completed-phases/27-single-program-whole-net-fusion.md`](../../archive/completed-phases/27-single-program-whole-net-fusion.md)。

## 1. 当前对象

```text
current pypto-lib  e6c7d8ec34a05c3051ccf0dd169639f40f041a57
current pypto      14de90fd74b3c0716f94b9d4eafdd004d4eaed73
current Main       models.step3p5.decode_fwd:whole_decode_step3p5
vLLM overlay       1b3e538c35999e62b6d24e0651b3a85b7d16c826
current image      stepfun-upgrade-20260826-r12
manifest           sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d
config             sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f
performance env    PYPTO_H4_RESIDENT=all
```

r12 已通过 registry/fresh digest、Main H4 all/none `126/128`、MTP
BS1/BS16、dep-only DFX 与 non-privileged device contract，最终合同
`1844/1844 PASS`。该结论仍是 standalone/sidecar admission，不等价于 live
request 已接管。whole-step 性能只来自 r11 source-overlay A/B/A，必须与显式
H4 env 一起引用，不能写成 r12 immutable 性能复测。

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
- r12 immutable image 已完成 Main precision、MTP BS1/BS16、dep-only DFX、
  registry/security admission；五仓 `stepfun/develop` 已同步。
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

## 6. 与 Attention 收口的关系

r12 standalone admission 已完成，Phase 28 不再等待新的基础镜像。
后续 live 工作必须使用该明确 digest，并单独处理 H4 deployment env、paged-KV、3-way HBM 与
Main→MTP；不能把 standalone source-overlay ITL/dep-only DFX 当作 live serving 准出。当前
standalone bs16×每请求64K 还会在约 16 GiB static-arena 分配时 OOM，这与 live
重复权重问题是两个容量口径，不能混写。下一步见 [`../handoff.md`](../handoff.md)。

## 7. 明确废弃的输入

- 历史 N1 branch/pin/stable-env 不能作为当前 checkout。
- 旧 `whole_decode_faithful_real*`、多 Main selector、`models/step3p5_opt` 不得恢复。
- 首 token `argmax=303` 只能作为 smoke，不能代替多步 precision。
- 0234 stall 不能外推到 0162 或 r12；它仍由独立 `N1-S-0234` blocker 跟踪。
