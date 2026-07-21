# 接力上下文（Handoff）

> **这是 ephemeral 接力文档**——给"接着干"的人一页纸当前工作面。durable 的
> 规划看 [`roadmap.md`](roadmap.md)，此刻状态看 [`../STATUS.md`](../STATUS.md)，
> 历史流水看 [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)。
> **最后更新：2026-07-18。** 更新时直接改写本文（不追加流水）。

## 1. 当前结论（N=1 整网 standalone gate 已关闭）

```text
program = whole_decode_faithful_real
machine = gpu-a910x-0162
devices = 8..15
token   = 6127
P_FAITHFUL_MOE_LAYERS = 42
weights = native W8A8 IPC
KV      = IPC
dispatch = fixed-slot pull
combine  = pull
golden  = argmax 303
```

- **2026-07-18 single-submit 合入三仓 `stepfun/develop`** 并干净回归：
  P42 20/20 PASS，每次 `argmax=303`，`TOP5=[303,9592,1043,768,2086]`，
  runtime mean≈0.67s，fingerprint 唯一。
  三仓 pin：simpler `c7fdc574`、pypto `9ec303f6`、pypto-lib `e1513d22`。
- 0162 clean 环境唯一 stable 记录：
  [`../develop/N1/N1-STABLE-ENV-0162-20260717.md`](../develop/N1/N1-STABLE-ENV-0162-20260717.md)。
- 唯一验收入口：[`../reference/canonical-test.md`](../reference/canonical-test.md)。

## 2. 架构边界（收敛后，不要动）

- **dispatch**：`_dispatch_pack_publish`（本地 fixed-slot pack + pub_counts row）→ `_dispatch_pull`（AtomicAdd/Ge rendezvous，拉 counts_all，生成 recv_counts/CSR/inverse_map，self local-load + peer remote-load）→ `_dispatch_stage`（peer-major → expert-major compact）。
- **routed expert**：`local_routed_x` INT8 + `local_routed_x_scale` FP32，native INT8 gate/up/down matmul，signed tile remainder，**无 BF16 权重 dequant fallback**。
- **combine**：`_stage_routed_src` → `_pull_routed_y`（消费 dispatch 产出的 inverse_map）→ `_weighted_gather_and_add`；self 用 `pl.load`，peer 用 `remote_load`。
- **control signal**：逻辑 `[8,1] INT32`（32B），物理 `COMM_CONTROL_SIGNAL_BYTES=512`（216/216 signal 都 512B，offset %512=0）。
- **generator 已收敛**：不再用旧 A/B helper 把 inverse-map 重建移到 combine；任何生成器改动必须重跑 round-trip（`PRECOMMIT_ROUNDTRIP=PASS` / `ROUNDTRIP_CMP_RC=0`）。

## 3. 下一步（Phase 28 live serving）

standalone gate 与 live serving 是**两个独立结论**，不能混写成"serving 已完成"。

1. **恢复 0234 访问**，生成三仓/build/environment manifest 并复核历史记录的
   fresh-canonical stall（0234 只拉 pypto-lib 不足以复现，需全栈等价）。
2. 保持 0162 clean stack 为回归基线。
3. **live vLLM per-layer paged KV bridge**（现为单 flat pool）。
4. **消除 vLLM 与 exporter 的冗余权重**，解决 3-way HBM。
5. **live single-handoff token-exact A/B**（`e2_ab.py`，需可用 vLLM 容器）。

> live holder/sidecar/KV importer/容器 backend 是独立 live WIP，不在
> standalone release commit 内；进入 Phase 28 前先按工作区实际状态整理、
> review 并单独提交 live 组件。详见
> [`../design/vllm-pypto/02-detailed-design.md`](../design/vllm-pypto/02-detailed-design.md)。

## 4. 历史结论复核（防止重走）

1. 历史 `argmax=303` 证明数学路径正确，但不证明历史版本无 stall。
2. 旧 push/pull kernel 位置是线索，不足以证明某个 TPUT / signal bit 是唯一硬件根因。
3. `routed_h_quant` 不是统一挂点；失败 build 曾表现为 rank 8–14 卡 `_pull_routed_y`、rank 15 卡 `_dispatch_pull`。
4. fixed-slot / count-pull / signed tile / self local-load / AtomicAdd signal / per-layer distinct buffers 都有依据，保留。
5. 32B→512B control signal 是最小布局 A/B 变量，0162 上强关联 stall 消失，但不是跨机器充分条件或唯一硬件根因。

> 相关复盘：[`../postmortems/07-whole-net-scheduler-timeout.md`](../postmortems/07-whole-net-scheduler-timeout.md)、
> [`../postmortems/08-multiprogram-coprepare-deadlock.md`](../postmortems/08-multiprogram-coprepare-deadlock.md)、
> [`../postmortems/12-integration-churn-meta.md`](../postmortems/12-integration-churn-meta.md)。
