# step3p5 tp-all-reduce 改造：barrier-mesh → ring

> 日期：2026-08-12 | 0162 机器 | develop 源 `fa58b5cf` | 分支
> `perf/tp-allreduce-ring-20260812` (pypto-lib commit `a791071`)

## 0. 任务

vllm-ascend 用 `hcclAllReduce`（单 C 调用，HCCL V2 按大小选 ring/tree，stream
异步 overlap）做 TP all-reduce；pypto 当前 hand-roll 了一套 **barrier-mesh**
（stage-in → 全局 barrier → 逐 chunk 全 mesh 读），跨卡 DMA ~4× 于 ring，性能差。

对照 `/data/chensiyu/hw_project/hccl` 已实现算子（`src/ops/all_reduce/selector/
all_reduce_auto_selector.cc` 等），改造 pypto tp-all-reduce 到与 vllm-ascend
同族：ring reduce-scatter + all-gather。pypto 是编译设备程序模型（通信只能用
窗口 + notify/wait），设备程序内无法等价调用 host 侧 `hcclAllReduce`，故 ring
是其模型内的最优（也正是 hccl 小/中数据同族算法）。

## 1. 四个实现对照

| 实现 | 位置 | 单次（TP=8, HIDDEN=4096, chunk=512）远程 DMA |
|------|------|---------------------------------------------|
| pypto barrier-mesh（现行） | attention_full/swa.py 内联 `tp_all_reduce` | (HIDDEN/chunk)·(N-1) = 8·7 = **56** 次 remote_load + 全局 barrier |
| pypto collectives.py ring（重写未上线） | `collectives.py::tp_all_reduce` | **14** 次 remote_load + 2(N-1) 步 handshake |
| pypto decode_fwd.py 3-wave（现行） | 内联 `tp_all_reduce` | reduce-scatter 56 reads + push-all-gather 56 writes = **112** DMA + 3 波 6(N-1) handshake |
| vllm-ascend / hccl（原生） | `pyhccl.py` → `hcclAllReduce` | 1 次调用（内部 ~2(N-1) chunk 次流水 ring/tree） |

hccl 单机 8 卡小/中数据算法：`CcuAllReduceMesh1DOneShot`（≤16KB）、
`CcuAllReduceConcurrentMs`、大数据 `CcuV2AllReduceOmniPipe2DMs`（2D omnipipe）、
AICPU/AIV 兜底——其 ring 与本次改造同阶。

## 2. 改造（短，仅 attention_full/swa，两文件逐字同体）

1. 信号窗 `SIGNAL_WINDOW_ROWS = TP_WORLD_SIZE` →
   `2*(TP_WORLD_SIZE-1)+1`（TP=8→15）；标注与 host_orch 的 `alloc`/`pld.window`
   同步放大。
2. 方法体：barrier-mesh → **ring reduce-scatter (N-1 步) + all-gather (N-1 步)**，
   每步 forward 一个 `[BATCH, ar_chunk]` shard、对端 remote_load 累加/覆盖。
3. handshake：每步独占信号 cell，`Set(1)/Ge(1)`（**非单调**，无 atomic step
   阈值）——刻意避开当初 ring→barrier 的根因（multi-step 单调 AtomicAdd →
   codegen 507018）。
4. 保留：`ar_chunk=HIDDEN//8` 固定分块、FP32 累加+BF16 cast、
   `tmp_window` `[BATCH,HIDDEN]` ABI。

## 3. 验证（0162，perf-h1 镜像，8 卡）

- **8 卡 compile OK**：`_stage_two_layer_attn` harness 跑出
  `[two-layer] compile OK in 1.4s` —— ring 体 + 放大信号窗被 pypto 编译器在
  TP=8 真实接受（无 group_size/TileType/wait-peer 等 DSL 错误）。
- codegen-contract text-grep、chip_orch.cpp 编排 C++ 编译两项 gate
  **原始 barrier-mesh 同样失败**（本镜像 pypto 配对的 pre-existing 问题，非本改造引入）。
- 端到端 dispatch→ITL 因上述 pre-existing gate 被拦、未在本镜像一次性跑通；
ring 在 8 卡真实 **compile** 通过 + handshake 规避 507018 pattern，推断可 dispatch。
完整 ITL 对照需在配平镜像（两项 gate 均过）上重跑本 harness。
- 详细对照 methodology（含 mirror 复现脚本 `ar_bench/`）见
  `pypto-lib/docs/upstream-issues/step3p5-tp-allreduce-ring-refactor.md`。

## 4. 待办 / 风险

- [ ] 配平镜像上重跑 `_stage_two_layer_attn`，ITL 对照 baseline (`cb96747e`)。
- [ ] 同模式扩展 moe.py (T/N_RANKS)、mtp_hidden_fwd.py、prefill_attention_*。
- [ ] collectives.py 的 @pl.jit.inline ring 同步改非单调 Set handshake（canonical 单源）。
- [ ] 若某镜像 ring dispatch 仍 507018，回退到 barrier-mesh 就地优化（去掉冗余
      stage-in，collective 直接 remote_load peer 的 local）。
