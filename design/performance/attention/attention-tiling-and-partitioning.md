# Attention task 切分与 tile 校准（最终实现）

> **状态：2026-08-02 current source。** 权威代码为
> `pypto-lib stepfun/develop@76d96bdbeac280f12ecf626b1bbd722b9278719e`；
> 动态 SPMD launch bound 所需 codegen 修复为
> `pypto stepfun/develop@defa97c526fec7e8f032dbbfcc39c820add02bf7`。
> 旧文档中的 fixed-24、四阶段 Full online-softmax、standalone Pass-A/B/C 和
> out-proj cast 默认关闭均为历史 probe，不代表当前产品实现。

## 1. 核心结论

当前实现选择的是 **workload-derived logical tasks + runtime wave mapping**，不是固定
24 个物理核心：

```text
logical_tasks(row, stage) = ceil(actual_work(row) / grain(stage))
total_tasks(stage)        = sum(logical_tasks(active rows))
```

PyPTO orchestration 只描述 logical task DAG；runtime 根据目标架构的 AIC/AIV 数量、
可用资源和调度状态把这些 task 分成一个或多个 wave。由此同时支持：

- 按物理资源校准，使任务数接近一个或若干完整 wave；
- 按工作量校准，使不同架构可以选择不同 grain；
- active batch 与异构 `seq_len`，避免按最大 context 给所有 row 铺相同任务；
- `BATCH=16` 只作为 storage capacity，而不是 16 个永久 logical batch。

**单 task 5–10 us 只是 sweep 的起始区间，不是目标函数。** 最终选择必须联合比较：

```text
per-task duration + stage span + wave/core-wait + packing + tail
+ dispatch overhead + 后续 reduction/finalize 依赖尾部 + batch16 表现
```

在 A2A3 上，某些 5–10 us 配置因进入额外 wave，反而慢于约 15–20 us 的单-wave配置。

## 2. shape、storage 与逻辑工作量

| 参数 | Full | SWA |
|---|---:|---:|
| TP-local Q heads | 8 | 12 |
| Q physical pad | 16 | 24/32（阶段相关） |
| `HEAD_DIM` | 128 | 128 |
| KV cache block | 128 tokens | 128 tokens |
| 最大有效 KV blocks | context 决定；64K 为 512 | window 512 tokens，即最多 4 |
| storage batch capacity | 16 | 16 |

必须区分：

1. **storage block/tile**：例如 KV block 128 tokens、cube 合法 `N=64`；
2. **logical task grain**：一个 task 连续处理多少 block/tile；
3. **physical resource mapping**：runtime 把 logical tasks 映射到 AIC/AIV；
4. **active workload**：实际 active rows 和每行真实 context；
5. **capacity**：静态 tensor 上限，不应生成 inactive-row 工作。

因此“两个 128-token block 由同一个 task 处理”不等于已经形成一个 256-token fused
matmul；它只是调度 grouping，除非实现还完成 operand gather 与新的合法 tile lowering。

## 3. Full attention 当前任务图

```text
full_qk_matmul
  -> full_softmax
  -> full_sv_matmul                  # SV + segment-local recurrence
  -> full_online_softmax_reduce      # write-disjoint group reduction
  -> full_online_softmax_finalize    # per-row final merge/normalize/BF16 store
  -> full_out_proj_matmul_{aic,aiv}  # FP32 accumulator -> BF16 cast 已融合
```

在默认 decode 配置（`FUSE_CAST=1`）下，不会生成独立的：

```text
full_online_softmax_pass_a
full_online_softmax_pass_b
full_online_softmax_pass_c
full_out_proj_cast
```

`FUSE_CAST=0` 的 decode fallback 分支仍保留；prefill 路径的
`prefill_full_out_proj_cast` / `prefill_swa_out_proj_cast` 也不属于本节的
默认 decode graph。

### 3.1 QK / block-softmax

对每个 active row，先由真实 `seq_len` 得到 `context_blocks`，再按 stage grain 计算 task
数。task 映射到 `(batch_row, task_in_row, block_start)`，不会按最大 context 给短 row
补齐无效 task。

A2A3 默认：

```text
QK blocks/task       = 22
softmax blocks/task  = 12
```

这些是 architecture profile，而不是语义常量。以 64K、bs=1 为例：

```text
QK       ceil(512 / 22) = 24 logical tasks
softmax  ceil(512 / 12) = 43 logical tasks
```

它们分别由 AIC/AIV runtime 资源形成 wave；源码没有写死“使用 24 核”。

### 3.2 SV 与 online-softmax

`full_sv_matmul` 现在同时完成：

1. 每个 KV block 的 `P @ V`；
2. 同一 task 所拥有 segment 内的 `(m,l,o)` recurrence；
3. 写出一个 segment partial。

因此历史 Pass-A 已消失。当前 A2A3 profile：

```text
SV + segment recurrence blocks/task = 16
reduce fan-in                       = 8
```

随后两个 kernel 仍需保留：

- `full_online_softmax_reduce`：不同 task 产生的 segment partial 必须跨 task 合并；每个
  reduce task 写自己 group 的唯一 destination，避免 concurrent-writer race；
- `full_online_softmax_finalize`：每个 active row 合并 group outputs、normalize、flatten，
  并完成 FP32→BF16 最终 store。

若把二者机械并入所有 SV task，要么多个 task 写同一 row 形成 race，要么退回单 task
串行消费整行而丢失 context parallelism。它们是必要的 RAW/liveness 边界，不是遗留拆分。

## 4. SWA 为什么保持不同结构

SWA 的有效 window 最多 4 个 KV blocks。当前每个 active row 是一个 logical task，task 内
顺序处理完整 window：

```text
swa_qk_matmul -> swa_softmax -> swa_sv_matmul -> swa_online_softmax
```

`SWA online_softmax` 的代表性执行仅约 3 us；再拆层次归约会增加 scratch、dispatch 和依赖，
没有收益证据。因此 Full 与 SWA 不应机械共享相同 task graph：

- Full：长 context，多 work item，适合 workload-driven context split 和层次归约；
- SWA：window 很短，优先保持单 active-row task 的计算密度。

## 5. out_proj 参数与 cast fusion

Full/SWA 分别保留 profile knob，因为两者 K width 和周边 stage timing 不同；当前 A2A3
默认恰好相同：

```text
matmul N tile        = 64        # 当前 910B lowering 的合法 cube tile
matmul tiles/task    = 3
vector N             = 128
cast fusion          = 1
```

`N=64 × 3 tiles/task` 把 `4096/64=64` 个合法 N tiles 分成 22 个 logical tasks，约为
A2A3 的一个 AIC wave，但这只是 calibration 结果。换架构后可以分别覆盖 Full/SWA 的
`MATMUL_TILES_PER_TASK`，无需改数学实现。

cast 融合后的数据流为：

```text
FP32 matmul accumulator -> same mixed task AIV cast -> BF16 partial output
```

因此默认 decode 生成物没有 `full_out_proj_cast` 或 `swa_out_proj_cast`。独立开关仍
保留，便于新架构发现 mixed-kernel 不合适时回退；这不表示 fallback 或 prefill
路径中的同名/相关 kernel 被从源码仓库删除。

## 6. active batch=16 与异构 context

### 6.1 capacity 不是 workload

`BATCH=16` 只决定 tensor/ABI 的 storage 上限。logical tasks 由 `active_tokens` 和每行
`seq_lens` 推导；inactive rows 不参与 attention/KV metadata 工作。

### 6.2 已验证边界

active-batch=16、ctx=1 的 canonical smoke 已证明 16 行都 finite/nonzero，TP spread=0。
异构 16-row context（从 65536 递减到 1）也完成 DFX，task 数按各行 `ceil` 求和，而不是
按 65536 给每一行铺满。

uniform batch16 / 64K 的 Full online grain 单轮对比：

| grain | logical tasks | 两层 wall p50 |
|---:|---:|---:|
| 16 | 512 | 5.5590 ms |
| 24 | 352 | 5.5494 ms |
| 32 | 256 | 5.6126 ms |

16 与 24 只差约 0.17%，不足以把 batch-aware 分支硬编码进模型语义。当前保留 bs=1
校准出的默认 16；如果未来 batch16 是主要服务点，应补多轮 median，再形成独立 architecture/
workload profile。

## 7. all-reduce 与 Vec 小算子

canonical TP all-reduce 使用单个 InCore kernel：

```text
stage local BF16 window
-> Wave 1 notify/wait
-> rank-owned reduce-scatter（固定 peer 顺序，单 FP32 accumulator，一次 BF16 cast）
-> push all-gather
-> Wave 2 notify/wait
-> copy completed window to local output
```

当前 transfer grain 为 512 columns，是通信 profile，不应被 residual Vec epilogue 机械继承。
已验证结论：

- producer 直接写 AR window：正确，但没有稳定收益；不合入；
- AR final copy + residual：512 粒度明显变慢，128 粒度仅噪声级收益；不合入；
- residual + RMS stats：三个粒度都变慢；不合入；
- dense RMSNorm：删除 `[16,4096]` FP32 GM staging，直接两遍读取 BF16 tile；保留；
- dense down-proj cast：融合进 mixed matmul task；保留；
- gate/up + SiLU cast、RMSNorm + projection：可运行但无稳定收益或破坏 split；仅保留 probe。

这体现同一原则：**能融合不等于应该融合**。必须同时满足 correctness、稳定收益、资源映射、
batch16 和最小改动。

## 8. PyPTO 架构边界

当前改动符合既有分层：

- Orchestration 负责由 runtime scalar 构建 logical task DAG 与 dependency；
- InCore task 只执行自己的 tile/segment，不在 worker 内递归 submit；
- runtime 决定 logical task 到物理核和 wave 的映射；
- task-grain 参数属于 architecture profile，不进入模型数学语义；
- 未增加 app-side work-stealing loop。

用户提出的“每个核去拉取 5–10 us task”在现有 runtime 中已经以 logical task scheduler
的方式实现了核心效果：task 数可超过物理核，runtime 分 wave 分发。若要进一步做真正的
persistent worker + device-side work queue，需要改 runtime/scheduler ABI，风险和改动范围都
远高于本轮；当前数据也不支持为了它替换已经有效的 orchestration 模式。

## 9. A2A3 当前 profile

```text
Full QK blocks/task                    = 22
Full block-softmax blocks/task         = 12
Full SV+segment recurrence blocks/task = 16
Full online reduce fan-in               = 8
Full/SWA out-proj matmul N              = 64
Full/SWA out-proj tiles/task            = 3
Full/SWA vector N                       = 128
Full/SWA out-proj cast fusion           = 1
TP all-reduce transfer chunk            = 512
```

每个参数都应在新架构上重新 sweep。建议自动/离线校准的目标：

```text
minimize total critical-path stage span
subject to:
  logical-task counter limit
  legal cube/vector tile and UB/L1 budget
  active-batch/capacity correctness
  finite + TP consistency
  canonical precision gate
```

## 10. 验证与当前发布状态

源码/镜像验证已覆盖：source contracts、compile/lowered、bs=1/64K DFX、active-batch=16、
异构 context、immutable-image audit/smoke，以及 canonical N=128。

最终 candidate 镜像：

```text
hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260802-attn-final-canonical
manifest sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d
config   sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea
```

64K immutable ITL（bs=1、warmup=3、20 iterations）：

```text
min=49.213 ms, mean=50.568 ms, p50=50.563 ms, p99=max=52.537 ms
```

但 fresh oracle 的三次 N=128 都为：

```text
121/128 = 94.53125% < 95%
```

所有 hidden finite；run2 在 step39 出现 TP spread，run3 在 step68/70 出现 TP spread。
所以源码合入、镜像审计和性能验证已经收尾，但该镜像只能标记为 **candidate / release
blocked**，不能宣称 canonical release PASS。完整证据见
[`../../../benchmark/2026-08-02-step3p5-attention-final.md`](../../../benchmark/2026-08-02-step3p5-attention-final.md)。
