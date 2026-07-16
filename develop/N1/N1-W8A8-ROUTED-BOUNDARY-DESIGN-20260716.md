# N=1 原生 W8A8 routed boundary 最终设计（2026-07-16）

## 1. 固定约束

```text
program: whole_decode_faithful_real
machine: gpu-a910x-0162
devices: 8..15
P42: 42 MoE layers
token: 6127
golden: argmax 303
protocol: dispatch pull + combine pull
weights: native W8A8 IPC
KV: IPC
release: pypto-lib 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
```

## 2. Layer 交接

### 2.1 attention -> MoE input

```text
resid1/post_norm BF16
  -> per-token quant
  -> x_disp_i8 [BATCH,HIDDEN] INT8
  -> x_disp_scale [BATCH,8] FP32（有效 scale 在 col0）
```

router gate 与 shared expert 仍读取 BF16 `post_norm`；只有 routed expert
payload 走 INT8 dispatch。

### 2.2 dispatch

```text
_dispatch_pack_publish
  input:
    x INT8
    x_scale FP32
    expert_indices INT32
  output window:
    send_x
    send_scale
    send_route
    local pub_counts row

_dispatch_pull
  barrier:
    AtomicAdd + Ge
  local outputs:
    recv_counts [n_ranks,n_local_experts_pad] INT32
    local_expert_offset/count
    inverse_map [BATCH,TOPK] INT32
  payload:
    self local-load
    peer remote-load

_dispatch_stage
  peer-major fixed-slot -> expert-major compact
```

fixed-slot 每个 source/destination block 的容量为 `n_routes_per_rank`；
`recv_counts` 只负责有效范围和 expert-major compact，不改变 storage 边界。

### 2.3 routed expert

```text
local_routed_x [local_recv_max,HIDDEN] INT8
local_routed_x_scale [1,local_recv_max] FP32
gate/up/down weights INT8
weight scales FP32
local_routed_y [local_recv_max,HIDDEN] BF16
```

`tile_rem` 先以 INT32 计算，`tile_rem > 0` 后才 cast 为 INDEX，
避免 empty tail 下溢。

### 2.4 combine

```text
_stage_routed_src(local_routed_y -> routed_src_buf)
_pull_routed_y(inverse_map)
_weighted_gather_and_add(routed_y_buf, expert_weights, shared_y)
```

`inverse_map` 在 dispatch 生成并跨 orchestration 边界传给 combine。
combine 不重新拉 count matrix。self row 走 local load，peer row 走
remote load。

## 3. Buffer 与 lifetime

每个 MoE layer 使用独立 window：

```text
attn_tmp/signal
pub_counts
count_done
recv_x/recv_scale/recv_route
data_done
send_x/send_scale/send_route
shared tmp/signal
routed_y
routed_src
combine_done
```

跨 layer 不复用 active communication window，避免 RAW-only scheduler
把不同 layer 的控制状态或 payload 视为同一 lifetime。

## 4. Signal physical/logical boundary

逻辑：

```text
[8,1] INT32 = 32B
```

物理：

```text
COMM_CONTROL_SIGNAL_BYTES = 512
```

审计结果：

```text
216 signals
216/216 physical nbytes=512
all relative offsets %512=0
total window=766525440B, %512=0
```

该设计只隔离 control-plane allocation，不改变 notify/wait rank indexing。

## 5. 初始化、dtype、padding

- comm domain 整窗由 runtime `aclrtMemset(..., 0, window_size)` 初始化；
- routed output buffer 在 gather 前显式 zero；
- `n_local_experts_pad=40`，保证 count row burst/alignment；
- route row 使用 `idx_pad=8`，只消费 col0；
- single-batch 口径是 row0 单有效 token，row1..15 为 padding；
- 权重不可回退 BF16 dequant；KV 本身按 vLLM 定义为 BF16，不属于权重回退。

## 6. 生成器一致性

generator 必须直接表达上述边界，不允许用旧 post-string helper 把
inverse-map 计算搬到 combine。

release 验证：

```text
strip active real builder
regenerate
cmp
ROUNDTRIP_CMP_RC=0
```

## 7. Device 证据

20-run：

```text
.../signal512_p42_20_20260716_220004
20/20 argmax=303
runtime 2.53/2.5685/2.62s
```

最终 smoke：

```text
.../signal512_final_smoke_20260716_230225
2.57s, argmax=303
```

dmesg 时间窗无 fault、507018、running-stalled 或 stranded CQE。

## 8. 结论

当前 routed boundary 已同时满足：

- 原生 W8A8；
- fixed-slot dispatch pull；
- dispatch-produced inverse map；
- combine pull；
- self local-load；
- signed tail；
- per-layer distinct buffer；
- 512B control-signal physical isolation；
- canonical P42 20/20 精度与稳定性。
