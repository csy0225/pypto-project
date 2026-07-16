# NEXT SESSION — N=1 当前状态与后续边界

> 2026-07-16 更新。旧的 4/5、残余 stall、P20、push/pull 组合提示和阶段性
> prompt 已删除，不得再作为当前状态使用。

## 1. 当前结论

N=1 standalone whole-net canonical gate 已完成：

```text
program = whole_decode_faithful_real
machine = gpu-a910x-0162
devices = 8..15
token = 6127
P_FAITHFUL_MOE_LAYERS = 42
weights = native W8A8 IPC
KV = IPC
dispatch = fixed-slot pull
combine = pull
golden = argmax 303
```

fresh exporter pool 连续 20 次：

```text
pass = 20/20
argmax = 303（每次）
TOP5 = [303, 9592, 768, 1043, 410]
runtime min/mean/max = 2.53 / 2.5685 / 2.62 s
```

日志：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/signal512/
  signal512_p42_20_20260716_220004
```

最终整理后 smoke：

```text
.../signal512_final_smoke_20260716_230225
2.57s, argmax=303, FINAL_SMOKE=PASS
```

20-run 与最终 smoke 的 dmesg 时间窗没有新增 fault、507018、
running-stalled 或 stranded CQE。

## 2. 已发布代码

```text
repo = csy0225/pypto-lib
branch = feat/whole-net-n1-fusion
commit = 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
message = fix(step3p5): stabilize N1 pull MoE with isolated control signals
```

提交只包含：

```text
models/step3p5/decode_layer.py
models/step3p5/moe.py
tools/step3p5/_gen_faithful_real.py
```

旧 `*.bak.*` 和临时 probe 已从 0162 工作树删除。

## 3. 最终架构边界

### 3.1 dispatch

```text
_dispatch_pack_publish
  本地 fixed-slot pack + 本地 pub_counts row

_dispatch_pull
  AtomicAdd/Ge rendezvous
  拉完整 counts_all
  生成 recv_counts、local expert CSR、inverse_map
  self local-load，peer remote-load

_dispatch_stage
  peer-major fixed-slot -> expert-major compact
```

### 3.2 routed expert

```text
local_routed_x INT8
local_routed_x_scale FP32
native INT8 gate/up/down matmul
signed tile remainder
no BF16 weight dequant fallback
```

### 3.3 combine

```text
_stage_routed_src
_pull_routed_y(dispatch-produced inverse_map)
_weighted_gather_and_add
```

self routed row 使用 `pl.load`；peer row 使用 `remote_load`。

### 3.4 control signal

逻辑 view 保持：

```text
[8,1] INT32 = 32B
```

物理 allocation：

```text
COMM_CONTROL_SIGNAL_BYTES = 512
216/216 signal nbytes = 512
all relative offsets % 512 = 0
window size = 766525440, %512 = 0
```

runtime 对 comm domain 整窗 zero-init。

## 4. generator 已收敛

generator 不再通过旧 A/B helper 把 inverse-map 重建移到 combine。
当前 generator 原生生成已验证边界，并通过真实字节 round-trip：

```text
PRECOMMIT_ROUNDTRIP=PASS
ROUNDTRIP_CMP_RC=0
```

后续任何生成器修改都必须重新做该检查。

## 5. 历史结论复核

1. 历史 `argmax=303` 证明数学路径可以正确，但不能证明历史版本无 stall。
2. 旧 push/pull kernel 位置是定位线索，不足以证明某个 TPUT 或 signal bit 是唯一硬件根因。
3. `routed_h_quant` 不是本轮失败的统一挂点；实际失败 build 曾表现为
   rank 8–14 在 `_pull_routed_y`、rank 15 在 `_dispatch_pull`。
4. fixed-slot、count-pull、signed tile、self local-load、AtomicAdd signal、
   per-layer distinct buffers 都有框架或边界依据，应保留。
5. 最终最小布局 A/B 变量是 control signal 物理分配 32B → 512B；
   其后 fresh pool 20/20 canonical PASS。

## 6. 下个 session 的真正工作

不要继续把 standalone canonical stall 当作 open blocker。后续工作转到 Phase 28：

1. live vLLM per-layer paged KV bridge；
2. 消除 vLLM 与 exporter 的冗余权重，解决 3-way HBM；
3. live single-handoff token-exact A/B；
4. 保持 standalone commit `0e7a0fdd` 为回归基线。

live blocker 与 standalone canonical 已通过是两个独立结论，不能混写成
“整个 serving 集成已经完成”。

另外，standalone release `0e7a0fdd` 只包含：

```text
decode_layer.py
moe.py
_gen_faithful_real.py
```

历史 live holder/sidecar/KV importer/容器 backend 不在该 commit 内。进入
Phase 28 前先按当前工作区实际状态整理、review 并单独提交 live 组件，不能
假设它们已由 `0e7a0fdd` 发布。

## 7. 唯一测试入口

测试命令、checkpoint、设备、20-run gate 和清理要求见：

```text
/data/chensiyu/hw_project/pypto/pypto-project/N1-CANONICAL-TEST.md
```
