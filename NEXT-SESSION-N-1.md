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

release commit `0e7a0fdd` exact-source 在 fresh exporter pool 连续 20 次：

```text
pass = 20/20
argmax = 303（每次）
TOP5 = [303, 9592, 768, 1043, 410]
runtime min/mean/max = 2.50 / 2.5605 / 2.62 s
```

日志：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/signal512/
  signal512_p42_20_20260717_001135
```

最终整理后 smoke（release commit 的独立验证）：

```text
.../signal512_final_smoke_20260716_230225
2.57s, argmax=303, FINAL_SMOKE=PASS
```

20 个逐 worker-run dmesg 窗口与 smoke worker-run 窗口没有新增 fault、
507018、running-stalled 或 stranded CQE。20-run 结束后的 exporter
teardown outer 窗口新增 2 条 `stranded cqe`，已与 worker-run 窗口分离，
不得误写成模型 kernel stall。

## 2. 代码发布与复现边界

```text
repo = csy0225/pypto-lib
branch = feat/whole-net-n1-fusion
commit = 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
message = fix(step3p5): stabilize N1 pull MoE with isolated control signals
```

`pypto-lib` release commit 只包含：

```text
models/step3p5/decode_layer.py
models/step3p5/moe.py
tools/step3p5/_gen_faithful_real.py
```

旧 `*.bak.*` 和临时 probe 已从 0162 release 工作树移出；本地开发机仍有
live/debug WIP，未混入 standalone release。

这不是完整运行时 manifest。历史 exact-source 20-run 当时实际从
source/editable 路径加载：

```text
pypto HEAD 5e619dc7 + 未提交的 StackedDeviceTensor 分层 sub-view / import_ipc_all
simpler/runtime HEAD 98ce22a6 + 未提交的 child-process ACL IPC import
```

本次审计确认上述 pypto/simpler 支持在旧 20-run 时并未完整提交；现在已经
formalize、提交并推送为：

```text
pypto-lib  feat/whole-net-n1-fusion  0e7a0fddc90c4f2348f1d59e015fb817a0877a02
pypto      n1fusion-base             e277de9f2a55a686956d66933301204520bd7374
simpler    n1fusion-base             36957c6b56700ecba3aeb8dbbedd6240594e01de
```

0162 的这三个 release 工作树均 clean。最终 clean-pin canonical smoke：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/release_manifest/
  final_stack_smoke_20260717_015635

rc=0
RUN done 2.58s
argmax=303
TOP5=[303, 9592, 768, 1043, 410]
worker-window added relevant dmesg=0
```

outer 窗口仅在 exporter teardown 后新增 1 条 dev14 `stranded cqe`，不归入
worker kernel。**只拉 pypto-lib 仍不足以复现**；还要对齐三仓 pin、runtime
binary SHA、CANN/PTOAS/Python、checkpoint、设备和 ring 环境。0234 当前 SSH
返回 publickey/password permission denied，因此这些项目尚未现场独立核验。

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

> 证据审计补充：2026-07-16 候选 20-run 的 source SHA 与最终整理 smoke
> 不同。该缺口已通过 release commit `0e7a0fdd` 的 exact-source 20-run
> `signal512_p42_20_20260717_001135` 补齐。

1. 历史 `argmax=303` 证明数学路径可以正确，但不能证明历史版本无 stall。
2. 旧 push/pull kernel 位置是定位线索，不足以证明某个 TPUT 或 signal bit 是唯一硬件根因。
3. `routed_h_quant` 不是本轮失败的统一挂点；实际失败 build 曾表现为
   rank 8–14 在 `_pull_routed_y`、rank 15 在 `_dispatch_pull`。
4. fixed-slot、count-pull、signed tile、self local-load、AtomicAdd signal、
   per-layer distinct buffers 都有框架或边界依据，应保留。
5. 最终最小布局 A/B 变量是 control signal 物理分配 32B → 512B；
   在 0162 上其后 fresh pool 20/20 canonical PASS；这是强关联，不是跨机器
   充分条件、严格 matched 单变量因果证明或唯一硬件根因。

## 6. 下个 session 的真正工作

0162 scope 的 standalone gate 已关闭；但项目记录中 0234 在 pypto-lib 三个
release 文件与 `0e7a0fdd` byte-match 后 fresh canonical 3/3 stall，完整
runtime/build/environment 等价性未验证。该记录仍是 open blocker，不能忽略。
后续并行工作为：

1. 恢复 0234 访问，生成三仓/build/environment manifest 并复核 stall；
2. 保持 0162 clean stack `0e7a0fdd/e277de9f/36957c6b` 为回归基线；
3. live vLLM per-layer paged KV bridge；
4. 消除 vLLM 与 exporter 的冗余权重，解决 3-way HBM；
5. live single-handoff token-exact A/B。

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
