# N=1 whole-net 0162 最终交接（2026-07-16）

> 这是本轮 stall 修复的最终事实记录。旧 prompt、旧冻结状态、4/5 残余
> stall 和中间层数结论已删除。

## 1. Release 结论

```text
whole_decode_faithful_real
native W8A8 weights via IPC
KV via IPC
token 6127
P_FAITHFUL_MOE_LAYERS=42
devices 8..15
dispatch pull + combine pull
argmax golden 303
```

fresh exporter pool 20 次全部通过：

```text
FINAL pass=20/20 rc=0
each run argmax=303
TOP5=[303, 9592, 768, 1043, 410]
```

日志：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/signal512/
  signal512_p42_20_20260717_001135
```

运行时间：

```text
min=2.50s
mean=2.5605s
max=2.62s
```

数值指纹 20 次完全一致：

```text
max|next_hidden|=264192.0000
row0|next_hidden|=588.0000
max|h_mid|=294.0000
max|logits|=14.0506
```

最终整理后 smoke：

```text
.../signal512_final_smoke_20260716_230225
2.57s, argmax=303, FINAL_SMOKE=PASS
```

## 2. Release commit

```text
repo: csy0225/pypto-lib
branch: feat/whole-net-n1-fusion
commit: 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
```

提交文件：

```text
models/step3p5/decode_layer.py
models/step3p5/moe.py
tools/step3p5/_gen_faithful_real.py
```

当次 `pypto-lib` 三个 tracked release 文件与 `0e7a0fdd` 一致；这不代表
当时完整三仓 runtime 已形成 clean manifest。历史 20-run 仍依赖 pypto /
simpler 的未提交支持。后续已将其 formalize 并推送为：

```text
pypto   e277de9f2a55a686956d66933301204520bd7374
simpler 36957c6b56700ecba3aeb8dbbedd6240594e01de
```

最终三仓 clean-pin smoke：

```text
.../release_manifest/final_stack_smoke_20260717_015635
rc=0, 2.58s, argmax=303, worker-window relevant dmesg=0
```

本地 live/debug backup/probe 未混入 standalone release。

## 3. 核心修复

### 3.1 物理 control-signal isolation

```python
COMM_CONTROL_SIGNAL_BYTES = 512
```

逻辑 tensor 仍是 `[8,1] INT32`，只扩大物理 allocation。

覆盖：

- dense prefix attention / MLP signal；
- 每个 MoE layer 的 `attn_sig_buf`；
- `count_done_buf`；
- `data_done_buf`；
- `sh_sig_buf`；
- `combine_done_buf`。

生成物审计：

```text
CommBufferSpec = 684
signal/done = 216
216/216 nbytes = 512
all relative offsets % 512 = 0
window size = 766525440, %512 = 0
```

runtime 在 `comm_alloc_domain_windows` 对整个新 comm window 做 zero-init。

### 3.2 dispatch 边界

```text
pack_publish -> dispatch_pull -> dispatch_stage
```

`_dispatch_pull` 在同一个 InCore 边界：

- 拉完整 `counts_all`；
- 生成 receiver-local `recv_counts`；
- 生成 local expert offset/count；
- 生成 source-local `inverse_map`；
- self payload 用 local load；
- peer payload 用 remote load。

### 3.3 combine 边界

```text
stage_routed_src -> pull_routed_y(inverse_map) -> weighted_gather
```

combine 不再重新读取分布式 count matrix 构造 inverse map。
self routed row 使用 `pl.load`，peer row 使用 `remote_load`。

### 3.4 native W8A8

- routed input 在 dispatch 前量化为 INT8 + FP32 per-token scale；
- routed gate/up/down 保持原生 INT8 matmul；
- 不允许回退 BF16 dequant 权重；
- signed `tile_rem` 避免空 tail 的 INDEX 下溢；
- padding、single-valid-row batch、window zero-init 维持既定约束。

## 4. generator 发布门槛

历史 generator 会把完整 count snapshot 和 inverse-map 计算移到 combine，
与 20/20 active 边界不一致。该问题已经修复。

真实检查：

```text
剥离 active real builder
运行 generator
cmp regenerated active
ROUNDTRIP_CMP_RC=0
PRECOMMIT_ROUNDTRIP=PASS
```

后续不得只运行“generator 拒绝覆盖现有 builder”的伪 round-trip。

## 5. dmesg 与 stall 边界

20 个逐 worker-run dmesg before/after 窗口没有：

```text
devmm/page fault
illegal VA / illegal instruction
DMA/UB fault
507018
running-stalled
stranded CQE
```

最终 smoke 的 worker-run 窗口同样无新增相关 dmesg。fresh exporter
pool teardown 后 outer 窗口新增 2 条 `stranded cqe`（dev8/dev11 exporter）；
它们不在任何 worker-run 窗口内，必须记录为 cleanup 边界，不能归因给整网 kernel。

历史失败 build 曾映射为：

```text
device 8–14: _pull_routed_y
device 15: _dispatch_pull
```

这说明旧的“统一挂在 routed_h_quant”判断不成立。但 kernel 位置只能用于定位，
不能单独证明 PUSH/TPUT 或某个 signal bit 是唯一根因。

## 6. 根因措辞

可以写：

> 在 0162 的固定测试环境中，512B signal isolation 是最终最小 layout
> A/B 变量，并与历史随机 stall 消失强关联；`0e7a0fdd` exact-model-source
> 完成 fresh pool 20/20。该结论不是 matched 单变量因果证明，也不覆盖 0234。

不应写：

> 已由硬件层证明某个具体 signal bit 丢失，或某个 TPUT 是唯一根因。

## 7. 下一阶段

0162 scope 的 standalone canonical stall gate 已关闭。0234 项目记录中的
stall 仍待恢复访问并按完整三仓/runtime manifest 复核。下一阶段是 live serving：

- per-layer KV bridge；
- 去除 vLLM/exporter 冗余权重；
- 解决 3-way HBM；
- live single-handoff token-exact A/B。

不得把这些 live blocker 误写成 standalone P42 仍未通过。
