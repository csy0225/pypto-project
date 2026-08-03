# Step3p5 Wave5 TP all-reduce 稳定性修复与 immutable 发布（2026-08-03）

## 1. 结论

Wave4 间歇性 `hidden_tp_spread != 0` 的发布 blocker 已在 Wave5 闭环。当前
0162 release-qualified 版本为：

```text
pypto-lib stepfun/develop
  7099476b7c4f13112b159e237e7a64344803caf0

image
  hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260803-attn-final-wave5

manifest
  sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32

config
  sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394
```

机器范围：

```text
0162 = release-qualified
其它机器/架构 = 未由本轮独立证明
```

## 2. 最小修复

Wave4 已有三波协议：

```text
source partial
-> Wave 1
-> rank-owned reduce-scatter
-> push all-gather
-> Wave 2
-> final local copy
-> Wave 3 lifetime close
```

Wave5 不改变 rank ownership、固定 peer 顺序、单 FP32 accumulator、最终一次 BF16
cast，也不新增 orchestration kernel。唯一关键变化是把 source partial 从普通 local
store 改为 self-target synchronous TPUT：

```python
pld.tensor.put(
    dst=tmp_window,
    peer=my_rank,
    src=local,
    chunk_rows=BATCH,
    chunk_cols=TP_ALL_REDUCE_CHUNK,
)
```

完整顺序变为：

```text
local source
-> self-target drained TPUT
-> Wave 1 source publication
-> rank-owned reduce-scatter
-> existing remote-store push all-gather
-> Wave 2
-> final local copy
-> Wave 3 lifetime close
```

该修改同步覆盖 canonical Main、selected MTP 与 two-layer harness；MTP input
projection 也显式保留 all-reduce 返回值 lineage。源码/AST focused contracts：

```text
25 passed
```

准确的根因表述是：0162 当前证据支持 source publication/lifetime ordering 是关键
边界，self-target TPUT 是当前最小整网稳定性修复；本轮没有做跨所有硬件的唯一根因
定论。

## 3. Immutable 审计与功能 gate

镜像内 pin：

```text
pypto      defa97c526fec7e8f032dbbfcc39c820add02bf7
pypto-lib  7099476b7c4f13112b159e237e7a64344803caf0
pto-isa    ecb6c303f797749f811a494742c3c08156aacabb
PTOAS      fc8c6caee561914b4fb991dfc8427bb63194269e
simpler    e2efebcbd190302609c0775d2984f409f5f42c76
ptoas-bin  v0.50
vLLM       1b3e538c35999e62b6d24e0651b3a85b7d16c826
```

以下全部 PASS：

- source pin、五仓 clean worktree、credential、canonical-only；
- CANN runtime、优化符号、PTOAS ldd；
- smoke；
- Main/MTP compile + codegen；
- 无宿主源码挂载的 immutable device run。

Main N=128 预定义三轮完全一致：

```text
123/128 = 96.09375%
miss = [2,8,13,22,82]
finite = true
tp_spread_max = 0.0
```

Main active-batch=16：

```text
8/8 exact
finite = true
active rank rows = 128
tp_spread_max = 0.0
```

MTP batch1、batch16 两轮：

```text
tokens = [6178,410,303]
pass_rate = [1.0,1.0,1.0]
max_abs_diff = [0.0,0.0,0.0]
tp_spread = [0.0,0.0,0.0]
```

独立 all-reduce stress/A-B：

```text
128/128 epochs PASS
256/256 epochs PASS
TPUT source + remote-store gather: 256/256 PASS
TPUT source + TPUT gather:         256/256 PASS
```

早期未修正 probe 在 epoch 128 的失败属于修复前/探针演进证据，不能覆盖上述最终协议。

## 4. ITL 与 DFX

### batch1 / context=65536

```text
min  = 48.523 ms
mean = 50.027 ms
p50  = 49.796 ms
p99  = 54.539 ms
max  = 54.539 ms
```

### active-batch=16 / context=1

```text
min  = 112.525 ms
mean = 112.819 ms
p50  = 112.827 ms
p99  = 113.203 ms
max  = 113.203 ms
```

两个 case 均有 8 rank `deps.json`、8 rank swimlane、8 rank
`critical_path_report.md`，且 `all_cases_gate_pass=true`。

注意：`tp_all_reduce` 内部自旋等待仍被 critical-path 工具计入 kernel compute。
LOW-WAIT rank 只能作为 heuristic，分析时仍须交叉检查所有 rank 的长尾，不能把任一
rank 的长 span 直接解释为算术耗时。

## 5. 证据路径与安全状态

0162 主证据：

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_wave5_20260803_immutable/
    audit_container.log
    main_n128_predefined3/verified_summary.json
    batch16_mtp_campaign/verified_summary.json
    mtp_compile/result.txt
    itl_dfx/verified_summary.json
```

验证只使用 cards `0--7`。cards `8--15` 与 PID `2045390--2045397` 未操作；
结束后 cards `0--7` 无残留，保护 PID hash 保持：

```text
b703fd347215b7f66ef2afe5c0b5838749f63457cffc4a0b71019d3565694e0b
```

## 6. 发布判断

Wave5 满足预先定义的 immutable audit、Main N=128 三轮、Main batch16、MTP
batch1/batch16、64K/batch16 ITL/DFX 与设备安全 gate。因此：

```text
ATTN-WAVE4-STABILITY blocker = CLOSED
Wave5 canonical release
machine scope = 0162 release-qualified
```

Attention/Vec 设计边界保持不变：任务按 active workload 与 architecture profile
切分，不固定 24 核；`5--10 us/task` 只作 sweep 起点；不机械合入无稳定收益的
all-reduce/residual/RMS 融合。
