# Step3p5 P1a gate 解耦收口与 `stepfun/develop` 发布（2026-08-10）

## 1. 发布结论

本轮 gate 解耦已发布到：

```text
csy0225/pypto-lib:stepfun/develop
  d13b2ca6e5ba4a507a2aeceb4c5f76cd7e348568
```

单 commit fast-forward，父提交为上一轮 N256 发布 `a31977fbb7ced6d2e599539c223d07813f161140`。
只改一个文件（`models/step3p5/decode_fwd.py`，+63/-35）：

```text
models/step3p5/decode_fwd.py
  sha256:28080c536a3731a9f40ad360b7d064f59bf70686de89e718cd99957d9984a07c
  （上一版 sha256:d392311ce1f38a67ddaa007173bb012c87e68cafeb5dca6b47813a2424683eea）
```

**该 sha256 与 A/B/A 的 candidate 臂逐字节相同**，因此下方全部设备数据直接绑定已发布代码，
不需要重测。

## 2. 落地优化

`gate_expert_fanout` 的 cube matmul 在数学上不需要 `norm_quant_moe_input` 的任何输出——
它吃的是 residual 与 gamma，只因 `inv_rms` 缩放被写在同一个 task 里而被串行拴住。

改法：

1. `gate_expert_fanout` 只写 raw FP32 logits，不再乘 `inv_rms`；
2. `inv_rms / sigmoid / bias` 尾巴搬进 `gate_topk`——它本来就要等 `inv_rms`，
   所以搬进去不引入新等待；
3. 算子顺序与数值语义不变。

codegen 实证：`params_t70` 不再 `add_input(moe_inv_rms)`，task 数与 `block_num=9` 均不变。

机理（见 §4）：MoE-only 段的依赖链从 15 hop 降到 14 hop，`norm_quant` 离开关键路径，
链头从 `81.8 us` 降到 `56.5 us`。

## 3. 整网 ITL（45 层，hidden-only holder.run）

三臂顺序 `parent -> candidate -> parent`，持 `0162-full-machine-perf.lock` 串行独占 16 卡。

### 3.1 bs=1 / ctx=65536 / blocks=512

```text
host                0162
active batch        1
context/max-seq     65536
blocks              512
warmup/measured     10/100
evidence            /mnt/persist/chensiyu/workspace/perf-2026q3/
                    p1a-gate-decouple-aba-20260810-110026
```

| 臂 | p50 (ms) | mean (ms) | min (ms) | p99 (ms) |
|---|---:|---:|---:|---:|
| A1 parent | 37.128 | 37.505 | 34.616 | 51.202 |
| **B candidate** | **33.849** | **34.302** | **33.270** | 47.296 |
| A2 parent | 35.859 | 36.199 | 35.332 | 43.281 |

| 指标 | parent center | candidate | 收益 |
|---|---:|---:|---:|
| p50 | 36.494 | 33.849 | **+2.645 ms（+7.25%）** |
| mean | 36.852 | 34.302 | +2.550 ms（+6.92%） |
| min | 34.974 | 33.270 | +1.704 ms（+4.87%） |

`parent_half_range = 0.634 ms` 即本工作点的检测地板；观测收益 `2.645 ms` 为其 4.2 倍。
裁决 **GO**。

### 3.2 bs=8 / ctx=65536 / blocks=4096

```text
active batch        8
context/max-seq     65536（per active sequence）
blocks              4096
warmup/measured     10/100
evidence            /mnt/persist/chensiyu/workspace/perf-2026q3/
                    p1a-bs8-ctx65536-nb4096-aba-20260810-113742
```

| 臂 | p50 (ms) | mean (ms) | min (ms) | p99 (ms) |
|---|---:|---:|---:|---:|
| A1 parent | 100.165 | 100.466 | 98.929 | 104.761 |
| **B candidate** | **91.722** | **92.144** | **90.327** | 97.820 |
| A2 parent | 94.891 | 95.396 | 93.651 | 103.328 |

| 指标 | parent center | candidate | 收益 |
|---|---:|---:|---:|
| p50 | 97.528 | 91.722 | **+5.806 ms（+5.95%）** |
| mean | 97.931 | 92.144 | +5.787 ms（+5.91%） |
| min | 96.290 | 90.327 | +5.963 ms（+6.19%） |

`parent_half_range = 2.637 ms`，观测收益为其 2.2 倍。裁决 **GO**。

### 3.3 bs=16

**物理不可行**，非性能问题：单次 `rtMalloc` 需 16 GiB，返回 `207001`。属容量上限。

### 3.4 口径提醒

- bs=1 用 `blocks=512`、bs=8 用 `blocks=4096`——编译期容量不同，**两档绝对值不可直接横比**；
  每一档内部三臂 blocks 一致，所以各自的 A/B/A 收益成立。
- `p99` 在 100 样本 harness 里下标为 99 即等于 `max`，只作诊断。

## 4. 精度

**byte-exact，不需要 vanilla oracle**：两档各自三臂的 hidden payload sha256 完全一致。

```text
bs=1  三臂 hidden sha256
  567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e
  （= 上一轮 N256 发布 golden；tail token 14371）

bs=8  三臂 hidden sha256
  1fcd4fcc9d0775a7c5fb08784725f9570246858291c85788fad6d4b234a8722e
```

按项目口径，hidden payload byte-exact 的改动用 sha256 即可准出；只有非 byte-exact 的
candidate 才需要多步 decode 逐 token 对 live vanilla（N=128 ALIGNED ≥ 95%）。

## 5. 前 5 层 swimlane（bs=1，已发布代码）

```text
evidence root
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  swimlane-p1a-candidate-20260810-130154

dfx_outputs（rank0..rank7 各一套）
  <root>/runtime/build_output/FiveLayerMoe_20260810_050452/
  dfx_outputs/rank<N>/d0/
    critical_path_report.md
    deps.json
    l2_swimlane_records.json
    merged_swimlane_20260810_0506*.json
    CPM_static.json
    CPM_observed.json
    name_map.json

汇总 stdout
  <root>/runtime/dfx_analysis/critical_path_stdout.txt

workload    active_batch=1, context_len=65536, num_blocks=512, seed_token=6127
layers      L0_full_dense / L1_swa_dense / L2_swa_dense / L3_swa_moe / L4_full_moe
devices     8-15
source      decode_fwd.py sha256 28080c53…（= 已发布代码）
health      hidden_l3 / hidden_l4 finite, tp_spread_max = 0.0
```

### 5.1 rank2（LOW-WAIT 分析 rank）

8 个 rank 的 makespan 从 `2.210 ms` 到 `555.892 ms` 极度倾斜——长 rank 的时间绝大部分是
kernel 内自旋吸收 rank skew，**不得当算术耗时**。按既有 LOW-WAIT 口径取 rank2：

```text
tasks 150 | happens-before edges 229
makespan            2.210 ms
static CPM path     1.806 ms (81.7%)  over 87 tasks
observed path       103 tasks
  compute           1.779 ms (80.5%)
  stall             0.431 ms (19.5%)   全部 data-wait
tiling check        compute+stall = makespan（exact）
```

关键路径上耗时前列的 kernel family：

| kernel family | compute ms | % makespan | # on path |
|---|---:|---:|---:|
| `tp_all_reduce` | 0.352 | 15.9% | 8 |
| `swa_chip_orch_dense_gate_up_matmul_tp` | 0.098 | 4.4% | 2 |
| `swa_out_proj_matmul` | 0.072 | 3.2% | 2 |
| `swa_q_proj` | 0.068 | 3.1% | 2 |
| `dense_gate_up_matmul_tp` | 0.051 | 2.3% | 1 |
| `swa_moe_chip_orch_expert_gate_up` | 0.046 | 2.1% | 1 |
| `expert_gate_up` | 0.044 | 2.0% | 1 |
| `swa_moe_chip_orch_expert_down` | 0.041 | 1.8% | 1 |

### 5.2 已知瑕疵（不影响上述结论）

该 run 进程 `rc=1`：`analyze_five_layer_moe_dfx.py` 最后的 task-level 结构契约在
**rank0 / rank1 / rank3 / rank6** 各报 5 个 `missing_on_swim` task id。
`rank2 / rank4 / rank5 / rank7` 契约干净，**分析用的 rank2 不在失败名单内**。
所有 8 个 rank 的 swimlane / deps / CPM / critical_path_report 均已完整落盘。

该 run 的 5 层 `p50 = 13.552 ms`（iters=3 / warmup=3）**含 DFX 插桩放大**，
不可当作 5 层的干净延迟，也不能乘 9 反推整网。

## 6. 同轮被否与新增硬约束

### 6.1 三个 NO-GO

| 候选 | 结论 | 依据 |
|---|---|---|
| `gate_up + act` 融合 | NO-GO（ROI，非能力） | 见 §6.2；且 `expert_gate_up_act` 在 rank2 关键路径只占 14.6 us = 0.65%，映射整网 0.13~0.17 ms，低于 bs=1 的 0.634 ms 检测地板 |
| `act + h_quant` 融合 | NO-GO | grid 维度不兼容 |
| `tp_all_reduce` 降 ring step | NO-GO（前提未证实） | AR 主要是吸收 rank skew 的 barrier；candidate swimlane 里单个 AR 观测到 35,530 us，128 KB payload 不可能是搬运耗时。AR 正确口径是 48 次 on-path、约 8.5~9.7%，**不是 15%** |

### 6.2 AIV Vec（UB）预算是 per-kernel-per-core，不是全局共享

实测证据（`33_after_AllocateMemoryAddr.py`，sha256 `e3f9d292…`，
脚本与 JSON 在 `0162:/mnt/persist/chensiyu/workspace/perf-2026q3/ub-scope-20260810/`）：

| 证据 | 内容 |
|---|---|
| 总和远超限额但编译通过 | 149 个 AIV 函数的 Vec 分配总和 `4676512 B = 24.8×` 的 `188416 B` 限额 |
| 每个函数 offset 从 0 重新开始 | 榜首全部 `min_offset = 0`；各函数地址空间独立 |
| spmd grid 内也是 per-core | `combine_reduce` `core_num=16`、每核 `40960 B`；若跨 grid 共享需 `655360 B` 会 FAIL，但它编过 |

推论：**kernel 不能把中间结果留在 UB 里给下一个 kernel 用**——这才是「融合」要同时装下两份
staging 的根因，也是 `gate_up+act` 撞 `401472 > 188416 B` 的机制（不是平台能力不足）。
树内本来就有能编过的融合路径 `full_moe_chip_orch_swiglu7_swiglu16_expert_gate_up_aiv`
（`4 × 8192 = 32768 B`，走 `K=64 / N=64 / RECV_SPECIAL_TILE=32`），
说明这是 tiling 取舍问题。

拟合与验证（预测→实测精确命中）：融合版 `= 1536 × KC`、baseline `= 512 × KC`；
预测 `KC=128 -> 196608 FAIL`、`KC=64 -> 98304 PASS`，实测一致。

### 6.3 K 归约 matmul 加 `pl.pipeline` 可行，但单独上测不出来

按 deepseek v4 `exp_gate_mm` 形式改写 `expert_gate_up` 的 K 循环
（`pl.create_tensor` 预建 accumulator + `pl.pipeline(0, HIDDEN, KC, stage=2)` +
`k0==0` 用 `pl.matmul`、其余 `pl.matmul_acc`），无卡 codegen 门实测：

| 变体 | 预测 `stage × 2 × KC × 256` | 实测 |
|---|---:|---|
| `KC=256, stage=2` | 262144 B | **FAIL，报 262144 B**（精确） |
| `KC=128, stage=2` | 131072 B | **COMPILE_OK** |

pass dump 证明 pipeline 结构真生效（不是被静默忽略）：`expert_gate_up_aiv` 的 Vec 分配
从 `2 × 65536` 变成 `4 × 32768`，`aiv_initialize_pipe` 的 `pipe: (32768, 4)`，
`27_after_LowerPipelineLoops.py` 出现 496 处 pipeline 结构。

代价：K trips `16 -> 32`、K tile 减半，cube 效率损失 vs ping-pong overlap 收益方向不确定。
且 stage=2 是**单 kernel 内**翻倍，全网 UB 大户按当前 tile 都装不下：

| kernel | 当前 Vec | ×2 | vs 188416 |
|---|---:|---:|---|
| `swa_qk_norm_zc` | 148352 | 296704 | ✗ |
| `full_rmsnorm_zc` | 135488 | 270976 | ✗ |
| `attn_residual_hold` | 131072 | 262144 | ✗ |
| `expert_gate_up_aiv` | 131072 | 262144 | ✗（KC 减半后 131072 ✓，已实测） |
| `tp_all_reduce` | 98304 | 196608 | ✗（差 8192） |
| `combine_reduce` | 40960 | 81920 | **✓ 直接装得下** |

所以 pipeline 化的正确打法不是逐个 kernel 试（每个都低于检测地板），而是把大 cube matmul
的 tile + pipeline 一次性改造成一个 bundle 上 A/B/A；`combine_reduce` 是唯一不用缩 tile
就能试的，可作第一个校准点。

## 7. 权威证据

```text
A/B/A bs=1     /mnt/persist/chensiyu/workspace/perf-2026q3/
               p1a-gate-decouple-aba-20260810-110026
A/B/A bs=8     /mnt/persist/chensiyu/workspace/perf-2026q3/
               p1a-bs8-ctx65536-nb4096-aba-20260810-113742
5 层 swimlane   /mnt/persist/chensiyu/workspace/perf-2026q3/
               swimlane-p1a-candidate-20260810-130154
UB 预算清单     /mnt/persist/chensiyu/workspace/perf-2026q3/ub-scope-20260810/
baseline dump  /mnt/persist/chensiyu/workspace/perf-2026q3/
               dump-baseline-20260810-144904
无卡 codegen 门 /mnt/persist/chensiyu/workspace/compile_gate.sh
```

镜像：`hub.i.basemind.com/stepcast/vllm-pypto@sha256:cab89668164cf85dc75e4f3ac53ef77ef4b8653767c7d147c5113cdee6a9d88c`
（`ATTN_TASK_PROFILE=a2a3` baked）。
