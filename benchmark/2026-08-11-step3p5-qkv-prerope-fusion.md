# Step3p5 Attention QKV packed projection 与 pre-RoPE 融合（2026-08-11）

> **验证口径。** 本文记录的是固定 immutable substrate 上的
> **read-only `pypto-lib` source-overlay 验证**。候选源码挂载到
> `/candidate`；`pypto` runtime 没有 overlay。
>
> **镜像不包含本文的新 QKV/pre-RoPE 代码。** 镜像内 `pypto-lib` 仍为
> `cb96747e`。设备门使用的 frozen source 以 `f9065261` 为 parent/base；同一
> 字节内容随后提交为 `fa58b5cf`，并 push 到 `stepfun/develop`。本轮仍未构建
> 包含该提交的新镜像。
>
> **2026-08-12 post-merge 补充（当前准出结论，以此为准）：**
> 最终 clean commit 的整网精度 PASS，但 ITL p50 相对 baseline center
> **回退 `+1.348 ms / +4.233%`**；fresh 五层 DFX strict `<46 us` 为
> **39/40 FAIL**，rank7/L0 Full=`54.54 us`。因此源码实现与精度正确，
> 但性能集成当前 **NO-GO**。本文下方 2026-08-11 的 40/40、max `43.60 us`
> 保留为历史单次 capture，不再代表最终 release decision。
>
> **当前 tip 提示：** 后续 RMS→QKV 调度补丁已把远端和 0162 指定 checkout
> 前进到 `e5e26f9f`；它相对 `fa58b5cf` 无回退并解决局部约 5 us Worker
> 调度等待，但不覆盖本文 `f9065261 → fa58b5cf` 的 I6 NO-GO。见
> [`2026-08-12-step3p5-rms-qkv-dispatch-gap.md`](2026-08-12-step3p5-rms-qkv-dispatch-gap.md)。

## 1. 结论

本轮把 Attention 前端从独立 Q/K/V projection、QKNorm 和 RoPE publication
边界收敛为：

```text
packed QKV projection
  -> fused split + QKNorm + RoPE + Q/KV publication
  -> existing Full/SWA attention mixed task
```

最终源码与 post-merge gate：

| 项目 | 结果 |
|---|---|
| parent/base | `f906526190dc2eca0d479f8e9fa9187ec6d31be9` |
| final commit | `fa58b5cffe41b30d3f8d94482230867ee34b9e84` |
| I6 landing-time local/candidate/origin | 当时三者对齐 `fa58b5cf`，worktree clean |
| frozen source / run 后 source manifest | byte-identical |
| unit | `362 passed, 7 skipped`，rc=0 |
| whole compile | rc=0，`75.457 s` |
| focused edge / inactive-row / KV-slot / SWA direct oracle | PASS |
| focused Q-publication | Full/SWA × 6 contexts，`12/12` PASS |
| heterogeneous contexts | `[1,2816,2817]` exact |
| five-layer hidden L3/L4 | byte-exact、finite、TP spread=`0` |
| historical 2026-08-11 five-layer gate | `40/40`，max `43.60 us` |
| fresh 2026-08-12 five-layer gate | **`39/40` FAIL** |
| fresh global max | **`54.54 us @ rank7/L0`** |
| whole precision | **PASS**，hidden SHA exact、finite、token `14371` |
| whole ITL | **FAIL**，`31.846 → 33.194 ms`，`+4.233%` |
| custom QKV inventory/dependency/legacy audit | PASS |
| canonical five-layer structural analyzer | **FAIL_CLOSED，container rc=1** |

准确状态：

```text
implementation in frozen source       PASS
unit + whole compile                  PASS
focused correctness/publication       PASS
fresh five-layer strict span gate     FAIL (39/40)
whole precision                       PASS
whole ITL                             REGRESSION_BEYOND_BRACKET
canonical structural release gate     FAIL_CLOSED
source integration commit             fa58b5cffe41b30d3f8d94482230867ee34b9e84
I6 landing-time source alignment      CLEAN
remote stepfun/develop push            DONE
new immutable image                   NOT BUILT
production release qualification      NO-GO
```

历史 `43.60 us` 与 fresh `54.54 us` 都采用从最早 packed projection Worker View
slice 到 fused split/QKNorm/RoPE Worker View slice 结束的 **layer/rank stage
span**；它不是单个 physical kernel duration。fresh failure 与整网回退也再次证明，
不能用局部单次 DFX 外推整网 ITL 收益。

## 2. 固定 substrate 与 source/commit provenance

### 2.1 immutable substrate

全部设备验证固定使用：

```text
image:
hub.i.basemind.com/stepcast/vllm-pypto@
sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3

config/image ID:
sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
```

镜像内 pin：

| 组件 | commit/version |
|---|---|
| pypto-lib | `cb96747eb21f5f4932d6a24eddaa69c85d095ef6` |
| pypto | `1c048a744d5f63a8bce1ddb45dac8d1b7f458bb0` |
| simpler | `e2efebcbd190302609c0775d2984f409f5f42c76` |
| pto-isa | `ecb6c303f797749f811a494742c3c08156aacabb` |
| PTOAS | `fc8c6caee561914b4fb991dfc8427bb63194269e` |
| ptoas-bin | `v0.50` |
| vLLM overlay | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |
| attention profile | `a2a3` |

这是复用既有镜像 substrate 的验证，不是本文候选的新镜像验证。镜像 digest/config
只证明 substrate 被固定；候选 `pypto-lib` 由只读 `/candidate` source overlay
提供，runtime 未 overlay。

### 2.2 parent/base、frozen source 与 final commit

0162 candidate worktree：

```text
/mnt/persist/chensiyu/workspace/develop-worktrees/qkv-prerope-mix

branch:
perf/qkv-prerope-mix-20260811

parent/base:
f906526190dc2eca0d479f8e9fa9187ec6d31be9

final commit / HEAD:
fa58b5cffe41b30d3f8d94482230867ee34b9e84
```

I6 当时集成后：

```text
candidate HEAD = local stepfun/develop = origin/stepfun/develop = fa58b5cf
fa58b5cf^ = f9065261
ahead/behind = 0/0
main worktree status = clean
candidate worktree status = clean
```

当前远端和 0162 指定 checkout 已由后续 I7 前进到 `e5e26f9f`；上述区块只记录
I6 landing-time provenance。

设备门先冻结了 `f9065261` parent 上的 source 内容；设备运行结束后，这一内容未再
改动，直接落为单一 final commit `fa58b5cf`。因此历史 run contract 中的
`source_head=f9065261` 仍是正确的运行时 provenance，而下面的文件 SHA 把该 frozen
source 与最终提交逐字节关联起来。

关键源码 SHA256：

```text
6db7fa5f5a5d9268719bfddd35c1262406001acfe5654e295c09c74a9ad4454d
  models/step3p5/attention_full.py
7895ebf575e45aaf61673af36a6ee950b031bf8d2c556c46daa66ef0361bc62b
  models/step3p5/attention_swa.py
181d75fa16054270eabdb57eaeeba2d6fd15057907f63549b3f023694a29e7de
  tests/step3p5/harnesses/_stage_two_layer_attn.py
e60bcc224f4177b4bc787e08891d7ac2939b2606313aef2017191989bf3e7dac
  tests/step3p5/unit/test_attention_full_runtime_active_bound.py
4d183c6f50e914ea1457ab1edd7046a8c4e2b8c83135eb9391cc8a31cde3a1f6
  tests/step3p5/unit/test_attention_swa_active_bound.py
e6649a3bffb5f9e5dd0e9a680c614a7abed402520208d020f248f93d37eb2672
  tests/step3p5/unit/test_attention_swa_runtime_task_chain.py
76231e90d8acfb220b4dae482468e4ce0c5ffb376bf4182cdddaa41534dff6f3
  tests/step3p5/unit/test_performance_bc_contract.py
83be58a973c0fbae97d3b029373dbbcabc2fd2017e4dc30683b408b058f80c5e
  tests/step3p5/unit/test_two_layer_attention_harness.py
```

focused 与 five-layer 的 run 前/后 source manifest 文件均为：

```text
sha256 593f11babb47a01575c4611e5fd067e8932910cad504817cbbb9d9b6e2f115e8
```

二者逐字节相同，说明设备运行期间 frozen source 没有漂移。
`attention_full.py`、`attention_swa.py` 与 two-layer harness 的上述 SHA 也与
`fa58b5cf` 中对应文件完全一致。

## 3. 实现

### 3.1 Full：10-slice packed QKV projection

Full 每个 batch tile 由一个 `full_qkv_proj` task family 发布 packed
`[Q | K | V]` FP32 tensor：

```text
8 × Q(128 columns)
1 × K(128 columns)
1 × V(128 columns)
= 10 logical projection slices
```

Q/K/V 保留原 weight ABI、128-column output tile 和历史 K 维累加顺序；变化只发生在
orchestration 与 publication layout。原先三个 projection family 合并后，fused
consumer 只依赖一个 packed producer task。

Full 的 `full_qkv_split_qknorm_rope` 每个 active row 一个 logical vector task：

1. 从 packed tensor split Q/K/V；
2. 把 8 个 Q heads reshape 为 `[8,128]`，一次完成 aligned row reduction；
3. K 临时复制为 8 个等价 row，避免单个 `[1,1]` reduction 的 AIV 对齐问题；
4. Q/K 分别应用自己的 zero-centred gamma；
5. 对 Q/K 的 rotary lanes 做 partial RoPE；
6. 发布 padded Q，并写当前 token 的 K/V cache。

Full 只旋转前 64 lanes；非 rotary tail 先按历史 BF16 值发布，再覆盖 rotary 部分，
因此没有把 partial-RoPE 错写成 full-head RoPE。

### 3.2 SWA：14-slice packed QKV projection

SWA 的 `swa_qkv_proj`：

```text
12 × Q(128 columns)
 1 × K(128 columns)
 1 × V(128 columns)
= 14 logical projection slices
```

projection 被提前到独立 head-gate expand 之前。两者仍由数据依赖保证后续
`o_proj` 的正确顺序，但 projection 可以先占满 AIC，避免五层调度中部分 QKV slice
被 gate expand 错峰约 4 us。

### 3.3 SWA fused split/QKNorm/RoPE

SWA 的一个 active-row fused task 构造：

```text
[12 Q rows | 1 K row | 3 zero rows] -> [16,128]
```

随后：

1. 对 `[16,128]` 一次做 `row_sum` / `rsqrt`；
2. Q 的前 12 rows 与 K 的第 13 row 分别应用 Q/K gamma；
3. 对 `[16,64]` low/high halves 批量做 Q RoPE；
4. rows `12:16` 最终由 padding zero publication 覆盖；
5. K/V 只在最终 cache publication 时读取，不再保留独立 normalized GM scratch。

这保留了每个真实 Q/K row 的归约顺序和 distinct gamma，同时把 12 条逐 head Q RoPE
链收敛为两次 aligned vector operation。最终 SWA fused group 的 Vec 占用为
`33.6 KiB / 184.0 KiB`。

### 3.4 退役边界

候选 task graph 不再保留独立的 Q projection、KV projection、QKNorm、Q RoPE 和
KV-cache RoPE publication task family。最终五层 raw inventory 特别确认：

```text
full/swa_rope_q        0
full/swa_rope_kv_cache 0
```

`qkv_proj` 与 `qkv_split_qknorm_rope` 仍是两个 logical task family；本文没有把它们
误写成一个物理 binary。性能门覆盖的是二者组成的完整前端 span。

## 4. Unit 与 whole compile

### 4.1 Unit

```text
362 passed, 7 skipped in 5.60s
pytest rc=0
pytest.log sha256:
0cba5a5849b5adfda0d58a4ace6a397ac0bfeba787fc0be84df7cf9a64d0b14f
```

新增/更新 contract 覆盖：

- Full `10 + 1`、SWA `14 + 1` task chain；
- packed projection dynamic bound；
- fused active-row ownership 与 dependency；
- SWA `[16,128]` Q/K row pack、separate gamma 和 batched Q RoPE；
- retired task family 不得重新出现；
- two-layer harness 的 codegen/inventory contract。

### 4.2 Whole compile

```text
program       whole_decode_step3p5
num_blocks    512
elapsed       75.45700550079346 s
compile rc    0
```

compile report SHA256：

```text
69c9506db540b98d97121a4aeae7e743e1c81e2fc2bc09c96e3ee544b54b96ca
```

关键 memory report：

| group | memory |
|---|---|
| `full_qkv_proj` | Mat `80.0 KiB`，Left `8.0 KiB`，Right `64.0 KiB`，Acc `8.0 KiB` |
| `full_qkv_split_qknorm_rope` | Vec `13.5 KiB / 184.0 KiB` |
| `swa_qkv_proj` | Mat `72.0 KiB`，Left `8.0 KiB`，Right `64.0 KiB`，Acc `8.0 KiB` |
| `swa_qkv_split_qknorm_rope` | Vec `33.6 KiB / 184.0 KiB` |

## 5. Focused correctness 与 publication gate

focused gate 使用 cards `8--15`、fresh container、`num_blocks=512`、
batch capacity `16`；三个 arm 均 `rc=0`：

| arm | 覆盖 | 结果 |
|---|---|---|
| edge contexts | `1,127,128,129,511,512,513,2815,2816,2817,65535,65536` | `12/12` exact |
| alternating/inactive | active input discrimination + inactive-row isolation | PASS |
| KV-slot / SWA direct oracle | cache slot ownership + SWA direct oracle | PASS |
| Q-publication | Full/SWA × `127,129,513,2817,65535,65536` | `12/12` PASS |
| heterogeneous | per-row contexts `[1,2816,2817]` | exact，max_abs=`0` |

focused Worker View 中，SWA fused
`swa_qkv_split_qknorm_rope` 的 device slice duration 为 `2.82--3.44 us`。该值只说明
fused vector task 本身，不代替 §6 的完整 projection-to-fused `40` 点门。

## 6. Five-layer、8-rank strict `<46 us` 门

### 6.1 工作点与正确性

```text
layers        L0_full_dense
              L1_swa_dense
              L2_swa_dense
              L3_swa_moe
              L4_full_moe
devices       8,9,10,11,12,13,14,15
active_batch  1
context_len   65536 per active sequence
num_blocks    512
warmup/iters  3/3
```

输出：

| tensor | shape/dtype | exact | max_abs | finite | TP spread |
|---|---|---:|---:|---:|---:|
| hidden L3 | `[8,1,4096]` BF16 | true | `0` | true | `0` |
| hidden L4 | `[8,1,4096]` BF16 | true | `0` | true | `0` |

### 6.2 Worker View 计时协议

最终 authority 是 merged-swimlane **Worker View**：

```text
start  = earliest layer-local `*_qkv_proj` Worker View ts
finish = latest layer-local `*_qkv_split_qknorm_rope` Worker View (ts + dur)
span   = finish - start
gate   = max(8 ranks × 5 layers) < 46.000 us
```

rank2 只作 diagnostic；最终 verdict 使用全部 8 ranks 的 worst point。

### 6.3 分层范围

| layer | kind | min us | p50 us | mean us | max us | pass |
|---|---|---:|---:|---:|---:|---:|
| L0 | Full dense | 42.40 | 42.80 | 42.895 | **43.60** | 8/8 |
| L1 | SWA dense | 39.10 | 40.41 | 40.330 | 41.60 | 8/8 |
| L2 | SWA dense | 38.80 | 39.84 | 39.800 | 41.14 | 8/8 |
| L3 | SWA MoE | 39.30 | 40.46 | 40.390 | 41.16 | 8/8 |
| L4 | Full MoE | 38.96 | 40.19 | 40.228 | 41.62 | 8/8 |

### 6.4 40 点结果

| rank | L0 Full | L1 SWA | L2 SWA | L3 SWA-MoE | L4 Full-MoE | rank max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 43.14 | 39.52 | 41.14 | 40.64 | 39.14 | 43.14 |
| 1 | 42.42 | 40.68 | 39.14 | 40.98 | 40.50 | 42.42 |
| 2 | 42.70 | 40.14 | 39.90 | 41.06 | 39.88 | 42.70 |
| 3 | 42.60 | 39.10 | 39.62 | 39.94 | 40.62 | 42.60 |
| 4 | 42.90 | 41.02 | 39.78 | 41.16 | 41.52 | 42.90 |
| 5 | 42.40 | 41.60 | 38.80 | 39.76 | 38.96 | 42.40 |
| 6 | 43.40 | 39.88 | 39.96 | 39.30 | 41.62 | 43.40 |
| 7 | **43.60** | 40.70 | 40.06 | 40.28 | 39.58 | **43.60** |

```text
pass points = 40/40
fail points = 0
global max  = 43.60 us at rank7/L0_full_dense
margin      = 46.00 - 43.60 = 2.40 us
```

### 6.5 Inventory 与 dependency

每个 rank：

| layer | projection raw task | Worker slices | fused raw task | Worker slices |
|---|---:|---:|---:|---:|
| L0 Full | 1 × `block_num=10` | 10 | 1 × `block_num=1` | 1 |
| L1 SWA | 1 × `block_num=14` | 14 | 1 × `block_num=1` | 1 |
| L2 SWA | 1 × `block_num=14` | 14 | 1 × `block_num=1` | 1 |
| L3 SWA | 1 × `block_num=14` | 14 | 1 × `block_num=1` | 1 |
| L4 Full | 1 × `block_num=10` | 10 | 1 × `block_num=1` | 1 |

`projection -> fused` 和 `fused -> attn_mix` 均存在显式 dependency edge 与
tensor-map lineage edge；legacy `full/swa_rope_q`、
`full/swa_rope_kv_cache` callable/task family 均为 `0`。

### 6.6 被否证的 P1

最终 batched Q RoPE 之前的 P1 只有 `39/40`：

```text
L0/L1/L2/L3/L4 max =
43.58 / 46.64 / 44.84 / 45.14 / 41.06 us

global max = 46.64 us at rank4/L1_swa_dense
strict gate = FAIL
```

因此不能把 P1 写成“基本达标”。最终 `[16,64]` SWA batched Q RoPE 是把 worst point
收回到 `43.60 us`、获得 `2.40 us` margin 的必要收尾。

## 7. Canonical analyzer 限制

五层 runtime、precision、deps/swimlane capture 均已产生，但 harness 随后的 canonical
`analyze_five_layer_moe_dfx.py` fail-closed：

```text
rank0: 5 task IDs missing_on_swim
rank1: 5 task IDs missing_on_swim
rank3: 5 task IDs missing_on_swim
rank6: 5 task IDs missing_on_swim
```

这些是零本地 routed-token 的 early-dispatch task：deps 中有 task ID，但 raw AICore
swim record 中没有对应 physical slice。结果是：

```text
candidate container rc       1
canonical structural verdict FAIL_CLOSED
PTOAS compile failure        no
runtime precision failure    no
outer runner/postflight rc   0
```

保留下来的 deps/swimlane 随后由独立 QKV/pre-RoPE analyzer 检查；该 analyzer 对本轮
专属的 projection/fused inventory、dependency、lineage、legacy absence 和 40 点
strict span 返回 rc=0。这个 rc=0 **不能覆盖** canonical analyzer 的 rc=1，也不能
升级成整网 structural PASS、cross-rank release seal 或 production qualification。

## 8. 发布边界

2026-08-11 的 frozen-source 结果证明最终提交 `fa58b5cf` 在所列 unit、compile、
focused correctness 与当次 five-layer capture 满足：

- 源码已提交、push；在 I6 landing 时 0162 main/candidate 与 origin 对齐且 clean；
- exactness、finite、TP consistency；
- Q/KV publication 与 heterogeneous mapping；
- packed projection / fused consumer inventory；
- 当次 `8 × 5` Worker View strict span `<46 us`。

2026-08-12 post-merge fresh 验证进一步证明：

- 整网精度 byte-exact PASS；
- 整网 ITL 回退 `+4.233%`；
- fresh five-layer strict gate 只有 39/40；
- 历史 40/40 不足以形成稳定性能准出。

最终状态必须保持：

```text
0162 source-overlay implementation gate  PASS
source commit / remote push               fa58b5cf / DONE
I6 landing-time source alignment          ALIGNED, CLEAN
current stepfun/develop                    e5e26f9f (I7 follow-up)
whole precision                           PASS
whole performance                         FAIL
fresh five-layer timing                   FAIL (39/40)
canonical structural release gate         FAIL_CLOSED
new image                                  NOT BUILT
release-qualified                          NO-GO
```

## 9. 权威证据（0162）

### Unit

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
qkv-prerope-p0-unit-20260811-r1/

pytest.log
  sha256 0cba5a5849b5adfda0d58a4ace6a397ac0bfeba787fc0be84df7cf9a64d0b14f
```

### Whole compile

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
qkv-prerope-p0-compile-latest-20260811-r1/

compile_report.json
  sha256 69c9506db540b98d97121a4aeae7e743e1c81e2fc2bc09c96e3ee544b54b96ca
compile.log
  sha256 561698a3423637bd78442aae58026347d89bb0b9afeb4ad335dca9b01632f87d
```

### Focused + historical 2026-08-11 five-layer capture

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
qkv-prerope-final-device-gate-20260811-r1/
```

focused 关键报告：

```text
focused_candidate_edges/alternating_input_audit.json
  sha256 c981881e801ffdd2f0f992c7bb2a09954944233d7e7f5d8141ade811d0a25e1f
focused_candidate_edges/kv_slot_audit.json
  sha256 4db51962d24c7fc34aa12d2f47517e703795ec367682a3fd03dd1cfe3bd32bbb
focused_candidate_edges/swa_direct_oracle_audit.json
  sha256 0218f96f09599dadb1daa49febb6bbda6ef9d62f9650bd83ac92a24033287b38
focused_candidate_q_publication_clean/q_publication_audit.json
  sha256 7d83af11982632c4b6c5a693e7c6b78f258a9d6937d91228438149e9c934d82b
focused_candidate_heterogeneous/itl_report.json
  sha256 eb747ee9ad9c83f79503e12c6f86a25dbceeb912b5a5b7cff4d3b48051704aa9
```

five-layer 关键报告：

```text
five_layer/analysis_final/attention_gate_report.json
  sha256 e12e6bd2d39a43119ea3c747b4fabf78f5a0676aeacf2b7db5480d9b681b279e
five_layer/analysis_final/attention_gate_report.md
  sha256 96467afbc0dad7c2d5e212c3df965599ca4ef58752a269ec6e9da4fd7d4e0a99
five_layer/analysis_final/standalone_analyze_qkv_prerope_five_layer.py
  sha256 82313824af34e14d453a484b3337c2b69e617ae5aec53ad92ae239ba3eb674a1
five_layer/candidate_dfx/runtime/dfx_protocol_report.json
  sha256 9588b1b3bc818d67b3d42593969a2d50bcd219ee16e954a68539ea7f4b6c58b8
five_layer/candidate_dfx/runtime/five_layer_moe_report.json
  sha256 8de45fdef9d5e74c86929324a4d4e8c257b8d727c10219cfae20347937337037
```

P1 失败报告保留在：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
qkv-prerope-p1-device-gate-20260811-r1/
five_layer/analysis_p1/attention_gate_report.json

sha256 1c5034f6b18f5371c65ecc864aa4f82a2cde0cd85bfdbc32b976d5d597487014
```

上述 focused 与 40/40 capture 的验证日期为 **2026-08-11**；它们是历史证据。

## 10. 2026-08-12 post-merge 整网与 fresh DFX

### Whole-network A/B/A

```text
A1 baseline p50    31.787 ms
A2 baseline p50    31.905 ms
baseline center    31.846 ms
candidate p50      33.194 ms
delta              +1.348 ms / +4.233%
precision           PASS
performance         REGRESSION_BEYOND_BRACKET
```

三臂 hidden SHA256 均为
`567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e`，
finite、token `14371` exact。

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
attn-mix-device-gate-20260811/out/aba-bs1-ctx64k-20260812-102231/

ABA_RESULT.json
  sha256 065f67c889a5eb108c49770261ccadf4d8f2970882b657efafb205ee35d6510b
```

### Fresh five-layer DFX

8 份 merged swimlane 全部生成；L3/L4 exact、finite、TP spread=0。

```text
L0 Full      41.46–54.54 us  7/8
L1 SWA       38.90–43.14 us  8/8
L2 SWA       38.66–40.02 us  8/8
L3 SWA-MoE   39.16–41.72 us  8/8
L4 Full-MoE  39.38–41.50 us  8/8

total        39/40 FAIL
worst        rank7/L0 Full = 54.54 us
```

rank7/L0 的 QKV 与 fused kernel compute 时长正常；超门限来自约 `12 us`
AICPU scheduler dispatch stall。dependency/lineage 与 inventory 全部通过，但
严格端到端门必须包含该 launch skew。

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
qkv-prerope-postmerge-validation-20260811-r1/five_layer/analysis_final/

attention_gate_report.json
  sha256 0b5cbe2064663d179a509739e8c6ccd89777c839fcaca1023c4d1403c3a025a1
attention_gate_report.md
  sha256 f00149e36403e264018abd55fee4531672535a4b517dd43e5a052a78715c582e
```

当前结论：实现/精度 PASS，性能 NO-GO；不得构建 release image，先拆分定位
packed projection 与 fused epilogue。
