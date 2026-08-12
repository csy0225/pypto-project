# Step3p5 Attention Mix Kernel 与 SWA RMSNorm 多核集成（2026-08-11）

> **验证口径。** 本文记录的是：在最新 immutable 镜像上的 **source overlay
> 验证**。固定镜像只提供已发布 substrate，候选 `pypto-lib` 源码以只读
> `/candidate` overlay 编译、运行；`pypto` runtime 没有 overlay。
>
> **镜像不包含本文的新 Attention/RMSNorm 代码。** 新代码已在 0162 本地
> `stepfun/develop` 做 fast-forward 集成并 push 到远端，但尚未制作新的
> immutable candidate image。
>
> **当前 tip 提示（2026-08-12）：** 本文记录 I3/I4 landing-time 的
> `f9065261`；远端和 0162 指定 checkout 已经后续 I6/I7 前进到 `e5e26f9f`。
> RMS→QKV 调度 follow-up 见
> [`2026-08-12-step3p5-rms-qkv-dispatch-gap.md`](2026-08-12-step3p5-rms-qkv-dispatch-gap.md)。

## 1. 结论

本轮两项实现均已完成 source-level 集成和 0162 设备门：

| 项目 | 结果 |
|---|---|
| A/B baseline `cb96747e` 与固定镜像源码 pin 对齐 | PASS |
| I3/I4 landing-time GitHub 与 0162 `stepfun/develop` | PASS，当时均为 `f9065261`；本地 clean |
| combined unit | `357 passed, 7 skipped` |
| combined whole compile（`num_blocks=512`） | PASS |
| `swa_moe_chip_orch_swa_rmsnorm_zc` 多核 strict `<5 us` | PASS |
| combined focused byte-exact / Q-publication | PASS |
| combined mixed-kernel 8-rank DFX | PASS |
| 最终 commit 前五层 8-rank DFX | capture PASS；canonical structural gate FAIL_CLOSED，limited delivery |
| BS1、ctx64K 整网 source-overlay A/B/A | PASS，`-0.486 ms / -1.506%` |
| 本地 fast-forward 集成 | 已执行 |
| remote push | **已执行** |
| 新镜像发布 | **未执行** |

I3/I4 landing-time 准确状态：

```text
implementation             PASS
unit + whole compile       PASS
focused correctness/DFX    PASS
source-overlay A/B/A       IMPROVEMENT_BEYOND_BRACKET
precision                  PASS
remote push                DONE
image bake                 NOT DONE
```

## 2. 固定 substrate 与源码 pin

### 2.1 immutable 镜像

全部设备验证均固定使用以下 substrate digest，并在其上挂载只读
`/candidate` source overlay：

```text
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

镜像配置使用 `ATTN_TASK_PROFILE=a2a3`，并要求 prepared L2 swimlane reuse。
0162 的 `develop/{pypto,pto-isa,PTOAS}` 与上述镜像 pin 对齐。
`develop/pypto-lib` 的 A/B baseline `cb96747e` 与镜像一致；该轮 GitHub 与 0162
本地 `stepfun/develop` 当时均 fast-forward 到 `f9065261`，本地 worktree clean。

### 2.2 本地候选

```text
baseline:
  cb96747eb21f5f4932d6a24eddaa69c85d095ef6

standalone Attention:
  branch  perf/attn-mix-kernel-20260811
  commit  ffb08667af7b77dee2b0f42c3148cb390073f58c

standalone RMSNorm:
  branch  perf/swa-rmsnorm-multicore-20260811
  commit  0a0835e0f8bd3427fc4673f4f06514aadeeee84b

authoritative combined candidate:
  branch  perf/combined-attn-rmsnorm-20260811
  commit  f906526190dc2eca0d479f8e9fa9187ec6d31be9
```

combined commit 栈为：

```text
cb96747e
  -> 21d928b9e257f14aeb4b151cdcea720083f460d0  Attention mix
  -> f906526190dc2eca0d479f8e9fa9187ec6d31be9  RMSNorm multicore
```

`21d928b9` 与 `ffb08667`、`f9065261` 与 `0a0835e0` 的 stable patch-id
分别相同。combined worktree clean，且 base-ancestor contract 为 true。
三个 perf branch 均没有 upstream tracking；该轮 GitHub 与 0162 本地
`stepfun/develop` 当时均完成 fast-forward 集成到 `f9065261`。

focused 与最终 A/B/A 使用同一份 frozen candidate manifest；manifest 文件
SHA256 均为：

```text
1efd17510b26801a18b567e460bd8c6266664dce3f3ea7c886102585ba825f33
```

关键 combined 源文件 SHA256：

```text
models/step3p5/attention_full.py
  5b3121544c03900a155a2f047214470f374b7ade7fd85d2507203c35badc97bc
models/step3p5/attention_swa.py
  aa8d0188fda5798d99ecd7b1a53e0fae1488d6d36f7f60392b3d307cb983b3a5
models/step3p5/config.py
  0b9c74ed1de413ae71b4007425bd313a212656011f9892f7bd66e2c0634f91d9
models/step3p5/decode_fwd.py
  c2054854a4a0e3f5618f27694bc33b4fd5146be6e2372b8c98dfe95e3d3602fb
```

## 3. 实现一：SWA RMSNorm 多核

### 3.1 任务划分

`SWA_RMSNORM_ROWS_PER_TASK=2`，在当前 `BATCH=16` storage capacity 下产生：

```text
logical tasks = BATCH / rows_per_task = 16 / 2 = 8
```

每个 logical task 独占两个 `[4096]` 行及其输出，不存在跨 task concurrent
writer。代码不绑定物理 core id；runtime 根据可用 AIV 资源把这 8 个
storage-capacity-row-derived logical tasks 映射到物理核。

只允许已校准的 `rows_per_task=2`。`1/4/8/16` 及其他未验证值 fail-closed，
避免把无法满足 UB/设备门的配置暴露成支持项。

### 3.2 数值顺序与对齐

每个 task：

1. 一次读取并转换 `[2,4096]`；
2. reshape 为 `[32,256]`，得到每个源行的 16 个 FP32 partial；
3. 将两个源行 pack 到 `[16,8]` 的前两 lane，其余 lane 为零；
4. 用 15 次 tile `tadd` 按 chunk `0..15` 顺序 left-fold；
5. 分别完成两个源行的 RMS scale 和 BF16 输出。

因此保留历史实现的 256-element chunk 归约顺序，同时避开 A2A3 不支持的 scalar
FP32 `arith.addf`。最终 whole PTO 中：

```text
swa_moe_chip_orch_swa_rmsnorm_zc
  Vec UB       96.2 KiB / 184.0 KiB = 52.3%
  memrefs      23
  arith.addf   0
  PTO sha256   312ec9208383863c385473cd0ae028497a2f5d639879ea4065f166746deaa9fe
```

RMS-only whole compile 与 combined `f9065261` whole compile 的上述 PTO
逐字节相同。

## 4. 实现二：Attention mixed InCore task

这里的“一个 mix kernel”准确含义是一个 mixed InCore task/function group；
backend 会把它 lower 成配对的 AIC 与 AIV physical callable，并非一个单一物理
binary。

### 4.1 Full Attention

`full_attn_mix` 的每个 logical task 独占一段连续 KV blocks，在同一 mixed task 内完成：

```text
QK matmul (AIC)
  -> typed mask + softmax (AIV)
  -> SV matmul (AIC)
  -> segment-local online recurrence (AIV)
```

`a2a3` profile 为每 task 22 个 KV blocks。ctx64K 的 512 blocks 因而得到
24 个 mixed tasks；跨 segment 的 O/M/L partial 仍由 3 个
`full_online_softmax_reduce` task 和 1 个 finalize task 收口。

### 4.2 SWA

`swa_attn_mix` 每 active row 一个 logical task，在同一 task 内遍历可见 SWA window，
完成 QK、双边 window mask、softmax、SV、online recurrence 和最终
`attn_out` 写入。

原先用于跨 kernel 搬运 raw scores、exp、M/L、O partial 的分裂前端被移除。设备图中
不再出现：

```text
full_qk_matmul / full_softmax / full_sv_matmul
swa_qk_matmul / swa_softmax / swa_sv_matmul / swa_online_softmax
```

## 5. Unit 与 compile

### 5.1 combined unit

```text
357 passed, 7 skipped in 5.91s
pytest rc=0
pytest.log sha256:
09d0e695b3448e65f7ce361bdee24a4c26bc0f683d4149bc212a013d502a1f18
```

RMSNorm contract 覆盖默认 8 logical tasks、只接受 grain=2、拒绝未校准 grain、
storage-capacity-row-derived SPMD extent、aligned lane pack、一次 full-row load
以及 16-part left-fold。

### 5.2 whole compile

| 源码 | 结果 |
|---|---|
| standalone Attention `ffb08667` | `COMPILE_OK 12.8s` |
| standalone RMSNorm final | `COMPILE_OK 13.7s` |
| combined `f9065261`, `num_blocks=512` | `COMPILE_OK 83.258s`, rc=0 |

combined compile log SHA256：

```text
fad61e08f2640761b8182810f805903e9da583037459e202802b69cea13ec700
```

compile artifact 同时确认 `full_attn_mix.pto`、`swa_attn_mix.pto` 存在，旧 split
PTO family 不存在。UB report 中：

| mixed group | AIC memory | AIV Vec |
|---|---|---:|
| `full_attn_mix` | Mat 68 KiB / Left 4 KiB / Right 32 KiB / Acc 8 KiB | 52.4 KiB |
| `swa_attn_mix` | Mat 104 KiB / Left 8 KiB / Right 32 KiB / Acc 16 KiB | 120.8 KiB |

## 6. RMSNorm strict `<5 us` 设备门

工作点为 FiveLayer、64K、active batch 1、warmup 3、measured 20；8 ranks。
每个 rank 均观察到 8 logical blocks 映射到 8 个 distinct physical AIV cores。

| 指标 | count | min | mean | p50 | max |
|---|---:|---:|---:|---:|---:|
| block duration | 64 | 3.000 us | 3.76375 us | 3.800 us | **4.460 us** |
| logical stage span | 8 | 4.180 us | 4.555 us | 4.610 us | **4.900 us** |

```text
strict_block_max_under_5us        true
strict_logical_span_max_under_5us true
```

精度：

| 输出 | dtype/shape | max_abs | exact | finite | TP spread |
|---|---|---:|---|---|---:|
| hidden L3 | BF16 `[8,1,4096]` | 0 | true | true | 0 |
| hidden L4 | BF16 `[8,1,4096]` | 0 | true | true | 0 |

权威 metrics SHA256：

```text
b3d07bfc119529b037a77be7b57334a8903046e97bb4b58aadbc5c2830264180
```

## 7. Combined focused correctness、Q-publication 与 8-rank DFX

combined focused seal：

```text
COMBINED_FOCUSED_SEAL          PASS
EXACT_REPLACEMENT_GATE         PASS
HETEROGENEOUS_MAPPING_GATE     PASS
Q_PUBLICATION_CLEAN_GATE       PASS
MIXED_KERNEL_DFX_8RANK_GATE    PASS

FOCUSED_CLEAN_SEAL.json sha256:
ff8cd797a5eb4a7ff41731c48fe3f10fc58bb5cfb8f8743b218506f2609d721d
```

### 7.1 byte-exact replacement

BS1 edge matrix：

```text
ctx = 1,
      127,128,129,
      511,512,513,
      2815,2816,2817,
      65535,65536
```

12/12 cases 均为 candidate 对 baseline `max_abs_diff=0`、byte-exact、
iteration-stable、finite、TP spread 0。异构 BS3
`per_row_context_lens=[1,2816,2817]` 同样 exact，且 heterogeneous task mapping
合同通过。alternating input、inactive row、KV-slot 与独立 SWA direct oracle
审计均通过。

### 7.2 Q-publication

固定 hidden 下，分别对 Full/SWA 的 q-norm gamma 做 canonical 与 `-1` 区分，
并执行 ABAB：

```text
attention kinds = full, swa
ctx             = 127,129,513,2817,65535,65536
cases           = 2 × 6 = 12
```

12/12 均 `A != B`、ABAB stable、所有目标 tile 均有区分度，证明 mixed
AIC/AIV 路径真实消费并发布 Q。这里 baseline replacement comparison
**不适用**；旧汇总中的 `all_replacement_exact=false` 是预期的 legacy 字段，
不是失败。

### 7.3 8-rank DFX

所有 8 rank 均观察到 mixed AIC+AIV lane：

| stage | block_num | execution records/rank |
|---|---:|---:|
| `full_attn_mix` | 24 | 72（3 iterations） |
| `swa_attn_mix` | 1 | 3（3 iterations） |
| `full_online_softmax_reduce` | 3 | 3 |
| `full_online_softmax_finalize` | 1 | 1 |

旧 split family 在 8 rank 均为空。`MIX_DFX_GATE.json`：

```text
sha256 be3a9b54f34c56b14539928aac09e80af6a29d56088eb29208fd8b2d3d18071e
DFX_GATE=PASS
```

### 7.4 最终 commit 前五层 DFX swimlane

在最终 `f9065261` 上另跑 L0–L4、BS1、ctx64K、8-rank source-overlay DFX：

```text
DFX_CAPTURE                  PASS
PRECISION_GATE               PASS
MIXED_ATTENTION_INVENTORY    PASS
SWA_RMSNORM_MULTICORE_LT_5US PASS
CANONICAL_STRUCTURAL_GATE    FAIL_CLOSED
DELIVERY_STATUS              LIMITED_NOT_RELEASE_QUALIFIED
candidate container rc       1 (postprocess analyzer fail-closed)
```

L3/L4 对 `cb96747e` baseline byte-exact、max_abs=0、finite、TP spread=0。
LOW-WAIT 参考为 rank2：

```text
makespan             2.124 ms
static CPM           1.742 ms (82.0%)
compute              1.709 ms (80.5%)
runtime stall        0.415 ms (19.5%)
tiling check         exact
```

五层 task inventory（每 rank）：

| stage | logical blocks | physical slices |
|---|---:|---:|
| L0 Full mix | 24 | 24 AIC + 48 AIV |
| L1+L2 SWA mix | 2 | 2 AIC + 4 AIV |
| L3 SWA mix | 1 | 1 AIC + 2 AIV |
| L4 Full mix | 24 | 24 AIC + 48 AIV |
| Full reduce/finalize | 6 / 2 | — |
| L3 SWA RMSNorm | 8 | 8 AIV / 8 distinct cores |

forbidden split Attention family=0。L3 RMSNorm 的全 rank最坏 slice 为
`4.28 us`、stage span 为 `4.30 us`。

限制必须显式保留：rank0/1/3/6 各有 5 个零本地 routed-token 的
early-dispatch task 没有 AICore swim record；canonical analyzer 按全 dep-task
覆盖合同返回 `FAIL_CLOSED`。所有已出现 task 的 physical-slice count 正确，但本轮
`candidate_dfx/container.rc=1` 来自该 canonical postprocess，而不是 runtime
precision failure；因此**不声明 structural PASS、cross-rank release seal 或
production qualification**。

可交付文件：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
five-layer-dfx-combined-f906526-20260811-final-v4/delivery/

ALL_RANKS_swimlane_bundle.tar.gz
  sha256 d6f689c73b7ecb19b7febbf019a99baea4f96d59a778b37bfecacadbdc00def5
LOW_WAIT_rank2_bundle.tar.gz
  sha256 e0bb2cc2beaa196b52547a04019d69720c0cb410b57b1c64524a476d09cd6d9a
DFX_DELIVERY_REPORT.md
  sha256 1e145446c8474fbb0da5cccd7e663e13229b504054c665814402677bc1cad5c0
DFX_DELIVERY_REPORT.json
  sha256 7bc5811da7cf543d3ddf812ee90e8297c3238ce0e7b24160899c19584fc29688
DFX_DELIVERY_SEAL.json
  sha256 088cf05ffbff717fd6da9fcf443122da88c4c9373c41276e4f5ae8dbfa51eb94
```

## 8. 整网 BS1、ctx64K source-overlay A/B/A

```text
active batch       1
context length     65536
num_blocks         512
storage capacity   16
warmup/measured    10/100
devices            8-15（另一半保持 idle）
source order       baseline -> candidate -> baseline
runtime overlay    false
```

| 臂 | min | mean | p50 | p99/max |
|---|---:|---:|---:|---:|
| A1 baseline `cb96747e` | 31.837 ms | 32.399 ms | 32.222 ms | 35.421 ms |
| **B candidate `f9065261`** | **31.482 ms** | **31.854 ms** | **31.790 ms** | **34.111 ms** |
| A2 baseline `cb96747e` | 31.891 ms | 32.421 ms | 32.330 ms | 35.574 ms |

```text
baseline p50 center       32.276 ms
baseline half-range floor  0.054 ms
candidate delta            -0.486 ms / -1.506%
delta / floor               9.0x
performance verdict        IMPROVEMENT_BEYOND_BRACKET
```

三臂 hidden 均 finite、逐字节相同，并命中既有 expected SHA：

```text
hidden sha256:
567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e

tail token:
14371（token_exact=true）

PRECISION_GATE=PASS
A_B_A_COMPLETED=true
```

最终 `ABA_RESULT.json` SHA256：

```text
7eca25b23d3d944a841433a43f65cd5a4c829b9341984b5d58410388f08c4c80
```

## 9. 局限与发布边界

1. **不是新镜像验证。** 所有结果均是 pinned immutable K8 镜像上的只读 source
   overlay；镜像内仍是 `pypto-lib@cb96747e`。
2. **源码已推送，但尚无新镜像。** I3/I4 landing commit 为 `f9065261`；
   当前 `stepfun/develop` 已前进到 `e5e26f9f`，固定镜像内仍是 `cb96747e`。
3. RMS strict timing 来自 FiveLayer harness；combined whole graph 未单独重跑
   RMS strict profile。RMS-only 与 combined whole compile 的目标 PTO 均为
   `312ec920...` 且逐字节相同。
4. RMS grain 目前只校准 `rows_per_task=2`。它产生 logical tasks 后交给 runtime
   调度，不是 app-side persistent worker/work-stealing queue，也未验证其他 storage
   capacity/grain。
5. Attention 的“一个 mix kernel”是一个 mixed task group，lowering 仍有 AIC/AIV
   physical callable；Full 的跨 segment reduce/finalize、RoPE 与 out-proj 仍是独立
   stage。
6. 前五层 DFX 是 limited delivery；canonical structural gate 因部分零本地路由
   early-dispatch task 无 AICore record 而 fail-closed，不能作 release seal。
7. 整网性能只完成一轮 BS1/ctx64K A/B/A；尚未覆盖更大 active batch。整网精度门是
   固定 step hidden byte-exact，不等同于 N=128 多步 vanilla token oracle。
8. 一个旧 `_probe_g1_active_batch --compile` wrapper 因 AST 仍期待已删除的
   `_quant_moe_input` 而在进入 backend compile 前失败；本文 compile 结论来自
   canonical direct whole compile。该 probe contract 后续需单独更新。

因此，本轮结论是：

```text
local + remote `stepfun/develop` integration: DONE
immutable image release: PENDING
```

## 10. 权威证据（0162）

```text
combined unit
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  combined-attn-rmsnorm-unit-f906526/

combined whole compile
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  compile-combined-f906526-nb512-direct-20260812/

RMS strict metrics
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  swa-rmsnorm-scan-20260811/rows2-pack32-final-failclosed2/
    target_metrics.json

combined focused seal / Q-publication / 8-rank DFX
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  attn-mix-device-gate-20260811/out/focused-combined-20260812-001101/

final source-overlay A/B/A
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  attn-mix-device-gate-20260811/out/aba-bs1-ctx64k-20260812-001854/
    ABA_RESULT.json

final L0-L4 DFX limited delivery
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  five-layer-dfx-combined-f906526-20260811-final-v4/delivery/

frozen source provenance
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  attn-mix-device-gate-20260811/frozen/combined-f906526-20260812-0018/
```
