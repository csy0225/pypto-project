# 接力上下文（Handoff）

> **只描述下一位 agent 现在要接的工作。最后更新：2026-08-11。**
> 当前状态以 [`../STATUS.md`](../STATUS.md) 为准；历史过程不要复制回本文件。

## 1. 当前源码、镜像与发布边界

0162 权威 checkout：

```text
/mnt/persist/chensiyu/workspace/develop/pypto-lib
branch  stepfun/develop
HEAD    f906526190dc2eca0d479f8e9fa9187ec6d31be9
status  clean; aligned with origin/stepfun/develop
push    DONE
```

本轮普通 fast-forward push 后复核远端为：

```text
f906526190dc2eca0d479f8e9fa9187ec6d31be9
```

本地两提交：

```text
21d928b9e257f14aeb4b151cdcea720083f460d0
  perf(step3p5): fuse decode attention mixed kernels

f906526190dc2eca0d479f8e9fa9187ec6d31be9
  Update: parallelize step3p5 SWA RMSNorm
```

全部设备验证固定使用下列 immutable 镜像：

```text
manifest sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
```

镜像内 `pypto-lib` 仍为 `cb96747e`。候选代码通过只读 `/candidate`
**source overlay** 编译、运行，`pypto` runtime 没有 overlay。不得写成该镜像包含
`21d928b`/`f906526`，也不得写成新镜像已发布或 production release-qualified。

## 2. 本轮已完成

### 2.1 SWA RMSNorm 多核

- `SWA_RMSNORM_ROWS_PER_TASK=2`，当前 storage capacity `BATCH=16` 产生 8 个
  storage-capacity-row-derived logical tasks（非 active-token-derived）；
- 每 task 处理 2 行 `[4096]`，设备上每 rank 为 8 blocks / 8 distinct cores；
- 保持 16 个 256-element partial 的历史 left-fold 顺序；
- 配置 fail-closed，只接受已校准的 grain=2；
- strict block max `4.46 us`，logical-stage span max `4.90 us`，均 `<5 us`；
- L3/L4 byte-exact、finite、TP spread=0。

权威 metrics：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  swa-rmsnorm-scan-20260811/rows2-pack32-final-failclosed2/target_metrics.json
SHA256 b3d07bfc119529b037a77be7b57334a8903046e97bb4b58aadbc5c2830264180
```

### 2.2 Attention mixed kernels

- Full：QK→typed mask/softmax→SV→segment-local recurrence 合入
  `full_attn_mix`；跨 segment reduce/finalize 因并发写边界保留；
- SWA：每 active row 一个 `swa_attn_mix`；
- 旧 `full_qk_matmul/full_softmax/full_sv_matmul` 与
  `swa_qk_matmul/swa_softmax/swa_sv_matmul/swa_online_softmax` 不再出现。

Combined gate：

```text
unit             357 passed, 7 skipped
whole compile    PASS, num_blocks=512
focused seal     PASS
edge contexts    12/12 byte-exact
Q-publication    12/12 PASS
8-rank DFX       full_attn_mix=24, swa_attn_mix=1, split family=0
```

最终 BS1/ctx64K A/B/A：

```text
A1 baseline p50        32.222 ms
B candidate p50        31.790 ms
A2 baseline p50        32.330 ms
baseline center        32.276 ms
candidate delta        -0.486 ms / -1.506%
verdict                IMPROVEMENT_BEYOND_BRACKET
PRECISION_GATE         PASS
hidden SHA, all arms   567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e
tail token             14371
```

### 2.3 前五层 DFX swimlane

最终 `f9065261` 已完成 L0–L4、BS1、ctx64K、8-rank source-overlay capture：

```text
DFX_CAPTURE                  PASS
PRECISION_GATE               PASS
MIXED_ATTENTION_INVENTORY    PASS
SWA_RMSNORM_MULTICORE_LT_5US PASS
CANONICAL_STRUCTURAL_GATE    FAIL_CLOSED
DELIVERY_STATUS              LIMITED_NOT_RELEASE_QUALIFIED
candidate container rc       1 (postprocess analyzer fail-closed)
```

- L3/L4 对 baseline byte-exact、finite、TP spread=0；
- LOW-WAIT rank2 makespan `2.124 ms`；
- L3 RMSNorm 为 8 tasks / 8 distinct cores；全 rank最坏 slice/span
  `4.28/4.30 us`；
- L0/L4 Full 各 24 mixed blocks；L1/L2/L3 SWA 各 1 mixed block；
  forbidden split Attention family=0。

限制必须保留：rank0/1/3/6 各有 5 个零本地 routed-token 的 early-dispatch task
没有 AICore swim record，canonical structural analyzer 因而 fail-closed；
`candidate_dfx/container.rc=1` 来自该 canonical postprocess，而不是 runtime
precision failure。不能把该 delivery 写成 structural PASS、cross-rank release seal
或 production-qualified。

## 3. 权威证据

```text
combined unit
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  combined-attn-rmsnorm-unit-f906526/
  pytest.log SHA256 09d0e695b3448e65f7ce361bdee24a4c26bc0f683d4149bc212a013d502a1f18

combined whole compile
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  compile-combined-f906526-nb512-direct-20260812/
  compile.log SHA256 fad61e08f2640761b8182810f805903e9da583037459e202802b69cea13ec700

focused correctness / Q-publication / DFX
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  attn-mix-device-gate-20260811/out/focused-combined-20260812-001101/
  FOCUSED_CLEAN_SEAL.json SHA256
  ff8cd797a5eb4a7ff41731c48fe3f10fc58bb5cfb8f8743b218506f2609d721d

final A/B/A
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  attn-mix-device-gate-20260811/out/aba-bs1-ctx64k-20260812-001854/
  ABA_RESULT.json SHA256
  7eca25b23d3d944a841433a43f65cd5a4c829b9341984b5d58410388f08c4c80

five-layer DFX limited delivery
  /mnt/persist/chensiyu/workspace/perf-2026q3/
  five-layer-dfx-combined-f906526-20260811-final-v4/delivery/
  ALL_RANKS_swimlane_bundle.tar.gz SHA256
  d6f689c73b7ecb19b7febbf019a99baea4f96d59a778b37bfecacadbdc00def5
  LOW_WAIT_rank2_bundle.tar.gz SHA256
  e0bb2cc2beaa196b52547a04019d69720c0cb410b57b1c64524a476d09cd6d9a
  DFX_DELIVERY_SEAL.json SHA256
  088cf05ffbff717fd6da9fcf443122da88c4c9373c41276e4f5ae8dbfa51eb94
  DFX_DELIVERY_REPORT.json SHA256
  7bc5811da7cf543d3ddf812ee90e8297c3238ce0e7b24160899c19584fc29688
```

完整说明：
[`../benchmark/2026-08-11-step3p5-attention-mix-rmsnorm.md`](../benchmark/2026-08-11-step3p5-attention-mix-rmsnorm.md)。

## 4. 下一步

源码集成与远端 push 已完成。后续若继续做发布工作：

1. 以 `f9065261` 制作新的 immutable image，并在 **无 source/runtime overlay**
   口径复跑 audit、compile、focused correctness/DFX、RMS strict timing 和 A/B/A。
2. production release qualification 仍需完整 Main/MTP、N=128 多步 oracle 和所需
   batch/context matrix；本轮 source-overlay GO 不替代这些门。
3. RMS grain 只支持 2；若扩展到其他 storage capacity/grain，必须重新证明 UB、
   reduction order、核映射、精度与 strict `<5 us`。

## 5. 机器收尾

- 最终 DFX 后相关 campaign lock 均已释放；
- `npu-smi info` 显示 NPU 0–15 均 `No running processes found`；
- `nerdctl ps` 为空，未发现本轮遗留的 Attention/RMSNorm/DFX 测试进程。
