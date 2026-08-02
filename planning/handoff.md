# 接力上下文（Handoff）

> **这是 ephemeral 接力文档**——给“接着干”的 agent 一页纸当前工作面。
> durable 规划看 [`roadmap.md`](roadmap.md)，实时状态看
> [`../STATUS.md`](../STATUS.md)，发布 blocker 看 [`../blockers.md`](../blockers.md)，
> 本轮完整证据看
> [`../benchmark/2026-08-02-step3p5-attention-final.md`](../benchmark/2026-08-02-step3p5-attention-final.md)。
> **最后更新：2026-08-02。** 更新时直接改写本文，不追加流水。

## 1. 当前 source of truth

```text
machine:    gpu-a910x-0162
devices:    0..7（本轮验证）；8..15 及 PID 2045390--2045397 未操作
checkpoint: /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp

pypto-lib / vllm-pypto:
  76d96bdbeac280f12ecf626b1bbd722b9278719e
  branch stepfun/develop
pypto:
  defa97c526fec7e8f032dbbfcc39c820add02bf7
  branch stepfun/develop
simpler:
  e2efebcbd190302609c0775d2984f409f5f42c76
pto-isa: ecb6c303f797749f811a494742c3c08156aacabb
PTOAS:   fc8c6caee561914b4fb991dfc8427bb63194269e
ptoas:   v0.50
vLLM:    1b3e538c35999e62b6d24e0651b3a85b7d16c826
```

当前唯一默认 Main：

```text
models.step3p5.decode_fwd:whole_decode_step3p5
```

权威 clean pypto-lib worktree：

```text
/data/chensiyu/hw_project/pypto/workspace/vllm-pypto
```

不要使用 dirty 的 `workspace/pypto-lib` 或 `workspace/pypto`，也不要把 0162 裸机
checkout/overlay 当作 source of truth。产品源码以以上两个远端 commit 和镜像内 clean
checkout 为准；在 precision 根因闭环前不要修改或重写这两个已推送 commit。

## 2. clean canonical candidate

```text
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260802-attn-final-canonical

manifest:
  sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d

config/image ID:
  sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea
```

这是 **candidate，不是正式 release**。本轮验证为 immutable image：只挂载
host driver（只读）、checkpoint（只读）和 output（读写），没有宿主源码挂载。

```text
source integration        PASS
image audit/smoke         PASS
immutable 64K ITL/DFX     PASS
N=128 raw precision gate  FAIL
formal release            BLOCKED
```

已通过：image config/worktree/credential/canonical-only/CANN runtime/optimization
symbol/PTOAS ldd audit、smoke、64K ITL、DFX。测试结束后 cards 0--7 无残留进程。

## 3. 当前 attention / Vec 设计决策

- 不固定 24 个物理核心；logical task 按 active rows、每行真实 `seq_len` 和
  architecture-profile grain 推导，再由 runtime 映射到 AIC/AIV wave；
- `5--10 us/task` 只是 sweep 起点，最终联合比较 duration、stage span、wave/core-wait、
  packing、tail、dispatch、reduction/finalize dependency tail 与 batch16；
- A2A3 当前 profile：Full QK `22 blocks/task`、softmax `12`、SV + segment recurrence
  `16`、reduce fan-in `8`、Full/SWA out-proj matmul N `64`、`3 tiles/task`、vector N
  `128`、cast fusion 默认开启、TP all-reduce chunk `512`；
- Full Pass-A 已并入 `full_sv_matmul`；只保留有跨 task RAW/liveness 必要性的
  `full_online_softmax_reduce/finalize`；
- SWA 保持每个 active row 一个高密度 task，不复制 Full 的层次归约；
- Full/SWA decode out-proj 默认不生成独立 cast kernel；源码保留 `FUSE_CAST=0`
  fallback，prefill 的独立 cast 仍存在；
- 保留 dense RMS direct BF16 reread、dense down-proj cast fusion；
- AR+residual、residual+RMS stats、RMS+projection、gate/up+SiLU 等 probe 无稳定收益，
  不合 canonical；
- `BATCH=16` 是 storage capacity，不是固定 logical batch；
- 保持 PyPTO Orchestration/InCore/runtime 分层，不新增 app-side persistent
  work-stealing queue。

## 4. 性能与 DFX 证据

64K 条件：bs=1、context=65536、512 blocks、warmup=3、20 measured iterations。

```text
min  = 49.213 ms
mean = 50.568 ms
p50  = 50.563 ms
p99  = 52.537 ms
max  = 52.537 ms
```

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/itl64k/itl_report.json
```

LOW-WAIT DFX reference 必须使用 rank2：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/itl64k/build_output/
  WholeDecodeStep3p5_20260802_162729/dfx_outputs/rank2/d0/
    critical_path_report.md
    merged_swimlane_20260802_162823.json
```

```text
rank2 makespan                         = 38.924 ms
rank2 TP all-reduce critical-path compute = 2.049 ms
```

rank5 的 `344.553 ms` TP all-reduce compute 主要吸收 collective 自旋等待，只能用于
观察完整 wait span，不能称为 LOW-WAIT reference。

## 5. 当前唯一直接发布 blocker

同一 fresh oracle 三轮均：

```text
121/128 = 94.53125% < 95%
```

```text
run1 miss [2,8,13,15,22,82,93]；TP spread=none
run2 miss [2,8,15,22,29,82,93]；step39=0.953125
run3 miss [2,8,13,14,22,82,93]；step68=1.1875，step70=3.25
```

所有 hidden finite。报告：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/
    n128_teacher_forced/main_hidden_only_report.json
    n128_teacher_forced_rerun2/main_hidden_only_report.json
    n128_teacher_forced_rerun3/main_hidden_only_report.json
```

现象更像 collective/runtime 或数值非确定性，但尚未形成根因闭环。禁止：

- 借用 v2 历史 `123/128`；
- 修改 fresh oracle；
- 无限重跑直到偶然过线；
- 在未定位根因时改写已推送产品 commit；
- 把 candidate 写成正式 release PASS。

## 6. 下一步

1. 固定同一 clean image、oracle、输入与 cards 0--7，最小化采集 miss step 附近的
   per-rank hidden/logit、collective epoch/window 与 TP spread；
2. 先判定抖动是否出现在 TP all-reduce 前、通信中或通信后，再缩小到 runtime
   scheduling/collective ordering 或具体数值 kernel；
3. 只有定位并修复可复现根因后，才构建新的 clean immutable image；
4. 新镜像连续通过 N=128 `>=95%`、hidden finite 与 TP-spread 合同后，再复跑
   audit/smoke/64K ITL/DFX，记录新 manifest/config，并决定正式发布；
5. attention 发布 gate 闭环后，再继续 Phase 28 的 live front、paged-KV/dynamic batch、
   current Main→MTP absolute oracle 与 3-way HBM 收口。

操作时继续遵守：不触碰其他用户占用的 cards/PID，不使用 `kill -9` 或
`npu-smi reset`，不把历史 dirty checkout 或旧 candidate 当发布依据。
