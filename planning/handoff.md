# 接力上下文（Handoff）

> **这是 ephemeral 接力文档**——给“接着干”的 agent 一页纸当前工作面。
> durable 规划看 [`roadmap.md`](roadmap.md)，实时状态看
> [`../STATUS.md`](../STATUS.md)，发布 blocker 看 [`../blockers.md`](../blockers.md)，
> 完整证据看
> [`../benchmark/2026-08-02-step3p5-attention-final.md`](../benchmark/2026-08-02-step3p5-attention-final.md)。
> **最后更新：2026-08-03。** 更新时直接改写本文，不追加流水。

## 1. 当前 source of truth

```text
machine:    gpu-a910x-0162
devices:    0..7（本轮验证）；8..15 及 PID 2045390--2045397 未操作
checkpoint: /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp

pypto-lib / vllm-pypto:
  d7e1381be0236d6e068cd4d86aa815ea693ea5c7
  branch stepfun/develop
pypto:
  defa97c526fec7e8f032dbbfcc39c820add02bf7
  branch stepfun/develop
simpler: e2efebcbd190302609c0775d2984f409f5f42c76
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

GitHub `csy0225/pypto-lib:stepfun/develop` 已指向 `d7e1381b`。不要使用 dirty 的
`workspace/pypto-lib`，也不要把裸机 overlay 当 source of truth。

## 2. Wave4 immutable candidate

```text
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260802-attn-final-wave4

manifest:
  sha256:8125c678779c332d196b3d770242659d9a86185e0a8d96d89681647b00c864ab

config/image ID:
  sha256:c340001f791bd4666310b2f1755daba5492fec8c65f126888d46ed4366131c92
```

这是 **immutable candidate，不是正式 release**。只挂载 driver(ro)、checkpoint(ro)、
output(rw)，没有宿主源码挂载。

```text
source integration                PASS
audit/smoke/compile/ITL/DFX       PASS
N=128 raw token gate              PASS twice
TP-spread stability contract      NOT YET STABLE
formal release                    PENDING
```

`d58b6be7` 给 canonical all-reduce 增加第三 completion wave；`d7e1381b` 对齐
两层 harness 并增加 AST equality contract。focused contracts 为 `28 passed`。

## 3. attention / Vec 最终设计决策

- logical task 按 active rows、每行真实 `seq_len` 和 architecture-profile grain 推导，
  不固定 24 个物理核心；runtime 映射到 AIC/AIV wave；
- `5--10 us/task` 只是 sweep 起点，最终联合比较 duration、stage span、wave/core-wait、
  packing、tail、dispatch、reduction/finalize dependency tail 与 batch16；
- A2A3 profile：Full QK `22 blocks/task`、softmax `12`、SV+segment recurrence `16`、
  reduce fan-in `8`；Full/SWA out-proj `N=64`、`3 tiles/task`、vector `N=128`、
  cast fusion 开启；TP all-reduce chunk `512`；
- Full Pass-A 已并入 `full_sv_matmul`，只保留必要的
  `full_online_softmax_reduce/finalize`；SWA 保持 row-oriented；
- Full/SWA decode out-proj cast 均融合；fallback/prefill 独立 cast 仍保留；
- 保留 dense RMS direct BF16 reread、dense down-proj cast fusion；
- AR+residual、residual+RMS stats、RMS+projection、gate/up+SiLU 等 probe 无稳定收益，
  不合 canonical；
- `BATCH=16` 是 storage capacity，不是固定 logical batch；
- 保持 PyPTO Orchestration/InCore/runtime 分层，不增加 app-side work stealing。

## 4. Wave3/Wave4 对账

| 版本 | 作用 | N=128 | 结论 |
|---|---|---|---|
| `76d96bdb` clean canonical | 历史基线 | 三轮均 `121/128`，部分轮有 spread | release blocked |
| `d58b6be7` Wave3 | canonical window lifetime 三波闭合 | `124/128`，miss `[2,8,13,82]`，spread=0 | 修复有效；harness 未对齐 |
| `d7e1381b` Wave4 | harness 与 canonical 三波协议/AST 对齐 | Run1 `122/128`、step2 spread=2.0；Run2 `123/128`、spread=0 | 最新 candidate；稳定性待闭环 |

固定 oracle：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  fresh_vanilla_oracle_20260731/oracle_ids.json
sha256=c9b2c72121880e9c605ae70d1cf85c0d4fc8815b180598bc76f7e293551dd947
```

## 5. 性能、batch16 与 DFX

Wave4 64K（bs=1、512 blocks、warmup=3、20 measured iterations）：

```text
min  = 48.316 ms
mean = 50.529 ms
p50  = 50.204 ms
p99  = 56.355 ms
max  = 56.355 ms
```

其他点：active-batch=1/context=1 p50 `43.273 ms`；
active-batch=16/context=1 p50 `112.773 ms`。

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_wave4_20260802_immutable/itl64k/
    itl_report.json
    low_wait_reference.json
    critical_path_stdout.txt
```

LOW-WAIT reference：

```text
.../build_output/WholeDecodeStep3p5_20260803_000049/
  dfx_outputs/rank2/d0/
    critical_path_report.md
    merged_swimlane_20260803_000143.json
```

rank2 makespan `38.504 ms`，TP AR critical-path compute `2.125 ms`。其余 rank 的
超长 TP AR 主要吸收 kernel 内自旋等待，不可当真实计算量。

## 6. 下一步与硬边界

1. attention/Vec 产品优化本身可收尾；不要再机械融合 reduce/finalize、AR+residual 或
   RMS/Projection；
2. Wave4 只标 `immutable candidate / raw token gate PASS / formal release pending`；
3. 若继续 release gate，先定义有限次数和判据，再做 TP-spread 稳定性/根因实验；
   不得无限重跑挑选结果；
4. 新源码或新镜像出现后，重新完成 immutable audit、N=128、64K ITL 与 DFX；
5. 发布门闭环后，主线转回 Phase 28：live front、paged-KV/dynamic batch、
   current Main→MTP absolute oracle 与 3-way HBM。

继续遵守：只使用分配的 cards，不触碰 PID `2045390--2045397`，不使用 `kill -9` 或
`npu-smi reset`。
