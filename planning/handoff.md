# 接力上下文（Handoff）

> **这是 ephemeral 接力文档**——给“接着干”的 agent 一页纸当前工作面。
> durable 规划看 [`roadmap.md`](roadmap.md)，实时状态看
> [`../STATUS.md`](../STATUS.md)，发布 blocker 看 [`../blockers.md`](../blockers.md)，
> 完整证据看
> [`../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md`](../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md)。
> **最后更新：2026-08-03。** 更新时直接改写本文，不追加流水。

## 1. 当前 source of truth

```text
machine:    gpu-a910x-0162
devices:    0..7（本轮验证）；8..15 及 PID 2045390--2045397 未操作
checkpoint: /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp

pypto-lib / vllm-pypto:
  7099476b7c4f13112b159e237e7a64344803caf0
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

GitHub `csy0225/pypto-lib:stepfun/develop` 已指向 `7099476b`。不要使用 dirty 的
`workspace/pypto-lib`，也不要把裸机 overlay 当 source of truth。

## 2. Wave5 canonical release（0162 release-qualified）

```text
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260803-attn-final-wave5

manifest:
  sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32

config/image ID:
  sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394
```

这是 **0162 release-qualified 的 immutable canonical release**。只挂载
driver(ro)、checkpoint(ro)、output(rw)，没有宿主源码挂载。其它机器或架构没有由
本轮独立证明，不能无条件外推。

```text
source integration / clean tree         PASS
audit / smoke / Main+MTP compile        PASS
Main N=128 predefined runs              3/3 PASS
Main batch16 + MTP batch1/batch16       PASS
64K + batch16 ITL/DFX                   PASS
0162 release gate                       PASS
```

`7099476b` 将 Wave 1 前的 source partial publication 从普通 local store 改为
self-target synchronous TPUT，再保持既有三波 reduce-scatter + push all-gather：

```text
local source
-> self-target drained TPUT
-> Wave 1 source publication
-> rank-owned reduce-scatter
-> push all-gather
-> Wave 2 result publication
-> final local copy
-> Wave 3 lifetime close
```

修改同步覆盖 canonical Main、selected MTP、two-layer harness，并修正 MTP input
projection 的 all-reduce 返回值 lineage。focused contracts 为 `25 passed`。当前证据
支持 source publication/lifetime ordering 是 0162 的关键边界，但不宣称它是所有硬件的
唯一根因。

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

## 4. Wave3/Wave4/Wave5 对账

| 版本 | 作用 | N=128 | 结论 |
|---|---|---|---|
| `76d96bdb` clean canonical | 历史基线 | 三轮均 `121/128`，部分轮有 spread | release blocked |
| `d58b6be7` Wave3 | canonical window lifetime 三波闭合 | `124/128`，miss `[2,8,13,82]`，spread=0 | 修复有效；harness 未对齐 |
| `d7e1381b` Wave4 | harness 与 canonical 三波协议/AST 对齐 | Run1 `122/128`、step2 spread=2.0；Run2 `123/128`、spread=0 | 历史 candidate；source publication 仍有 race |
| `7099476b` Wave5 | self-target TPUT 发布 source；Main/MTP/harness/lineage 对齐 | 预定义三轮均 `123/128`、miss `[2,8,13,22,82]`、spread=0 | **0162 release-qualified** |

固定 oracle：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  fresh_vanilla_oracle_20260731/oracle_ids.json
sha256=c9b2c72121880e9c605ae70d1cf85c0d4fc8815b180598bc76f7e293551dd947
```

## 5. 性能、batch16、MTP 与 DFX

Wave5 64K（bs=1、512 blocks、warmup=3、20 measured iterations）：

```text
min  = 48.523 ms
mean = 50.027 ms
p50  = 49.796 ms
p99  = 54.539 ms
max  = 54.539 ms
```

active-batch=16/context=1：

```text
min  = 112.525 ms
mean = 112.819 ms
p50  = 112.827 ms
p99  = 113.203 ms
max  = 113.203 ms
```

Main batch16 为 `8/8 exact`、finite、active rank rows=`128`、TP spread=0。
MTP batch1/batch16 两轮均为 token `[6178,410,303]`、pass rate 1.0、
max diff 0、TP spread=0。

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_wave5_20260803_immutable/
    main_n128_predefined3/verified_summary.json
    batch16_mtp_campaign/verified_summary.json
    mtp_compile/result.txt
    itl_dfx/verified_summary.json
```

DFX LOW-WAIT heuristic 均为 rank2：64K makespan `38.367 ms`、TP AR compute
`2.437 ms`；batch16 makespan `107.076 ms`、TP AR compute `2.429 ms`。其余 rank
的超长 TP AR 主要吸收 kernel 内自旋等待，不可当真实算术耗时。

## 6. 下一步与硬边界

1. attention/Vec 与本轮 all-reduce stability 已收尾；不要再机械融合
   reduce/finalize、AR+residual 或 RMS/Projection；
2. Wave5 只在 0162 标记 release-qualified；跨机器/跨架构仍需独立 immutable gate；
3. 若 all-reduce 协议、window lifetime、数值顺序或镜像 pin 再变化，重新完成
   immutable audit、Main N=128 预定义轮次、batch16/MTP、64K/batch16 ITL 与 DFX；
4. 主线转回 Phase 28：live front、paged-KV/dynamic batch、
   current Main→MTP absolute oracle 与 3-way HBM。

继续遵守：只使用分配的 cards，不触碰 PID `2045390--2045397`，不使用 `kill -9` 或
`npu-smi reset`。
