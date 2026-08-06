# 实时状态（STATUS）

> **本文件只放当前真相，不保存流水或旧 pin。最后更新：2026-08-06。**
> 历史记录见 [`archive/`](archive/) 和 [`benchmark/`](benchmark/)；镜像组合见
> [`deployment/version-matrix.md`](deployment/version-matrix.md)。

## 0. Agent 判定当前状态的强制顺序

1. 读取本文件和 [`planning/handoff.md`](planning/handoff.md) 的日期与状态。
2. 用 GitHub 远端 `refs/heads/stepfun/develop` 核对 commit；**不得用本地同名分支、
   worktree 名称或历史 N1 文档推断当前 tip**。
3. 区分“当前源码 tip”和“最新 release-qualified 镜像”。源码前进不代表新镜像已准出。
4. 镜像只认 manifest digest、明确 pin 和 immutable gate；禁止借用旧镜像数据。
5. `develop/N1/`、旧 phase、旧 benchmark 和 hang-debug case study 都是历史证据，
   不能作为当前 checkout、构建 pin 或发布状态。

## 1. 当前源码（已推送）

| 仓库/组件 | 分支或 pin | 当前 commit | 状态 |
|---|---|---|---|
| pypto-lib | `csy0225/pypto-lib:stepfun/develop` | `c9af5790d5fe450e14fd43c88099b87539089d17` | workload-sized attention producer 已合入最新 MoE 基线；immutable ITL/DFX gate PASS |
| pypto | `csy0225/pypto:stepfun/develop` | `8e92b46808f9f7c09b6431ad4691503f09c12ee5` | prepared-worker immutable swimlane dep-gen reuse；`88 passed, 3 skipped` |
| simpler | immutable pin | `e2efebcbd190302609c0775d2984f409f5f42c76` | 当前 canonical image pin |
| pto-isa | immutable pin | `ecb6c303f797749f811a494742c3c08156aacabb` | 当前 canonical image pin |
| PTOAS | immutable pin | `fc8c6caee561914b4fb991dfc8427bb63194269e` | 当前 canonical image pin |
| ptoas-bin | release | `v0.50` | 当前 canonical image pin |
| vLLM overlay | immutable pin | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` | 当前 canonical image pin |

默认 Main 仍为：

```text
models.step3p5.decode_fwd:whole_decode_step3p5
```

## 2. 镜像与验证状态

### 当前 latest-source canonical image

```text
tag:
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260806-attn-taskmajor-canonical
manifest: sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479
config:   sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216
```

该镜像与 §1 当前源码一致。credential、五仓 pin、clean tree、CANN 8.5.1
absence、prepared-swimlane `RunConfig` 和 A2A3
QK/softmax/online blocks-per-task=`22/16/22` profile 审计均 PASS。
0162 digest-only、无源码/runtime overlay 验证：

- 整网 BS1、每请求64K、warmup=5、50 次：
  min/mean/p50/p99/max =
  `39.057/39.594/39.612/40.680/40.680 ms`；hidden finite、TP spread=0；
- 前两层 BS1×64K：p50 `3.6323 ms`，reference exact、TP spread=0；
  DFX `8/8` rank 完整，LOW-WAIT 为 `rank2/d0`。

完整记录：
[`benchmark/2026-08-06-attention-taskmajor-canonical.md`](benchmark/2026-08-06-attention-taskmajor-canonical.md)。

该镜像完成本轮 latest-source Attention/ITL/DFX gate，但尚未重跑 Wave5 的 Main
N=128×3、Main batch16 和 MTP 全矩阵，不能自动继承完整 production
release-qualified 标签。

### 最新完整 release-qualified 回退基线（Wave5）

```text
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260803-attn-final-wave5
manifest: sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32
config:   sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394
```

Wave5 只对 0162 完整 release-qualified；其源码 pin 是 pypto `defa97c5`、pypto-lib
`7099476b`，**不是当前源码 tip**。64K p50 `49.796 ms`，Main N=128 三轮均
`123/128` 且 TP spread=0。

### 历史 2026-08-05 R1/R2（已 supersede）

- R1 已撤销；R2 从未发布，且其 pypto-lib `91c7f46e` 已被 `c9af5790`
  supersede。不得恢复 R2 或用其状态覆盖上面的当前镜像。
- 历史记录：
[`benchmark/2026-08-05-attention-canonical-r1-r2.md`](benchmark/2026-08-05-attention-canonical-r1-r2.md)。

## 3. Attention 当前判断

- Full/SWA 核心计算中主要可避免的调度 bubble 已闭环；logical task 按 workload
  和 architecture profile 推导，不固定 24 个物理核。
- Full/SWA RoPE producer 已改为 workload-sized 单次 SPMD submit，QK 显式依赖
  两个 producer TaskId；A2A3 blocks-per-task profile 为 `22/16/22`、
  reduce fan-in=8。
- Full Pass-A 已并入 SV；只保留必要的 online-softmax reduce/finalize。
- Full/SWA out-proj cast 均融合。
- 已证伪或无稳定收益的 AR+residual、residual+RMS、RMS+projection 等方案不合入。
- focused 两层矩阵已覆盖 bs1/2/4/8/16/7、每请求64K；当前 immutable 镜像
  两层 BS1 p50 `3.6323 ms`，输出 exact，DFX task count 为 `24/32/24`。
- 当前 immutable 整网 BS1×64K p50 `39.612 ms`，相对 Wave5 下降 `20.45%`；
  该比较跨越最新 MoE 等整栈改动，不能把全部收益归因于 Attention。
- 整网 bs16×每请求64K 在 prewarm 前约 `52,013 MiB/卡` 的基础上申请约
  16 GiB static arena，`rtMalloc 207001`；没有有效 bs16 ITL。
- 后续优先级是完整 production matrix 与 BS16 容量门禁，其次才是跨架构
  profile 校准和可证明的 collective overlap。

设计入口：
[`design/performance/04-attention-optimization.md`](design/performance/04-attention-optimization.md)。

## 4. 当前下一步

1. 若提升为完整 production release，按 Wave5 同口径补 Main N=128×3、
   Main batch16、MTP batch1/16 和 smoke/precision matrix。
2. BS16×每请求64K 必须先通过 runtime-memory 容量门禁；不能把 OOM 或两层数据
   写成整网性能。
3. 新架构重新 sweep workload task grain；不能把 A2A3 blocks-per-task
   `22/16/22` 或物理核心数当作跨架构常量。

## 5. 其它项目级 active work

真实 vLLM live front、paged-KV/dynamic batch、同代 Main→MTP absolute gate 和
3-way HBM 仍未闭环；这些属于 serving 集成，不改变本轮 attention/R2 的准出顺序。
旧 N1 standalone、0234 stall 和早期 pin 只保留为历史案例，不再列为当前源码状态。

## 6. 机器状态口径

0162 是本轮验证机（driver `25.5.2` / firmware `7.8.0.7.220` /
CANN `9.0.0-beta.1`）。ITL/DFX 完成后 container 已退出，16 张卡均无 NPU
process；后续作业前仍须重新检查卡占用，不能沿用旧 session 的空闲结论。
