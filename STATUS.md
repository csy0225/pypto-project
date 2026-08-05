# 实时状态（STATUS）

> **本文件只放当前真相，不保存流水或旧 pin。最后更新：2026-08-05。**
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
| pypto-lib | `csy0225/pypto-lib:stepfun/develop` | `91c7f46ee949045e2fce807276412b48d8121763` | attention workload/task-grain 收口；`218 passed, 3 skipped` |
| pypto | `csy0225/pypto:stepfun/develop` | `8e92b46808f9f7c09b6431ad4691503f09c12ee5` | prepared-worker immutable swimlane dep-gen reuse；`88 passed, 3 skipped` |
| simpler | immutable pin | `e2efebcbd190302609c0775d2984f409f5f42c76` | R2 规格 pin |
| pto-isa | immutable pin | `ecb6c303f797749f811a494742c3c08156aacabb` | R2 规格 pin |
| PTOAS | immutable pin | `fc8c6caee561914b4fb991dfc8427bb63194269e` | R2 规格 pin |
| ptoas-bin | release | `v0.50` | R2 规格 pin |
| vLLM overlay | immutable pin | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` | R2 规格 pin |

默认 Main 仍为：

```text
models.step3p5.decode_fwd:whole_decode_step3p5
```

## 2. 镜像与验证状态

### 最新 release-qualified 镜像（仍是 Wave5）

```text
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260803-attn-final-wave5
manifest: sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32
config:   sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394
```

Wave5 只对 0162 release-qualified；其源码 pin 是 pypto `defa97c5`、pypto-lib
`7099476b`，**不是当前源码 tip**。64K p50 `49.796 ms`，Main N=128 三轮均
`123/128` 且 TP spread=0。

### 2026-08-05 R1/R2

- **R1：REVOKED。** 两层 bs1/64K timing 和数值检查完成，但镜像缺
  `l2_swimlane_reuse_dep_gen`，DFX 失败，没有最终 swimlane。
- **R2：BUILD PAUSED / UNPUBLISHED / UNVERIFIED。** 规格为
  `deployment/docker/builds/stepfun-develop-20260805-attn-final-canonical-r2.env`，
  pin pypto `8e92b468`、pypto-lib `91c7f46e`，`BUILD_JOBS=1`。
- 当前没有 R2 manifest/config digest、最终两层 swimlane 或同镜像整网 ITL。
- 按用户要求 build 已暂停；未经新指示不得恢复，也不得源码挂载绕过 immutable gate。

完整记录：
[`benchmark/2026-08-05-attention-canonical-r1-r2.md`](benchmark/2026-08-05-attention-canonical-r1-r2.md)。

## 3. Attention 当前判断

- Full/SWA 核心计算中主要可避免的调度 bubble 已闭环；logical task 按 workload
  和 architecture profile 推导，不固定 24 个物理核。
- Full Pass-A 已并入 SV；只保留必要的 online-softmax reduce/finalize。
- Full/SWA out-proj cast 均融合。
- 已证伪或无稳定收益的 AR+residual、residual+RMS、RMS+projection 等方案不合入。
- 后续优先级是 immutable R2 复核，其次才是 RoPE/KV-cache staging、跨架构 profile
  校准和可证明的 collective overlap。

设计入口：
[`design/performance/04-attention-optimization.md`](design/performance/04-attention-optimization.md)。

## 4. 恢复 R2 后的固定准出顺序

1. 保持 `BUILD_JOBS=1` 完成 R2，并执行 credential、pin、import、RunConfig 审计。
2. push 后固定 manifest/config digest；0162 只允许 digest-only、无源码挂载运行。
3. 清除八个 attention 单参数 override，只保留 `PYPTO_STEP3P5_ATTN_TASK_PROFILE=a2a3`。
4. 先采 bs1/context=65536 前两层 attention DFX，返回具体 rank 的
   `l2_swimlane_records.json`。
5. 再用同一 digest 采整网 50 次 ITL，报告 min/mean/p50/p99/max 和正确性。

## 5. 其它项目级 active work

真实 vLLM live front、paged-KV/dynamic batch、同代 Main→MTP absolute gate 和
3-way HBM 仍未闭环；这些属于 serving 集成，不改变本轮 attention/R2 的准出顺序。
旧 N1 standalone、0234 stall 和早期 pin 只保留为历史案例，不再列为当前源码状态。

## 6. 机器状态口径

0162 是本轮验证机（driver `25.5.2` / firmware `7.8.0.7.220` /
CANN `9.0.0-beta.1`）。当前仅确认本机和 0162 没有 R2 build/compile 残留进程；
每次恢复作业前仍须重新检查卡占用和保护进程，不能沿用旧 session 的空闲结论。
