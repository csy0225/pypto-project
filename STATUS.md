# 实时状态（STATUS）

> **本文件只放当前真相，不保存流水或旧 pin。最后更新：2026-08-08。**
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
| pypto-lib | `csy0225/pypto-lib:stepfun/develop` | `491267c45875e9b1e0071eed224e2e73526799e2` | 远端 tip；包含 active-route scheduling、SWA mask 修复和 MoE route/precision release harness；最终 immutable image 尚未构建 |
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

### 最终统一发布镜像（待构建）

截至 2026-08-08，**没有 immutable image 包含**
pypto-lib=`491267c45875e9b1e0071eed224e2e73526799e2`。本次只完成代码与文档收尾，
不把 pending spec 或历史 digest 标成发布镜像。下一次按
`deployment/docker/builds/stepfun-develop-20260808-moe-opt-latest-source.env`
构建并执行 0162 标准回归。

因此，下方两个 digest 都是旧源码层级的 **pre-fix evidence**，不能标成当前源码
的最终发布镜像，也不能把其 golden、性能或 DFX 自动升级为
`491267c4` 的准出结论。

### 最近一次 Attention canonical image（pre-fix evidence）

```text
tag:
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260806-attn-taskmajor-canonical
manifest: sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479
config:   sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216
```

该镜像绑定 pypto-lib=`c9af5790`，**不包含** §1 的 SWA mask 修复
`63814d4a`。其 credential、五仓 pin、clean tree、CANN 8.5.1 absence、
prepared-swimlane `RunConfig` 和 A2A3
QK/softmax/online blocks-per-task=`22/16/22` profile 审计均 PASS。
0162 digest-only、无源码/runtime overlay 验证：

- 整网 BS1、每请求64K、warmup=5、50 次：
  min/mean/p50/p99/max =
  `39.057/39.594/39.612/40.680/40.680 ms`；hidden finite、TP spread=0；
- 前两层 BS1×64K：p50 `3.6323 ms`，reference exact、TP spread=0；
  DFX `8/8` rank 完整，LOW-WAIT 为 `rank2/d0`。

完整记录：
[`benchmark/2026-08-06-attention-taskmajor-canonical.md`](benchmark/2026-08-06-attention-taskmajor-canonical.md)。

该镜像只完成 `c9af5790` 层级的 Attention/ITL/DFX gate；SWA mask 修复后，
这些性能与 DFX 只能作 pre-fix 对照。它也未重跑 Wave5 的 Main N=128×3、
Main batch16 和 MTP 全矩阵，不能自动继承完整 production release-qualified 标签。

### L0–L4 MoE formal image（pre-fix evidence）

```text
hub.i.basemind.com/stepcast/vllm-pypto@sha256:
  cab89668164cf85dc75e4f3ac53ef77ef4b8653767c7d147c5113cdee6a9d88c
```

该 digest 绑定 pypto-lib=`c9af5790`、pypto=`8e92b468`、attention profile=`a2a3`
和 prepared-swimlane reuse capability。0162 的 focused normal A/B 已完成
baseline/candidate × BS `1/2/4/7/8/16` × 3 轮，共 36/36 fresh-process run；
每条 sequence 独立 `context_len=65536`。六档 L3/L4 hidden 跨轮 hash exact，
性能均无回退。seal：

```text
/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-c9af-20260806-v2/
  campaign/normal_seal_authority.json
SHA256 875804ddbb81b4f15a907e41e454ed3004aca3b56075063431edef5efc70c531
```

该 campaign 在 `c9af5790` 上已 seal，但 SWA mask 随 `63814d4a` 发生源码变化，
因此旧 L3/L4 golden 与性能数据不能自动升级为最终 release evidence。统一发布
commit 确定后，六档每请求独立 64K、双 hidden golden 和 A/B 必须在最终镜像上重跑。

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

- R1 已撤销；R2 从未发布，且其 pypto-lib `91c7f46e` 已被当前
  `491267c4` supersede。不得恢复 R2 或用其状态覆盖上面的当前镜像。
- 历史记录：
[`benchmark/2026-08-05-attention-canonical-r1-r2.md`](benchmark/2026-08-05-attention-canonical-r1-r2.md)。

## 3. Attention 当前判断

- `63814d4a` 将 SWA tail-window mask 从 `pl.cmp` predicate 转换路径改为显式
  typed INT32 数值区间 mask，避免 predicate 数值转换破坏 sliding-window
  score mask。
- 0162 使用 `cab896…` substrate + `63814d4a` 精确 source overlay 的 N=128
  teacher-forced 回归为 `127/128=99.21875%`，唯一 miss 是
  `step94 expected=478 actual=320`，`hidden_tp_spread_max=0.0`，已通过
  `>=95%` source-level 精度门。证据：
  `/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-precision-fix-20260807-v2/runs/Lmask-v1-n128/summary.json`
  （SHA256 `7f91dcdb…`）。
- source-level PASS 不等于 immutable-image PASS；最终镜像上的精度、性能和 DFX
  仍需重跑，当前不能宣称 SWA 修复无性能回退。
- Full/SWA 核心计算中主要可避免的调度 bubble 已闭环；logical task 按 workload
  和 architecture profile 推导，不固定 24 个物理核。
- Full/SWA RoPE producer 已改为 workload-sized 单次 SPMD submit，QK 显式依赖
  两个 producer TaskId；A2A3 blocks-per-task profile 为 `22/16/22`、
  reduce fan-in=8。
- Full Pass-A 已并入 SV；只保留必要的 online-softmax reduce/finalize。
- Full/SWA out-proj cast 均融合。
- 已证伪或无稳定收益的 AR+residual、residual+RMS、RMS+projection 等方案不合入。
- pre-fix focused 两层矩阵已覆盖 bs1/2/4/8/16/7、每请求64K；`c9af5790` 镜像
  两层 BS1 p50 `3.6323 ms`，输出 exact，DFX task count 为 `24/32/24`。
- pre-fix immutable 整网 BS1×64K p50 `39.612 ms`，相对 Wave5 下降 `20.45%`；
  该比较跨越最新 MoE 等整栈改动，不能把全部收益归因于 Attention。
- 整网 bs16×每请求64K 在 prewarm 前约 `52,013 MiB/卡` 的基础上申请约
  16 GiB static arena，`rtMalloc 207001`；没有有效 bs16 ITL。
- 后续优先级是完整 production matrix 与 BS16 容量门禁，其次才是跨架构
  profile 校准和可证明的 collective overlap。

设计入口：
[`design/performance/04-attention-optimization.md`](design/performance/04-attention-optimization.md)。

## 4. MoE 当前判断

- 产品改动 `7928a275` 已包含在当前远端 `stepfun/develop@491267c4` 中；
  `cd19fe6b` 的 active-route scheduling 和 `491267c4` 的 route/precision
  release harness 也已进入当前源码 tip。
- 当前 tip 的 `decode_fwd.py` SHA256=`4b39aec7…`。旧正式 campaign 的 candidate
  SHA256=`7884da7c…`、baseline `56b3d477` SHA256=`3553664c…` 只绑定历史
  source policy。
- `c9af5790` pre-fix 六档 focused normal A/B 与 L3/L4 hidden golden 已通过；
  p50 改善分别为 `9.16/1.83/3.52/6.07/0.53/11.61%`，但最终镜像必须重跑。
- matched-source whole-net baseline/candidate 的 1-step×2、2-step×2 共 8/8 run
  均通过，输出分别为 `303` 和 `303,1207`；publication seal=`PASS`：
  `.../whole-net-matched-ab-20260807T024525Z/publication_seal_report.json`
  （SHA256 `c0a03127…`）。
- J1 保持 🟦/NO-GO：source-overlay N=128 已通过，但 `491267c4` 对应的 final
  immutable image 精度、六档 64K golden/A/B、formal matched-source DFX 12 runs、
  route-aware reanalysis 和 all-rank swimlane 尚未完成。

设计入口：
[`design/performance/05-moe-optimization.md`](design/performance/05-moe-optimization.md)。

## 5. 当前下一步

1. 按 pending spec 构建包含 `491267c4` 的 immutable image。
2. 在最终镜像上先完成 whole-net N=128 多步精度，再重跑 BS
   `1/2/4/7/8/16`、每请求独立 64K、L3/L4 golden 与 counterbalanced A/B。
3. 为最终 image/source 重新生成 matched source policy：current candidate 必须绑定
   `4b39aec7…`，baseline 从选定的 immutable control source 独立计算；完成 MoE
   formal all-rank DFX/swimlane 和 fail-closed 重分析。不得把历史
   `baseline=3553664c`、`candidate=7884da7c` policy 直接沿用为当前准出。
4. 用 `pypto-image-verify` 与 `pypto-perf-regression` 对最终 immutable image
   执行标准回归。
5. 若提升为完整 production release，按 Wave5 同口径补 Main N=128×3、
   Main batch16、MTP batch1/16 和 smoke/precision matrix。
6. BS16×每请求64K 必须先通过 runtime-memory 容量门禁；不能把 OOM 或两层数据
   写成整网性能。
7. 新架构重新 sweep workload task grain；不能把 A2A3 blocks-per-task
   `22/16/22` 或物理核心数当作跨架构常量。

## 6. 其它项目级 active work

真实 vLLM live front、paged-KV/dynamic batch、同代 Main→MTP absolute gate 和
3-way HBM 仍未闭环；这些属于 serving 集成，不改变本轮 attention/R2 的准出顺序。
旧 N1 standalone、0234 stall 和早期 pin 只保留为历史案例，不再列为当前源码状态。

## 7. 机器状态口径

0162 是本轮验证机（driver `25.5.2` / firmware `7.8.0.7.220` /
CANN `9.0.0-beta.1`）。ITL/DFX 完成后 container 已退出，16 张卡均无 NPU
process；后续作业前仍须重新检查卡占用，不能沿用旧 session 的空闲结论。
