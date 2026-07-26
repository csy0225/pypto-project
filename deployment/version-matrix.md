# 版本矩阵

5 个代码仓 + 3 个工具链支柱的兼容矩阵。下面"已验证组合"表的一行是一个
已知端到端能跑的状态集。跨行混搭**不**支持，混搭后必须重新验证。

## 已验证组合

### 当前 release（2026-07-26）

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `ca21ab5fcfd8203165928428302d273c377db5c6` | 与 0724 镜像一致 |
| pypto-lib / vllm-pypto | `53eb7212c29c9bd015ee060cd9924a13ea781ae0` | `stepfun/develop`；唯一 Main=`models.step3p5.decode_fwd:whole_decode_step3p5`；`step3p5_opt` package/aliases 已删除 |
| pto-isa | `ecb6c303f797749f811a494742c3c08156aacabb` | 与 0724 镜像一致 |
| PTOAS | `fc8c6caee561914b4fb991dfc8427bb63194269e` | 与 0724 镜像一致 |
| simpler | `216e7632267ae815c484cdeba7991c87fabf3086` | pypto `runtime` gitlink；与 0724 镜像一致 |
| ptoas-bin | `v0.50` | binary sha256 `ba93fabeff6dc7fdcd2278a72fd1d4fd92cb2949faedbc83fa58e801bd5ff23b` |
| vLLM overlay | `csy/pypto-tail-mtp-integration@1b3e538c35999e62b6d24e0651b3a85b7d16c826` | build 时按 commit checkout，不能只依赖可变 branch |
| Python | `3.11.14` | 镜像内 `/usr/local/python3.11.14/bin/python3` |
| 镜像 | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260726-step3p5-only@sha256:99b2b9718cfa6bf0bb87b221f7d565bf23afd2b89a30ba150e523c44a536ed81` | config `sha256:d296461051559e6ea0e22d04a4cc44f749c82f19a50418fe6db75387f1f067e9`；0162 credential/symbol/ldd audit + smoke + unit/contract/device regression PASS |

验证结论：

- 0162 overlay 环境默认 `whole_decode_step3p5` 已完成 N=256；
- N=256 canonical-only 与清理前镜像 token/hidden `256/256` exact，`max_abs_diff=0`，
  TP spread `0.0`，compatibility removal regression PASS；
- 与清理前 canonical 镜像产物 token/hidden `256/256` bit-exact，
  step127/128/255 PASS；
- 对同一 vanilla oracle，canonical-only 为 `240/256=93.75%`，低于历史
  `>=95%` raw gate；不能写成 vanilla raw precision PASS；
- `--baseline-main` rollback 编译成功并完成 3-step device smoke；step2
  退出是已知 stale hardcoded oracle，不是 rollback 失效；
- 最终镜像内默认 holder 实际打印
  `program=whole_decode_step3p5`；8-step device smoke hidden 全 finite、
  TP spread `0.0`，除已知 stale oracle step2 外其余 `7/8` token exact；
- 新镜像 N=256 raw `240/256=93.75%`；与既有 canonical N=256 artifact
  token/hidden `256/256` exact、`max_abs_diff=0`、TP spread `0.0`，
  step127/128/255 全通过；
- 当前只证明 canonical-only Main replacement，不等价于完整 production Main+MTP serving
  已无条件平替。

构建 spec：
[`docker/builds/stepfun-develop-20260726-step3p5-only.env`](docker/builds/stepfun-develop-20260726-step3p5-only.env)。

### 历史生产目标（2026-06-22，禁止作为当前 pin）

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | Phase 16 最小 |
| Firmware | `7.8.0.7.220` | chip flash，持久 |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `csy0225/pypto stepfun/develop:b00c8b23` | 比 origin/main 多 3 commit（DFX env hook + repros + simpler submodule pin） |
| pypto-lib | `csy0225/pypto-lib stepfun/develop:9c4773f` | 比 origin/main 多 ~9 commit（step3p5 模型 + Phase 19 padding + ST 脚手架 + dev-workflow docs；误置的 phase tracker 已撤回） |
| pto-isa | `csy0225/pto-isa stepfun/develop:e25732f0` | = origin/main（无本地 patch） |
| PTOAS | `csy0225/PTOAS stepfun/develop:da011a3d` | = origin/main；binary `ptoas-bin` `v0.45` |
| simpler | `csy0225/simpler a6e06406`（pypto submodule） | 比 origin/main 多 4 patch（zero-size view + `--no-as-needed` libhcomm + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip） |
| ptoas-bin | `v0.45` | binary release |
| Python | `3.11.14` | venv 在 `<workspace>/.venv311` |

验证证据见 [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
"2026-06-22（早段）—— 验证基线"。

## 兼容规则

### pypto / pto-isa / PTOAS / ptoas-bin

pypto codegen 产 MLIR 给 PTOAS 吃。wire format 会偶尔变；mismatched
pypto + ptoas-bin 编译时会报 parser error。

历史已知 mismatch：
- pypto 越过 `505abd64`（TCIOp `hasCustomAssemblyFormat`）之后需要
  ptoas-bin ≥ `v0.45`。Phase 19 blocker 1 就是这个 mismatch —— pypto
  跑前了，ptoas-bin 还在 `v0.44`。

规则：bump pypto 跨过会动 MLIR op 的上游 commit 时，同时 bump ptoas-bin。

### pypto / simpler

simpler 是 pypto 的 git submodule，在 `pypto/runtime/`。`pypto` 仓的
pin 决定编哪个 simpler commit。更新 simpler 时必须
`git submodule update` 并 commit pypto 侧的 submodule pin。

当前 release 的 simpler pin 是 `216e7632`，并由 pypto `ca21ab5f`
的 `runtime` gitlink固定。下方 `a6e06406` 仅属于 2026-06-22 历史组合。

### CANN

CANN beta.1 **必需**。CANN GA 会让 simpler init 失败（见
[`phase16-three-pillars.md`](phase16-three-pillars.md) "CANN GA failure
mode"）。**不要**升级 CANN 除非 Huawei 出了新 beta 或 GA 明确修复了
AICPU `libaicpu_extend_kernels.so` push path。

### Driver + firmware

总是成对。driver-only 或 firmware-only 升级未验证。
`support_shmem_map_exbus` cap 由两者共同 gate。

## 升级顺序（全部前进时）

推荐顺序：

1. Firmware（写 chip flash；先做，其余还在老版本上）
2. Driver（重装到 host filesystem；要 daemonset drain）
3. 重启主机
4. CANN（**只**在 Huawei 出新 beta/GA 验证过兼容时）
5. simpler（pypto submodule）
6. pypto + pto-isa + PTOAS + pypto-lib（任意顺序，但重装时按
   pypto → pto-isa → PTOAS → pypto-lib 顺序）
7. ptoas-bin（binary drop-in，跟 PTOAS source pin 配对）

每一步后都跑 smoke + simpler L3 allreduce 验证。

## 项目之外但邻接的仓库

| 仓库 | 角色 | 我们跟踪的 pin |
|------|------|----------------|
| `vLLM stepcast fork` | Phase 2 集成目标 | `csy/pypto-tail-mtp-integration@1b3e538c`（gitlab.basemind.com/sys/stepcast/vllm） |
| `pypto-serving` | 早期 serving wrapper（早于本项目） | 不积极跟踪；需要时见 `<workspace>/pypto-serving/` |

## 相关文档

- [`phase16-three-pillars.md`](phase16-three-pillars.md) —— driver/
  firmware/CANN 为什么硬绑
- [`machine-recovery.md`](machine-recovery.md) —— 怎么安装/升级
- [`../STATUS.md`](../STATUS.md) —— 最新 pin snapshot 一行
- [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
  "Pin snapshot history" —— 历史 pin
