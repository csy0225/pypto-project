# 版本矩阵

5 个代码仓 + 3 个工具链支柱的兼容矩阵。下面"已验证组合"表的一行是一个
已知端到端能跑的状态集。跨行混搭**不**支持，混搭后必须重新验证。

## 已验证组合

### 生产目标（2026-06-22）

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

当前 simpler pin (a6e06406) 带 4 个上游还没合的 patch。在
`<workspace>/pypto/runtime` 工作树里跟踪它们。

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
| `vLLM stepcast fork` | Phase 2 集成目标 | `0e0901376` on `develop`（gitlab.basemind.com/sys/stepcast/vllm） |
| `pypto-serving` | 早期 serving wrapper（早于本项目） | 不积极跟踪；需要时见 `<workspace>/pypto-serving/` |

## 相关文档

- [`phase16-three-pillars.md`](phase16-three-pillars.md) —— driver/
  firmware/CANN 为什么硬绑
- [`machine-recovery.md`](machine-recovery.md) —— 怎么安装/升级
- [`../STATUS.md`](../STATUS.md) —— 最新 pin snapshot 一行
- [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
  "Pin snapshot history" —— 历史 pin
