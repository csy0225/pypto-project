# PyPTO Step3p5 项目

在 Ascend NPU 上做 **step3p5** 大模型的端到端服务化。decoder kernel 走
**pypto** 编程框架，serving / 调度 / batching 走 **vLLM**。

本仓库是**项目级跟踪器**，覆盖 5 个代码仓库。实际代码在别处，本仓只放
状态、阶段跟踪、blocker、部署 spec 和架构 notes。

## 项目涉及的仓库（实际代码在哪里）

| 仓库 | 角色 | 上游 | 我们的 fork |
|------|------|------|------------|
| `pypto` | 编程框架 — multi-level IR + codegen | `hw-native-sys/pypto` | `csy0225/pypto` |
| `pypto-lib` | tensor 级 kernel + step3p5 模型 | `hw-native-sys/pypto-lib` | `csy0225/pypto-lib` |
| `pto-isa` | Tile-ISA 虚拟实现 | `hw-native-sys/pto-isa` | `csy0225/pto-isa` |
| `PTOAS` | LLVM/MLIR PTO 字节码 assembler | `hw-native-sys/PTOAS` | `csy0225/PTOAS` |
| `simpler` | PTO runtime（AICPU+AICore dispatcher） | `hw-native-sys/simpler`（pypto 的 submodule） | `csy0225/simpler` |
| **（集成目标）** vLLM stepcast fork | Serving / 调度 / sampler / tokenizer | 公司内部 stepcast fork | 无 fork |

我们所有的 fork 都在 `stepfun/develop` 分支。pin snapshot 在
[`STATUS.md`](STATUS.md)。

## 一眼看清现在哪儿

**Phase 1 — pypto kernel 原型**：✅ **已完成**（2026-06-22）。详见
[`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md)。

**Phase 2 — vLLM Ascend 后端集成**：🟡 **进行中**（设计已落，实现未启动）。
- Phase 20：vLLM monkey-patch e2e 流程 → [`phases/20-vllm-backend-monkey-patch.md`](phases/20-vllm-backend-monkey-patch.md)
- Phase 21：与 vLLM 原生精度对比 → [`phases/21-precision-validation.md`](phases/21-precision-validation.md)
- Phase 22：perf baseline + 调优 → [`phases/22-perf-baseline.md`](phases/22-perf-baseline.md)

**活跃 blocker**（跨阶段遗留）：见 [`blockers.md`](blockers.md)。

**生产部署**：见 [`deployment/`](deployment/)。Phase 16 三剑合璧绑定是多卡
部署的硬要求。

## 查什么去哪里

| 问题 | 路径 |
|------|------|
| 现在工作状态怎样？ | [`STATUS.md`](STATUS.md) |
| 当前活跃 phase 的任务从哪挑？ | [`phases/`](phases/) |
| 哪些卡住了 / 帮忙解什么？ | [`blockers.md`](blockers.md) |
| 在新机器上怎么部署？ | [`deployment/`](deployment/) |
| 项目架构怎么拼起来的？ | [`architecture/`](architecture/) |
| Phase 01-19 的原型开发是怎么走过来的？ | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| 写 pypto kernel 时有哪些坑？ | `pypto-lib/docs/known-pypto-pitfalls.md`（在 pypto-lib 仓） |
| 开发工作流有什么坑（pyc / 环境 / git）？ | `pypto-lib/docs/dev-workflow-gotchas.md`（在 pypto-lib 仓） |

## 快速起手（在 Phase 16 合规机器上，如 `gpu-a910x-0162`）

```bash
# 1. 三剑合璧环境（CANN 必须是 beta.1，不是 GA）
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa

# 2. 验证 pypto 前端
cd <workspace>/pypto-lib
python -m models.step3p5._smoke_program_build
# 期望: === probe rc=0 ===

# 3. 验证多卡 collective baseline
cd <workspace>/pypto/runtime
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
# 期望: max |out - expected| = 0.000e+00

# 4. 验证单卡 dense decode_layer ST
cd <workspace>/pypto-lib
python -m tests.step3p5.test_decode_layer_full_dense_st -p a2a3 -d 0
# 期望: ratio_allclose PASS，约 8 秒
```

任何一项失败 → 查 [`blockers.md`](blockers.md) 和 pypto-lib reference docs。

## 更新协议

phase / sub-task / blocker 状态变化时：

| 触发 | 更新什么 |
|------|---------|
| sub-task 完成 | 对应 `phases/NN-*.md` 的 Status 段 |
| Phase 准入 / 出 | `STATUS.md` 当前 phase + `phases/README.md` |
| 新 blocker 发现 | `blockers.md` |
| Blocker 解决 | 从 `blockers.md` 删除，可选追加到 `archive/` |
| session 末尾总结 | 追加 entry 到 `archive/milestones-2026-Q2.md` |
| 组件 pin 移动 | `STATUS.md` "Pin snapshot" |

`CLAUDE.md` 只用作 Claude session bootstrap，**保持 50 行以内**。状态 /
历史不要写进去。

## 仓库目录

```
pypto-project/
├── README.md                            本文件
├── CLAUDE.md                            Claude session bootstrap（精简）
├── STATUS.md                            实时状态板
├── blockers.md                          活跃 open issues（SSOT）
├── deployment/                          生产部署 spec
│   ├── README.md
│   ├── phase16-three-pillars.md         driver + firmware + CANN 绑定
│   ├── machine-recovery.md              0162/0234 runbook
│   └── version-matrix.md                5 仓库版本兼容
├── phases/                              活跃 phase 跟踪
│   ├── README.md
│   ├── 20-vllm-backend-monkey-patch.md
│   ├── 21-precision-validation.md
│   └── 22-perf-baseline.md
├── archive/                             历史记录
│   ├── README.md
│   ├── prototype-phase-01-19-summary.md
│   └── milestones-2026-Q2.md
└── architecture/                        跨仓库 design notes
    ├── README.md
    ├── overview.md
    └── vllm-step3p5-mapping.md
```
