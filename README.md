# PyPTO Step3p5 项目

在 **Ascend 910B NPU** 上把 **step3p5** 大模型端到端服务化：decoder kernel 走
**pypto** 编程框架，serving/调度/batching/sampling 走 **vLLM**（公司内部 stepcast
fork）。本仓是**项目级跟踪 + 设计 + 部署 + 复盘仓**——实际代码在 5 个 sub-repo +
vLLM fork（见 [`design/00-context-and-goals.md`](design/00-context-and-goals.md)）。

## 🧭 给别人介绍本项目，按这条线读

1. [`design/00-context-and-goals.md`](design/00-context-and-goals.md) —— **背景/目标/全景**（第一份）
2. [`design/whole-net/01-system-design.md`](design/whole-net/01-system-design.md) + [`design/vllm-pypto/01-system-design.md`](design/vllm-pypto/01-system-design.md) —— **两个子系统架构**（含流程图/时序图）
3. [`planning/roadmap.md`](planning/roadmap.md) —— **进度 / 路线图**
4. [`STATUS.md`](STATUS.md) —— **此刻状态一页纸**

深入技术再下钻到两份 `02-detailed-design.md`（LLD）。

## 📁 仓库怎么组织（7 分区）

| 分区 | 放什么 |
|------|--------|
| [`design/`](design/) | **软件工程设计**：context + 两子系统的 系统设计(HLD) + 详细设计(LLD) |
| [`planning/`](planning/) | **整体规划**：roadmap + 活跃 phase + ephemeral handoff |
| [`postmortems/`](postmortems/) | **工程专项复盘**（12 篇，标准五段：背景/现象/根因/解决/弯路/避免） |
| [`deployment/`](deployment/) | **生产部署 runbook**：三剑合璧 / 机器恢复 / 版本矩阵 |
| [`reference/`](reference/) | **参考资料**：canonical 测试、4+1 视图、编程 API、约束 |
| [`archive/`](archive/) | **历史**（追加式）：session 日志、原型摘要、已完成 phase、交付快照 |
| 根 | [`STATUS.md`](STATUS.md)（当前状态）· [`blockers.md`](blockers.md)（活跃 open）· [`GLOSSARY.md`](GLOSSARY.md)（术语） |

> `.claude/skills/`（pypto-dev-constraints / pypto-whole-net-hang-debug）与
> `develop/N1/`（脚本 + 0162 stable env SSOT）是运行工具/冻结环境，原地保留。

## 🔎 查什么去哪里

| 问题 | 路径 |
|------|------|
| 项目背景/目标？ | [`design/00-context-and-goals.md`](design/00-context-and-goals.md) |
| step3p5 模型本身（config + 层结构）？ | [`design/step3p5-model-architecture.md`](design/step3p5-model-architecture.md) |
| 整网怎么设计的？ | [`design/whole-net/`](design/whole-net/)（HLD + LLD） |
| vLLM 集成怎么设计的？ | [`design/vllm-pypto/`](design/vllm-pypto/)（HLD + LLD） |
| 进度 / 路线图？ | [`planning/roadmap.md`](planning/roadmap.md) |
| 此刻状态？ | [`STATUS.md`](STATUS.md) |
| 撞到 507018/507899/hang/编译报错怎么办？ | [`postmortems/`](postmortems/)（按 error signature 查索引） |
| 新机器怎么部署？ | [`deployment/`](deployment/) |
| 从零装 pypto 运行时环境（拉仓库→跑通）？ | [`.claude/skills/pypto-runtime-install/SKILL.md`](.claude/skills/pypto-runtime-install/SKILL.md) |
| 验收金标准？ | [`reference/canonical-test.md`](reference/canonical-test.md) |
| 术语看不懂？ | [`GLOSSARY.md`](GLOSSARY.md) |
| 每日进展历史？ | [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md) |
| 写 pypto kernel 的坑？ | `pypto-lib/docs/known-pypto-pitfalls.md`（sub-repo） |

## 涉及的仓库

| 仓库 | 角色 | 我们的 fork |
|------|------|------------|
| `pypto` | 编程框架（IR + codegen） | `csy0225/pypto` |
| `pypto-lib` | tensor kernel + step3p5 模型 | `csy0225/pypto-lib` |
| `pto-isa` | Tile-ISA 虚拟实现 | `csy0225/pto-isa` |
| `PTOAS` | 字节码 assembler | `csy0225/PTOAS` |
| `simpler` | PTO runtime（pypto submodule） | `csy0225/simpler` |
| vLLM stepcast fork | serving（集成目标） | 无 fork |

fork 都在 `stepfun/develop` 分支；pin snapshot 见 [`STATUS.md`](STATUS.md)。

## 🚀 快速起手（Phase 16 合规机，如 `gpu-a910x-0162`）

```bash
# 三件套激活（CANN 必须 beta.1，不是 GA）
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa

# 前端 smoke
cd <workspace>/pypto-lib && python -m models.step3p5._smoke_program_build   # 期望 rc=0

# 多卡 collective baseline
cd <workspace>/pypto/runtime
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1     # 期望 max|out-expected|=0
```

任一失败 → 查 [`postmortems/`](postmortems/) + [`deployment/machine-recovery.md`](deployment/machine-recovery.md)。

## 更新协议

见 [`CLAUDE.md`](CLAUDE.md)「同步协议」。要点：phase 状态改 `planning/` + `STATUS.md`；
每日流水追加 `archive/milestones-2026-Q2.md`；新 blocker 进 `blockers.md`，解决后转
`postmortems/`；设计变更改 `design/`。**代码 reference 写 sub-repo `docs/`，本仓只放项目级跟踪/设计/复盘。**
