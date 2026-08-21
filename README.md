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

## 📁 仓库怎么组织（分层规则 T0–T6）

**核心区分**：**确定落地的事实**（T1/T2）和**每次开发的过程**（T3/T4）物理分开，各有唯一落点和硬预算。

| 层 | 唯一落点 | 放什么 | 预算 |
|---|---|---|---|
| **T0 必读教训** | [`postmortems/LESSONS.md`](postmortems/LESSONS.md) | 触发式索引：你正要做 X → 先记住 Y → 出处。**开工前必读** | ≤150 行 |
| **T0′ 坑案例集** | [`postmortems/CASEBOOK.md`](postmortems/CASEBOOK.md) | 按**现象**查的单点坑档案（背景/现象/过程/处置）+ **仍在生效的绕路台账** | 每条 ≤14 行 |
| **T1 当前真相** | [`STATUS.md`](STATUS.md) | 只写**此刻为真**，每条带 sha/digest/门结论 | ≤130 行 |
| **T2 落地台账** | [`progress/landed.md`](progress/landed.md) | 已发布镜像 + 已合入源码 + 各自过了哪个门 + NO-GO 台账 | 每条 ≤3 行 |
| **T3 开发流水** | [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md) | 每 session 一条 + 完整 pin 时间线 | 每条 ≤40 行 |
| **T4 未决** | [`blockers.md`](blockers.md) | 只放 **open**；症状/根因/解除条件/链接 | 每条 ≤25 行 |
| **T5 证据** | [`benchmark/`](benchmark/)、0162 campaign | 长报告、原始数字、campaign 路径。**不回灌 T1–T4** | 不限 |
| **T6 接力** | [`planning/handoff.md`](planning/handoff.md) | 纯指针 + "现在接什么" ≤3 条 | ≤90 行 |

**判据**：一条内容含"我这次发现 / 曾写 / 已撤回 / 下一位应该" ⇒ 属 **T3 或 T4**，**不属于 T1**。
一条内容是 sha / digest / 门结论 ⇒ 属 **T1 或 T2**。

其余分区（与分层正交）：

| 分区 | 放什么 |
|------|--------|
| [`design/`](design/) | **软件工程设计**：context + 两子系统的 系统设计(HLD) + 详细设计(LLD) |
| [`planning/`](planning/) | roadmap + 活跃 phase（+ T6 handoff） |
| [`postmortems/`](postmortems/) | **工程专项复盘**（16 篇，五段：背景/现象/根因/解决/弯路/避免）+ T0 索引 + 坑案例集 |
| [`deployment/`](deployment/) | **生产部署 runbook**：三剑合璧 / 机器恢复 / 版本矩阵 |
| [`reference/`](reference/) | canonical 测试、4+1 视图、编程 API、约束、执行主机契约 |
| [`archive/`](archive/) | 历史（追加式）：session 日志、原型摘要、已完成 phase、交付快照 |
| 根 | [`STATUS.md`](STATUS.md) · [`blockers.md`](blockers.md) · [`GLOSSARY.md`](GLOSSARY.md) · [`CLAUDE.md`](CLAUDE.md) |

> `.claude/skills/` 是运行工具；`develop/N1/` 是历史 N1 复现材料，不是当前 pin
> 或环境 SSOT。当前状态只看 [`STATUS.md`](STATUS.md)、远端 ref 和
> [`deployment/version-matrix.md`](deployment/version-matrix.md)。

## 🔎 查什么去哪里

| 问题 | 路径 |
|------|------|
| **开工前该记住什么坑？** | [`postmortems/LESSONS.md`](postmortems/LESSONS.md) ← **必读** |
| **已经撞上了某个现象，之前有人踩过吗？** | [`postmortems/CASEBOOK.md`](postmortems/CASEBOOK.md)（按现象查） |
| **这段看着冗余的绕路能删吗？** | [`postmortems/CASEBOOK.md`](postmortems/CASEBOOK.md) §C 仍在生效的绕路（🩹 = 承重） |
| **哪些是确定落地的？哪些方向已被否决？** | [`progress/landed.md`](progress/landed.md) |
| 项目背景/目标？ | [`design/00-context-and-goals.md`](design/00-context-and-goals.md) |
| step3p5 模型本身（config + 层结构）？ | [`design/step3p5-model-architecture.md`](design/step3p5-model-architecture.md) |
| 整网怎么设计的？ | [`design/whole-net/`](design/whole-net/)（HLD + LLD） |
| vLLM 集成怎么设计的？ | [`design/vllm-pypto/`](design/vllm-pypto/)（HLD + LLD） |
| 进度 / 路线图？ | [`planning/roadmap.md`](planning/roadmap.md) |
| 此刻状态？ | [`STATUS.md`](STATUS.md) |
| 现在该接什么活？ | [`planning/handoff.md`](planning/handoff.md) |
| 撞到 507018/507899/hang/编译报错怎么办？ | [`postmortems/`](postmortems/)（按 error signature 查索引） |
| 新机器怎么部署？ | [`deployment/`](deployment/) |
| 从零装 pypto 运行时环境（拉仓库→跑通）？ | [`.claude/skills/pypto-runtime-install/SKILL.md`](.claude/skills/pypto-runtime-install/SKILL.md) |
| 验收金标准？ | [`reference/canonical-test.md`](reference/canonical-test.md) |
| 术语看不懂？ | [`GLOSSARY.md`](GLOSSARY.md) |
| 每日进展历史 / pin 时间线？ | [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md) |
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
