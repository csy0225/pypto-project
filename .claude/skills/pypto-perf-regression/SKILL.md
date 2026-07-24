---
name: pypto-perf-regression
description: >
  step3p5 性能优化专项的**回归 runbook**。做完任何一个 PERF-* 子任务
  （见 design/performance/）后，按本 skill 逐步回归：环境/pin → liveness 冒烟 →
  多步 decode 精度回归（≥95% vs vanilla）→ 性能 DFX 采集与对比 → 文档更新 →
  commit/push。目的是把"每次改完性能后要做的动作"固化到一处，避免散落。
  触发：改动 whole-net decode / MoE / attention / LM-head kernel 或调度，
  或认领/完成 design/performance/task-tracking.md 里的任一子任务。
---

# step3p5 性能优化回归 runbook

> **单一入口**：性能优化的设计在 [`design/performance/`](../../design/performance/README.md)，
> 状态在 [`design/performance/task-tracking.md`](../../design/performance/task-tracking.md)，
> 性能数据在 `pypto-lib/docs/step3p5/perf-baseline.md`。**本 skill 只讲"改完之后按什么顺序回归 + 更哪些文档"**，
> 不重复设计细节。
>
> **两条铁律**（贯穿全程）：① **精度只认多步 decode**（多步已含首 token；单 token `argmax==303`
> 只是冒烟/liveness，不是精度准出）；② **文档只落三处**（task-tracking / perf-baseline / 必要时
> canonical+STATUS），不新建散落文件。

---

## Step 0 · 前置（每次 fresh shell 必做）

1. **三件套激活**（activate.sh 不 source CANN）：
   ```bash
   source /usr/local/Ascend/cann/set_env.sh
   source /data/chensiyu/hw_project/pypto/workspace/activate.sh
   export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa
   ```
2. **pin substrate**：确认 5 仓 commit 与本次回归对象一致（`STATUS.md` Pin Snapshot 最新行）。
   跨机器/跨分支只 `git pull` **不构成同一测试对象**（见 canonical §3.3.1）。
3. **清 stale pyc**（跑过 monkey-patch / 改过 config 后必做）：
   ```bash
   find /data/chensiyu/hw_project/pypto/workspace/pypto-lib/models/step3p5 -name "*.py" -exec touch {} +
   ```
4. **确认改动范围**：本次回归对应 `task-tracking.md` 的哪个 `PERF-*`？把该子任务状态置 🟦。

---

## Step 1 · Liveness 冒烟（快，先过这关）

确认 whole-net 能跑通 + 首 token 对 + 不 hang。命令见 [`reference/canonical-test.md §3`](../../reference/canonical-test.md)。
必须同时满足：

```text
process rc = 0
存在 [worker] RUN done
argmax = 303                # 首 token 冒烟（非精度准出）
```

- stall / deadlock 用隔离探针 `tests/step3p5/_probe_barrier_scale.py`（PUSH + 三个 `PTO2_*` 超时）区分 slow vs deadlock。
- 冒烟挂 = 先修 liveness，别往下走。**冒烟过 ≠ 精度过**，必须继续 Step 2。

---

## Step 2 · 多步 decode 精度回归（唯一精度准出）

**多步 decode 逐 token** teacher-forced 对比 live vanilla vLLM W8A8 oracle：

```bash
# 驱动脚本（stepfun/develop）
cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
bash tests/step3p5/ci/run_live_precision_ab.sh    # 详见 tests/step3p5/ci/LIVE_PRECISION_AB.md
```

- 口径：seed=6127 / **N=128** → **ALIGNED ≥ 95%**（baseline 124/128=96.9%，miss 均为 vanilla 自身 near-tie）。
- **只验第一个 token 不算数**——多步逐 token 才是精度准出。
- 结构/数值改动（B*、C*、D*、G1）**必须**过这关；纯采集/调度（A1、C3、F*）也要确认不回退。
- 需要 live vanilla vLLM W8A8 oracle 在跑（cards 0-7）；pypto 用 free cards（8-15）。

---

## Step 3 · 性能 DFX 采集 + 对比 baseline

1. 采四件套（对照 A1 baseline）：`l2_swimlane`、`dfx_outputs/pmu.csv`、`perf_hints.log`、
   `memory_after_AllocateMemoryAddr.txt`。经 `whole_decode_holder.py` 的
   `--enable-l2-swimlane` / `--enable-pmu` / `enable_scope_stats` 开。
2. 记录**单步 decode wallclock** + 分层耗时 + 关键 kernel 的 cube/vec/mte 利用率。
3. 与 `pypto-lib/docs/step3p5/perf-baseline.md` 的上一版对比：**本次改动该快的地方是否变快、别处是否回退**。
4. HBM：若动了权重/KV 布局（B1/B3/D2），记录 `/rank` 占用变化（如 D2 应 47.6GB→~24GB）。

---

## Step 4 · 文档更新（集中三处，勿散落）

> **只更新这几处**，不要新建零散文档：

| 触发 | 落点 |
|------|------|
| 子任务状态/进度变化（每次都做） | [`design/performance/task-tracking.md`](../../design/performance/task-tracking.md)：改该行状态/owner/最后更新 + 进度汇总计数 + 底部「更新日志」追加一行（做了什么 / commit / 验证结果） |
| 本次性能数据（每次都做） | `pypto-lib/docs/step3p5/perf-baseline.md`：追加一行（改动 ID / 单步延迟 / 分层耗时 / HBM/rank / DFX 工件路径） |
| 设计/接口/shape 变了 | [`design/performance/02-detailed-design.md`](../../design/performance/02-detailed-design.md)（LLD，带 shape）+ 必要时 [`01-system-design.md`](../../design/performance/01-system-design.md) |
| 验收标准变了 | [`reference/canonical-test.md`](../../reference/canonical-test.md) §2 + `CLAUDE.md` 铁律#7（保持一致，别两处标准打架） |
| 组件 pin 移动（任意 fork push） | `STATUS.md` Pin Snapshot 最新行 + `archive/milestones-2026-Q2.md` pin 历史 |
| 子任务全部完成 / phase 状态变 | `STATUS.md` Phase 表 + `planning/roadmap.md` |

**完成一个子任务 = task-tracking 状态置 ✅**（前提：Step 2 多步 L3 ≥95% 通过；只过冒烟不算完成）。

---

## Step 5 · Commit + Push（HTTP/1.1 硬要求）

```bash
git add <files>
git commit -m "perf(step3p5): <PERF-ID 一句话>"
PAT="$(tr -d '\n\r' < /data/chensiyu/secrets/github.env)"
git -c http.version=HTTP/1.1 -c http.postBuffer=104857600 push \
    "https://x-access-token:${PAT}@github.com/csy0225/pypto-project.git" main:main \
    2>&1 | sed -E "s|x-access-token:[^@]+@|x-access-token:***@|g"
unset PAT
```

- pypto-lib 代码改动同理推 `csy0225/pypto-lib`（对应分支）。
- 跨仓 push 后，**本仓 STATUS Pin Snapshot 同一 session 一起推**，别漂。
- 默认 HTTP/2 在 0162 网络 130s 静默超时——务必带 `-c http.version=HTTP/1.1`。

---

## 回归判定表

| 检查 | PASS 条件 | 工具 |
|------|-----------|------|
| liveness | rc=0 + `RUN done` + 无 507018/stall | canonical §3 + `_probe_barrier_scale.py` |
| 首 token 冒烟 | `argmax==303` | canonical §2 |
| **精度（准出）** | **多步 N=128 ALIGNED ≥95% vs vanilla** | `run_live_precision_ab.sh` |
| 性能 | 目标项变快 + 无别处回退 | DFX 对比 perf-baseline.md |
| 文档 | task-tracking + perf-baseline 已更 | 人工 |

**全绿才算子任务完成**。任何一项红 → 子任务留 🟦/⛔，在 task-tracking「阻塞」列写原因。

---

## 避免散落原则

- 性能相关**设计**只进 `design/performance/`；**数据**只进 `pypto-lib/docs/step3p5/perf-baseline.md`；
  **状态**只进 `design/performance/task-tracking.md`。
- 不在 session 日志 / 随手 md / 各子目录另起 perf 记录。
- kernel 编码坑写 `pypto-lib/docs/known-pypto-pitfalls.md`，dev workflow 坑写
  `pypto-lib/docs/dev-workflow-gotchas.md`（**不写本仓**）——与本 skill 分工不重叠。

## 相关

- 设计索引：[`design/performance/README.md`](../../design/performance/README.md)
- 强开发约束：`.claude/skills/pypto-dev-constraints/`
- 整网 hang 排查：`.claude/skills/pypto-whole-net-hang-debug/`
- 金标准：[`reference/canonical-test.md`](../../reference/canonical-test.md)
