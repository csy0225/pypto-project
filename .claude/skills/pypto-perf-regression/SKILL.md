---
name: pypto-perf-regression
description: >
  step3p5 性能优化专项的**回归 runbook**。做完任何一个 PERF-* 子任务
  （见 design/performance/）后，按本 skill 逐步回归：环境确认 → liveness 冒烟 →
  多步 decode 精度回归（≥95% vs vanilla）→ 性能 DFX 采集与对比 → 文档更新 →
  commit/push。路径全部走环境变量（镜像内开发优先），不硬编码任何主机绝对路径。
  触发：改动 whole-net decode / MoE / attention / LM-head kernel 或调度，
  或认领/完成 design/performance/task-tracking.md 里的任一子任务。
---

# step3p5 性能优化回归 runbook

> **单一入口**：设计在 [`design/performance/`](../../design/performance/README.md)，状态在
> [`task-tracking.md`](../../design/performance/task-tracking.md)，性能数据在
> `$WS/pypto-lib/docs/step3p5/perf-baseline.md`。本 skill 只讲"改完之后按什么顺序回归 + 更哪些文档"。
>
> **三条铁律**：① 路径**全走环境变量**，勿硬编码主机路径；② 精度**只认多步 decode**（多步已含首
> token；单 token `argmax==303` 只是冒烟/liveness）；③ 文档**只落三处**（task-tracking / perf-baseline
> / 必要时 canonical+STATUS），不新建散落文件。

---

## 环境约定（镜像内开发优先；路径 = 环境变量）

后续开发在**镜像内**进行（构建见 `deployment/docker/`）。镜像已 baked 好：

- `WS=/workspace` —— 5 个代码仓：`$WS/pypto-lib`、`$WS/pypto`、`$WS/pto-isa`、`$WS/ptoas-bin`、`$WS/pypto/runtime`。
- CANN env + `PTO_ISA_ROOT=$WS/pto-isa` + `PTOAS_ROOT=$WS/ptoas-bin` + `PTO2_RING_*` 由 **ENTRYPOINT / `/etc/profile.d`** 自动 source —— **镜像内无需手动三件套**。
- `PYTHON=/usr/local/python3.11.14/bin/python3`。

本 skill 用到的环境变量（**镜像给默认；用户按自己环境覆盖**）：

| 变量 | 镜像默认 | 用途 |
|------|----------|------|
| `WS` | `/workspace` | 代码仓根 |
| `PYPTO_PROJECT` | 跟踪仓 checkout 路径（用户设） | `design/performance/`、`reference/canonical-test.md` 所在（本仓） |
| `CKPT` | 用户设 | W8A8 checkpoint 目录 |
| `DEVICES` | `8,9,10,11,12,13,14,15` | pypto 用的卡（vanilla oracle 占 `0-7`） |
| `GITHUB_TOKEN` | 用户设 / CI secret | push 用 PAT（或用 `GITHUB_ENV` 指向含 token 的文件） |

> **裸机（非镜像）回退**：手动
> `source <CANN>/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa`。
> 其余步骤命令不变（都已用 `$WS`/`$CKPT`/`$DEVICES`）。

---

## Step 0 · 前置

1. **确认环境就绪**（镜像内应已由 entrypoint source 好）：
   ```bash
   : "${WS:?set WS (image default /workspace)}"
   python -c "import pypto, pypto.language" && echo "PTO_ISA_ROOT=$PTO_ISA_ROOT"
   # 裸机若报错：source <CANN>/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa
   ```
2. **pin substrate**：确认 5 仓 commit 与回归对象一致（`$PYPTO_PROJECT/STATUS.md` Pin Snapshot 最新行）。
   跨机器/跨分支只 `git pull` **不构成同一测试对象**（canonical §3.3.1）。
3. **清 stale pyc**（跑过 monkey-patch / 改过 config 后必做）：
   ```bash
   find "$WS/pypto-lib/models/step3p5" -name "*.py" -exec touch {} +
   ```
4. **确认改动范围**：对应 `task-tracking.md` 的哪个 `PERF-*`？把该子任务置 🟦。

---

## Step 1 · Liveness 冒烟（快，先过这关）

确认 whole-net 跑通 + 首 token 对 + 不 hang。命令模板见 [`reference/canonical-test.md §3`](../../reference/canonical-test.md)，
**把其中的主机路径按环境变量替换**（`$WS`、`$CKPT`、`--device "$DEVICES"`）。必须同时满足：

```text
process rc = 0
存在 [worker] RUN done
argmax = 303                # 首 token 冒烟（非精度准出）
```

- stall / deadlock 用隔离探针 `$WS/pypto-lib/tests/step3p5/_probe_barrier_scale.py`（PUSH + 三个 `PTO2_*` 超时）区分 slow vs deadlock。
- 冒烟挂 = 先修 liveness。**冒烟过 ≠ 精度过**，必须继续 Step 2。

---

## Step 2 · 多步 decode 精度回归（唯一精度准出）

**多步 decode 逐 token** teacher-forced 对比 live vanilla vLLM W8A8 oracle：

```bash
cd "$WS/pypto-lib"
bash tests/step3p5/ci/run_live_precision_ab.sh    # 详见 tests/step3p5/ci/LIVE_PRECISION_AB.md
```

- 口径：seed=6127 / **N=128** → **ALIGNED ≥ 95%**（baseline 124/128=96.9%，miss 均为 vanilla 自身 near-tie）。
- **只验第一个 token 不算数**——多步逐 token 才是精度准出。
- 结构/数值改动（B*、C*、D*、G1）**必须**过这关；纯采集/调度（A1、C3、F*）也要确认不回退。
- 需要 vanilla vLLM W8A8 oracle 在跑（占 cards `0-7`）；pypto 用 `$DEVICES`（默认 `8-15`）。

---

## Step 3 · 性能 DFX 采集 + 对比 baseline

1. 采四件套（对照 A1 baseline）：`l2_swimlane`、`dfx_outputs/pmu.csv`、`perf_hints.log`、
   `memory_after_AllocateMemoryAddr.txt`。经 `$WS/pypto-lib/tools/step3p5/whole_decode_holder.py` 的
   `--enable-l2-swimlane` / `--enable-pmu` / `enable_scope_stats` 开。
2. 记录**单步 decode wallclock** + 分层耗时 + 关键 kernel 的 cube/vec/mte 利用率。
3. 与 `$WS/pypto-lib/docs/step3p5/perf-baseline.md` 上一版对比：本次该快的地方是否变快、别处是否回退。
4. HBM：动了权重/KV 布局（B1/B3/D2）时记录 `/rank` 占用变化（如 D2 应 47.6GB→~24GB）。

---

## Step 4 · 文档更新（集中三处，勿散落）

> 路径相对 `$PYPTO_PROJECT`（本仓）或 `$WS/pypto-lib`。**只更新这几处**：

| 触发 | 落点 |
|------|------|
| 子任务状态/进度变化（每次都做） | `$PYPTO_PROJECT/design/performance/task-tracking.md`：改行状态/owner/最后更新 + 进度汇总计数 + 底部「更新日志」追加一行 |
| 本次性能数据（每次都做） | `$WS/pypto-lib/docs/step3p5/perf-baseline.md`：追加一行（改动 ID / 单步延迟 / 分层耗时 / HBM/rank / DFX 工件路径） |
| 设计/接口/shape 变了 | `$PYPTO_PROJECT/design/performance/02-detailed-design.md`（LLD，带 shape）+ 必要时 `01-system-design.md` |
| 验收标准变了 | `$PYPTO_PROJECT/reference/canonical-test.md` §2 + `CLAUDE.md` 铁律#7（保持一致） |
| 组件 pin 移动（任意 fork push） | `$PYPTO_PROJECT/STATUS.md` Pin Snapshot 最新行 + `archive/milestones-2026-Q2.md` |
| 子任务全部完成 / phase 状态变 | `$PYPTO_PROJECT/STATUS.md` Phase 表 + `planning/roadmap.md` |

**完成一个子任务 = task-tracking 状态置 ✅**（前提：Step 2 多步 L3 ≥95% 通过；只过冒烟不算完成）。

---

## Step 5 · Commit + Push（HTTP/1.1 硬要求）

```bash
cd "$PYPTO_PROJECT"
git add <files>
git commit -m "perf(step3p5): <PERF-ID 一句话>"
# token 优先取 $GITHUB_TOKEN；否则从 $GITHUB_ENV 指向的文件读
PAT="${GITHUB_TOKEN:-$(tr -d '\n\r' < "${GITHUB_ENV:?set GITHUB_TOKEN or GITHUB_ENV}")}"
git -c http.version=HTTP/1.1 -c http.postBuffer=104857600 push \
    "https://x-access-token:${PAT}@github.com/csy0225/pypto-project.git" main:main \
    2>&1 | sed -E "s|x-access-token:[^@]+@|x-access-token:***@|g"
unset PAT
```

- pypto-lib 代码改动同理推 `csy0225/pypto-lib`（对应分支），`cd "$WS/pypto-lib"`。
- 跨仓 push 后，**本仓 STATUS Pin Snapshot 同一 session 一起推**，别漂。
- 默认 HTTP/2 在部分网络 130s 静默超时——务必带 `-c http.version=HTTP/1.1`。

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

- 性能**设计**只进 `$PYPTO_PROJECT/design/performance/`；**数据**只进 `$WS/pypto-lib/docs/step3p5/perf-baseline.md`；
  **状态**只进 `design/performance/task-tracking.md`。
- 不在 session 日志 / 随手 md / 各子目录另起 perf 记录。
- kernel 编码坑写 `$WS/pypto-lib/docs/known-pypto-pitfalls.md`，dev workflow 坑写
  `$WS/pypto-lib/docs/dev-workflow-gotchas.md`（不写本仓）。

## 相关

- 设计索引：[`design/performance/README.md`](../../design/performance/README.md)
- 镜像构建：`$PYPTO_PROJECT/deployment/docker/`（`Dockerfile` 定义 `WS=/workspace` 等 baked env）
- 强开发约束：`.claude/skills/pypto-dev-constraints/`
- 整网 hang 排查：`.claude/skills/pypto-whole-net-hang-debug/`
- 金标准：[`reference/canonical-test.md`](../../reference/canonical-test.md)
