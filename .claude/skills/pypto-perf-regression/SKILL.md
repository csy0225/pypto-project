---
name: pypto-perf-regression
description: >
  step3p5 性能优化专项的**立项门 + 回归 runbook**。做任何一个 PERF-* 子任务
  （见 design/performance/）时，按本 skill 逐步走：先算 ROI 天花板 vs A/B/A 检测地板
  决定是否立项 → 无卡 codegen 门筛候选 → 环境确认 → liveness 冒烟 →
  精度（byte-exact sha256 或多步 decode ≥95% vs vanilla）→ 三臂 A/B/A 与 DFX 采集对比 →
  文档更新 → commit/push。路径全部走环境变量（镜像内开发优先），不硬编码任何主机绝对路径。
  触发：改动 whole-net decode / MoE / attention / LM-head kernel 或调度，
  或认领/完成 design/performance/task-tracking.md 里的任一子任务。
---

# step3p5 性能优化立项门 + 回归 runbook

> **单一入口**：设计在 [`design/performance/`](../../design/performance/README.md)，状态在
> [`task-tracking.md`](../../design/performance/task-tracking.md)，性能数据在
> [`benchmark/`](../../benchmark/)（按 `YYYY-MM-DD-<主题>.md` 命名）。本 skill 讲
> "改之前怎么判断值不值得做 + 改完之后按什么顺序回归 + 更哪些文档"。
>
> **四条铁律**：① **天花板 < 地板就别写**（Step 0.5）；② 路径**全走环境变量**，勿硬编码主机路径；
> ③ 精度认 **byte-exact sha256 或多步 decode**（单 token `argmax==303` 只是冒烟/liveness）；
> ④ 文档**只落三处**（task-tracking / benchmark / 必要时 canonical+STATUS），不新建散落文件。

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

## Step 0.5 · 立项门：先算 ROI 天花板，再决定要不要写代码 ★

**这一步是本 runbook 里最省时间的一步。** 一个 kernel 就算优化到 0，也只能拿回它在
关键路径上的份额；如果那份额小于 A/B/A 的检测地板，**这个改动在方法上就不可能被证明有效**
——写了也只会得到「统计不可区分」。

```text
ROI 天花板 = 目标在 LOW-WAIT rank 关键路径上的 compute + stall
             / 该 rank makespan
             x 该工作点实测 ITL
             x (整网层数 / swimlane 采集的层数)

检测地板   = max(absolute_floor,
                parent_half_range,      # = |A1_p50 - A2_p50| / 2
                ratio_floor x parent_center)
```

**天花板 < 地板 → 不立项**（或者只能并进 bundle，见下）。实测参考值：0162 / ctx=64k
下 `parent_half_range` 约为 bs=1 `0.634 ms`、bs=8 `2.637 ms`；也就是说**任何映射到整网
不足约 0.6 ms 的单 kernel 改动，都测不出来**。

推论——**不要逐个 kernel 试融合**。一个占关键路径 0.65% 的 kernel（映射整网
0.13~0.17 ms）永远过不了地板。正确做法是把同类改造（例如所有大 cube matmul 的
tile + pipeline）**打成一个 bundle 一次上 A/B/A**：合计基数够大，10% 就能过地板。

配套两条纪律：

- **地板要从数据算，不能拍**。它由 `A1`/`A2` 两个 parent 臂的离散度决定，所以
  A/B/A 必须真跑三臂；只跑 `parent -> candidate` 无法给出地板。
- **先用无卡 codegen 门筛掉编不过的候选**（~14 s，不占卡），再排设备门。
  见 `$WS/pypto-lib/docs/dev-workflow-gotchas.md` §6；UB 预算模型见
  `known-pypto-pitfalls.md` §8。

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

### 捷径：byte-exact 改动用 hidden sha256，不必起 oracle ★

如果改动是**语义保持**的（搬移缩放位置、改 tiling、换 task 边界、改调度顺序，
但不改数值顺序），先看 A/B/A 三臂的 hidden payload sha256：

```text
三臂 sha256 完全一致  ->  byte-exact  ->  精度准出成立，跳过 oracle
任一臂不同           ->  非 byte-exact ->  必须跑上面的 N=128 多步门
```

byte-exact 比 oracle **更硬**（逐字节 vs 95% 阈值）且**更便宜**（不占 cards 0-7、
不用拉起 vanilla）。A/B/A 已经跑了三臂，sha256 是免费副产品。

⚠ 但要检查的是 **hidden payload 的 sha256**，不是 tail token——token 相同不代表
hidden 相同（near-tie 会掩盖差异）。同时确认 `finite=true` 与 `tp_spread_max=0`。

---

## Step 3 · 性能 DFX 采集 + 对比 baseline

1. 采四件套（对照 A1 baseline）：`l2_swimlane`、`dfx_outputs/pmu.csv`、`perf_hints.log`、
   `memory_after_AllocateMemoryAddr.txt`。经 `$WS/pypto-lib/tools/step3p5/whole_decode_holder.py` 的
   `--enable-l2-swimlane` / `--enable-pmu` / `enable_scope_stats` 开。
2. 记录**单步 decode wallclock** + 分层耗时 + 关键 kernel 的 cube/vec/mte 利用率。
3. 与 `$PYPTO_PROJECT/benchmark/` 里上一份同工作点的 benchmark 对比：本次该快的地方是否变快、别处是否回退。
4. HBM：动了权重/KV 布局（B1/B3/D2）时记录 `/rank` 占用变化（如 D2 应 47.6GB→~24GB）。

### 读 swimlane 的三条纪律（踩过就知道多贵）★

1. **不要从任意 rank 读 makespan。** 同一次 8 卡采集实测跨 rank
   `2.210 ms ~ 555.892 ms`；长 rank 的时间绝大部分是 kernel 内自旋在吸收 rank skew，
   会被记成 kernel compute。取 **LOW-WAIT rank**（makespan 最短那个）做分析，
   并交叉看其余 rank 确认没有把 skew 当算术。
2. **payload 对不上的 span 一定是等待，不是搬运。** 反例：单个 `tp_all_reduce`
   观测到 `35,530 us` 而 payload 只有 128 KB——不可能是数据移动。想动 collective
   前先要 step 级插桩证据（notify / wait 分离），否则「降 ring step」是在优化 barrier
   该等的最慢 rank。
3. **插桩 run 的绝对时间不能外推。** 5 层 DFX run 的 `p50` 含插桩放大，
   **不可当干净延迟、不可乘层数比反推整网**；它只能用来算**占比**。
   绝对延迟只认非插桩的 ITL harness。

---

## Step 4 · 文档更新（集中三处，勿散落）

> 路径相对 `$PYPTO_PROJECT`（本仓）或 `$WS/pypto-lib`。**只更新这几处**：

| 触发 | 落点 |
|------|------|
| 子任务状态/进度变化（每次都做） | `$PYPTO_PROJECT/design/performance/task-tracking.md`：改行状态/owner/最后更新 + 进度汇总计数 + 底部「更新日志」追加一行 |
| 本次性能数据（每次都做） | `$PYPTO_PROJECT/benchmark/YYYY-MM-DD-<主题>.md` 新建一份：发布 pin + 改动 + 各工作点三臂 ITL（含 parent_center / gain / gain_floor / 裁决）+ 精度口径 + swimlane 固定路径 + NO-GO 清单 + 权威证据绝对路径 |
| 设计/接口/shape 变了 | `$PYPTO_PROJECT/design/performance/02-detailed-design.md`（LLD，带 shape）+ 必要时 `01-system-design.md` |
| 验收标准变了 | `$PYPTO_PROJECT/reference/canonical-test.md` §2 + `CLAUDE.md` 铁律#7（保持一致） |
| 组件 pin 移动（任意 fork push） | `$PYPTO_PROJECT/STATUS.md` Pin Snapshot 最新行 + `archive/milestones-2026-Q2.md` |
| 子任务全部完成 / phase 状态变 | `$PYPTO_PROJECT/STATUS.md` Phase 表 + `planning/roadmap.md` |

**完成一个子任务 = task-tracking 状态置 ✅**（前提：Step 2 精度过关——byte-exact sha256 或多步 N=128 ≥95%；只过冒烟不算完成）。

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
| **立项** | ROI 天花板 > 检测地板（Step 0.5） | swimlane 占比 × ITL vs `parent_half_range` |
| codegen | 候选能编过（先筛，不占卡） | `compile_gate.sh`（~14s @ NB=512） |
| liveness | rc=0 + `RUN done` + 无 507018/stall | canonical §3 + `_probe_barrier_scale.py` |
| 首 token 冒烟 | `argmax==303` | canonical §2 |
| **精度（准出）** | 三臂 hidden sha256 一致（byte-exact）**或** 多步 N=128 ALIGNED ≥95% | A/B/A 产物 / `run_live_precision_ab.sh` |
| 性能 | 收益 > 检测地板 + 无别处回退 | 三臂 A/B/A + DFX 对比 |
| 文档 | task-tracking + 性能数据已更 | 人工 |

**全绿才算子任务完成**。任何一项红 → 子任务留 🟦/⛔，在 task-tracking「阻塞」列写原因。

---

## 避免散落原则

- 性能**设计**只进 `$PYPTO_PROJECT/design/performance/`；**数据**只进 `$PYPTO_PROJECT/benchmark/`；
  **状态**只进 `design/performance/task-tracking.md`。
- 不在 session 日志 / 随手 md / 各子目录另起 perf 记录。
- kernel 编码坑写 `$WS/pypto-lib/docs/known-pypto-pitfalls.md`，dev workflow 坑写
  `$WS/pypto-lib/docs/dev-workflow-gotchas.md`（不写本仓）。

## 相关

- 设计索引：[`design/performance/README.md`](../../design/performance/README.md)
- **UB 预算模型 / 融合与 pipeline 的硬约束**：`$WS/pypto-lib/docs/known-pypto-pitfalls.md` §8
  （per-kernel-per-core 预算、融合的 c2v pipe 附加成本、`pl.pipeline` 用 buffer 换 overlap、
  「拟合定律再预测数字」的方法）；§7 有 `pl.range` 展开的**反例**（展开本身不涨 UB，
  真正的区分器是有没有 Vec 值跨迭代存活）
- **无卡 codegen 门 / 读 pass dump**：`$WS/pypto-lib/docs/dev-workflow-gotchas.md` §6–§7
- 镜像构建：`$PYPTO_PROJECT/deployment/docker/`（`Dockerfile` 定义 `WS=/workspace` 等 baked env）
- 强开发约束：`.claude/skills/pypto-dev-constraints/`
- attention 专项优化方法：`.claude/skills/pypto-attention-performance/`
- 整网 hang 排查：`.claude/skills/pypto-whole-net-hang-debug/`
- 执行主机契约（0162 唯一执行主机 / 卡与锁 / 并发计时约束）：
  [`reference/execution-host-contract.md`](../../reference/execution-host-contract.md)
- 金标准：[`reference/canonical-test.md`](../../reference/canonical-test.md)
