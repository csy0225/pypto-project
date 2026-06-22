# Claude Session Bootstrap

打开本仓库时 Claude session 自动加载本文件。所有项目级跟踪的入口、
工作约定、同步协议、铁律都在这里。

## 本仓库是什么

pypto step3p5 项目的**项目级跟踪仓**。详见 [`README.md`](README.md)。
五个代码仓（`pypto` / `pypto-lib` / `pto-isa` / `PTOAS` / `simpler`）在
别处，本仓只跟踪它们。

## 项目工作语言

中文。所有文档、commit message、Issue 描述用中文。**技术专有名词保留
英文**（pypto / simpler / `AllocateMemoryAddr` / barrier all_reduce /
monkey-patch / `pl.range` 等 API / 系统名）。code block 一律不译。

## 先看哪里

1. **当前状态**：[`STATUS.md`](STATUS.md)
2. **被什么卡住**：[`blockers.md`](blockers.md)
3. **当前阶段的任务**：[`phases/`](phases/)
4. **生产部署 spec**：[`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md)

---

## 阶段性进展同步协议 ★

任何 phase / sub-task / blocker / 跨仓 commit 发生时，**必须** 把变化
同步到本仓对应文档，然后 commit + push。下面是触发 → 落点 → 操作映射。

### 触发 → 文件 映射

| 触发 | 改哪个文件 | 改什么 |
|------|----------|--------|
| sub-task 完成 | `phases/NN-*.md` | 把对应任务在 "任务" 表标 ✅；如果整个 sub-task 段完了改 Status 段 |
| Phase 入场 / 准出 | `STATUS.md` | "Phase 2 sub-phases" 表的对应行 + "Phase tracker" 表 |
| Phase 完成 | `phases/README.md` + `STATUS.md` + 追加到 `archive/milestones-2026-Q2.md` + 该 phase doc 的 Status 段 | 4 处一起更 |
| 新 blocker 发现 | `blockers.md` + `STATUS.md` "硬 Blocker" 表 | 新建一节（按严重度归位）+ 表里加行 |
| Blocker 解决 | 从 `blockers.md` 删掉这一节 + 追加 post-mortem 到 `archive/milestones-2026-Q2.md` "Resolved blockers" + `STATUS.md` 表行删掉 | 3 处 |
| Session 末尾 milestone | `archive/milestones-2026-Q2.md` | 按降序追加一段 |
| 任意 sub-repo（pypto / pypto-lib / pto-isa / PTOAS / simpler）push HEAD 移动 | `STATUS.md` "Pin snapshot" + `archive/milestones-2026-Q2.md` "Pin snapshot 历史" | 改最新一行 + 加历史行 |
| 新 phase 启动 | 新建 `phases/NN-*.md` + 在 `phases/README.md` index 表加行 + `STATUS.md` "Phase tracker" 表加行 + `architecture/` 如有跨仓 design 变化 | 3-4 处 |
| 新发现需要记入项目历史的关键决策 | `archive/milestones-2026-Q2.md` 追加一段；如果决策涉及跨仓部署变化，同步更新 `deployment/` 对应 spec | 1-2 处 |
| 新 dev workflow 坑 / 新 pypto kernel 限制 | **不要写本仓**，写在 `pypto-lib/docs/dev-workflow-gotchas.md` 或 `pypto-lib/docs/known-pypto-pitfalls.md`（在 pypto-lib 仓） | 在 pypto-lib 而非本仓 |

### Commit + push 标准流程

每次同步都走这条路（已经验证可用，HTTP/1.1 是 0162 网络上的硬要求）：

```bash
# 1. 改 markdown（在本仓的文件）

# 2. commit（中文 message，简明扼要）
git add <changed files>
git commit -m "$(cat <<'EOF'
docs(<scope>): <一句话说改了什么>

<可选：1-3 句 body 解释 why / 链接到 phase doc / 引用 commit SHA>
EOF
)"

# 3. push via PAT（不落 .git/config）
PAT="$(tr -d '\n\r' < /data/chensiyu/secrets/github.env)"
git -c http.version=HTTP/1.1 -c http.postBuffer=104857600 push \
    "https://x-access-token:${PAT}@github.com/csy0225/pypto-project.git" \
    main:main 2>&1 | sed -E "s|x-access-token:[^@]+@|x-access-token:***@|g"
unset PAT
```

如果 0162 远端 push（dev host 上的本仓 clone），步骤同上但用
`scp` 把 secret 临时传过去 + `shred -u` 删除。见
`deployment/machine-recovery.md` "git push 走 HTTPS 130 秒超时" 段。

### 跨仓 push 后必做的同步

当 `csy0225/<repo>` 任一仓 push 之后（如把 barrier all_reduce 修复推到
pypto-lib stepfun/develop），**本仓 STATUS.md "Pin snapshot" 一行也要
同步推上来**。两次 push 同步进行：

```bash
# 1. push 代码到 sub-repo（在 sub-repo 工作目录）
git -C <pypto-lib> -c http.version=HTTP/1.1 push fork stepfun/develop:stepfun/develop

# 2. 更新本仓 STATUS.md 的 pin snapshot 行 + 追加 archive milestone
# 3. push 本仓
git push <pypto-project> main:main
```

**两个 push 不要分会话做**。失忆漂掉一个会让 pin snapshot 跟实际不
对应。

### 不要 push 的内容

不要把 sub-repo 的实现代码、tracker 文档、phase log 等放进任意 sub-
repo（如 pypto-lib 的 `docs/step3p5/phases/`）。2026-06-22 这个仓
建立的原因就是修正这一点。**项目级跟踪在本仓，代码 reference 在
sub-repo `docs/`，模型代码在 sub-repo `models/`**。

---

## 铁律（每个 session 都适用）

下面这些都是过去 session 重犯过的错。再犯一次会浪费几个小时。

### 1. 单卡 ST/UT 必须保 TP=8 per-rank slice 宽度

写或跑 kernel-级 ST/UT 时用 `apply_perrank_patch()`，**不要用**
`apply_tp1_patch()`。per-rank helper 把 `TP_WORLD_SIZE`/`EP_WORLD_SIZE`
切到 1 让 codegen 消掉 collective，但保留 canonical TP=8 切片宽度
（8/12/1/1408/160/36 等）。Unslice helper（全宽）只适合 Phase 15 e2e；
chunk 跟 slice 走的 kernel（`sh_mlp` / `gate_matmul` 等）会爆。

详见：`pypto-lib/tests/step3p5/_perrank_setup.py` docstring；
`pypto-lib/docs/known-pypto-pitfalls.md` 反复引用。

### 2. Phase 16 三剑合璧版本绑定

任何生产多卡部署都必须**三件齐备**：

| 组件 | 必需版本 | 旧版本失败模式 |
|------|----------|---------------|
| Driver | 25.5.2 | `support_shmem_map_exbus=0`，IPC 507899 |
| Firmware | 7.8.0.7.220（chip flash，持久） | 同样的 cap 缺口 |
| CANN | 9.0.0-beta.1（NOT GA） | simpler init 507018 |

完整 spec：[`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md)。

### 3. 测试中 monkey-patch 模块全局后 .pyc stale

任何跑了 `apply_perrank_patch` / `apply_tp1_patch` / `cfg.X = Y` 的
测试都会把 patched 值序列化进 `__pycache__/*.pyc`。下次 fresh
`python -m ...` 会读到。再跑前必须：

```bash
find <pypto-lib>/models/step3p5 -name "*.py" -exec touch {} +
```

详见：`pypto-lib/docs/dev-workflow-gotchas.md` §1。

### 4. 三件套激活

`activate.sh` 只激活 venv。部署机上每个新 shell 都要三次 source：

```bash
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa
```

详见：`pypto-lib/docs/dev-workflow-gotchas.md` §2。

### 5. 部署机网络上 git push 必须用 HTTP/1.1

```bash
git -c http.version=HTTP/1.1 push ...
```

默认 HTTP/2 在 130 秒后静默超时。详见：
`pypto-lib/docs/dev-workflow-gotchas.md` §3。

---

## 本文件**不要**写什么

下面这些会让 CLAUDE.md 失焦，必须写去别处：

| 内容 | 应该去 |
|------|--------|
| session-by-session milestone | `archive/milestones-2026-Q2.md` |
| Phase 任务清单 / 设计 | `phases/NN-*.md` |
| open issue | `blockers.md` |
| Pin snapshot 历史 | `archive/milestones-2026-Q2.md` |
| 部署 runbook | `deployment/` |
| 跨仓架构图 / 映射 | `architecture/` |
| pypto kernel 编码坑 / 调试技巧 | `pypto-lib/docs/`（sub-repo） |
| 模型代码 | `pypto-lib/models/`（sub-repo） |

CLAUDE.md 只放：**项目工作语言 + 入口指针 + 同步协议 + 铁律 + 边界规则**。
其他都是过路客。
