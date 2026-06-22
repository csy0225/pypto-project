# Claude Session Bootstrap

打开本仓库时 Claude session 自动加载本文件。**保持简短** —— 不属于
铁律 / 路由指针的内容应该写在
[`STATUS.md`](STATUS.md) / [`blockers.md`](blockers.md) 或 phase doc 里。

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

完整 spec + 失败分析：[`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md)。

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

## 这个文件**不要**写什么

- session-by-session milestone → `archive/milestones-2026-Q2.md`
- Phase 任务清单 → `phases/NN-*.md`
- open issue → `blockers.md`
- Pin snapshot 历史 → `archive/milestones-2026-Q2.md`
- 部署 runbook → `deployment/`

如果要写超过 50 行了，写去别处然后从这里 link。
