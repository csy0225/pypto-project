# Claude Session Bootstrap

打开本仓库时 Claude session 自动加载本文件。项目级跟踪的入口、工作约定、
同步协议、铁律都在这里。

## 本仓库是什么

pypto step3p5 项目的**项目级跟踪 + 设计 + 部署 + 复盘仓**。详见
[`README.md`](README.md)。五个代码仓（`pypto`/`pypto-lib`/`pto-isa`/`PTOAS`/
`simpler`）+ vLLM fork 在别处，本仓只跟踪/设计它们。

## 项目工作语言

中文。文档/commit/Issue 用中文，**技术专有名词保留英文**（pypto / simpler /
`AllocateMemoryAddr` / `tp_all_reduce` / monkey-patch / `pl.range` 等 API/系统名）。
code block 一律不译。

## 先看哪里

1. **给别人讲项目**：[`design/00-context-and-goals.md`](design/00-context-and-goals.md) → 两个 `design/*/01-system-design.md` → [`planning/roadmap.md`](planning/roadmap.md)
2. **当前状态**：[`STATUS.md`](STATUS.md)
3. **被什么卡住**：[`blockers.md`](blockers.md)
4. **接着干什么**：[`planning/handoff.md`](planning/handoff.md)
5. **撞到已知坑**：[`postmortems/`](postmortems/)
6. **部署**：[`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md)

## 仓库分区职责（写东西前先对号入座）

| 分区 | 放什么 | 不放什么 |
|------|--------|----------|
| `design/` | HLD/LLD 设计（context→系统设计→详细设计） | 状态、日志、blocker |
| `planning/` | roadmap（规划）、handoff（接力）、活跃 phase | 每日流水（去 archive） |
| `postmortems/` | 已解/在查工程问题的五段复盘 | 活跃未分析的 blocker（去 blockers.md） |
| `deployment/` | 纯生产 runbook | troubleshooting 复盘（去 postmortems） |
| `reference/` | canonical 测试、4+1 视图、编程 API、约束 | 跨仓设计（去 design） |
| `archive/` | session 日志、原型摘要、已完成 phase、交付 | 当前状态 |
| 根 | STATUS / blockers / GLOSSARY / README / CLAUDE | 其他都进分区 |

---

## 同步协议 ★（触发 → 落点）

| 触发 | 改哪个文件 |
|------|-----------|
| phase 状态变化 | [`planning/phases/NN-*.md`](planning/phases/) 的 Status 段 + [`planning/roadmap.md`](planning/roadmap.md) 表 + [`STATUS.md`](STATUS.md) Phase 表 |
| session 末尾 milestone | 追加到 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)（**每日流水 SSOT，不写 STATUS/roadmap**） |
| 新 blocker 发现 | [`blockers.md`](blockers.md) + [`STATUS.md`](STATUS.md) blocker 摘要 |
| blocker 解决 | 从 [`blockers.md`](blockers.md) 删掉 → 新建/更新 [`postmortems/NN-*.md`](postmortems/)（五段模板 [`postmortems/TEMPLATE.md`](postmortems/TEMPLATE.md)）+ 更 STATUS 摘要 |
| 设计变更（架构/接口/数据流） | 对应 [`design/`](design/) 的 HLD 或 LLD |
| 组件 pin 移动（任意 fork push） | [`STATUS.md`](STATUS.md) Pin Snapshot（最新行）+ [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md) pin 历史 |
| 部署/版本变化 | [`deployment/`](deployment/) 对应 spec |
| 新 dev workflow 坑 / kernel 限制 | **写 sub-repo** `pypto-lib/docs/dev-workflow-gotchas.md` 或 `known-pypto-pitfalls.md`，**不写本仓** |

### Commit + push（HTTP/1.1 是 0162 网络硬要求）

```bash
git add <files>
git commit -m "docs(<scope>): <一句话>"
PAT="$(tr -d '\n\r' < /data/chensiyu/secrets/github.env)"
git -c http.version=HTTP/1.1 -c http.postBuffer=104857600 push \
    "https://x-access-token:${PAT}@github.com/csy0225/pypto-project.git" main:main \
    2>&1 | sed -E "s|x-access-token:[^@]+@|x-access-token:***@|g"
unset PAT
```

跨仓 push（如把修复推到 pypto-lib）后，**本仓 STATUS Pin Snapshot 同一 session 一起推**，别漂。

---

## 铁律（每个 session 都适用）

1. **单卡 ST/UT 保 TP=8 per-rank slice 宽度**：用 `apply_perrank_patch()`，不用 `apply_tp1_patch()`（unslice 只适合 Phase 15 e2e，chunk-follow-slice 的 kernel 会爆）。
2. **Phase 16 三剑合璧**：多卡部署必须 driver 25.5.2 + firmware 7.8.0.7.220 + CANN 9.0.0-beta.1（NOT GA）。见 [`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md) + [`postmortems/01-multirank-ipc-507899-507018.md`](postmortems/01-multirank-ipc-507899-507018.md)。
3. **monkey-patch 后 .pyc stale**：跑过 `apply_perrank_patch`/`cfg.X=Y` 后，下次 fresh run 前 `find <pypto-lib>/models/step3p5 -name "*.py" -exec touch {} +`。
4. **三件套激活**：每个新 shell `source CANN/set_env.sh` + `source activate.sh` + `export PTO_ISA_ROOT=...`。
5. **git push 用 HTTP/1.1**（默认 HTTP/2 在 130s 静默超时）。
6. **生产整网只允许单个 `@pl.program`**（多 program 有 co-prepare 死锁墙，见 [`postmortems/08-multiprogram-coprepare-deadlock.md`](postmortems/08-multiprogram-coprepare-deadlock.md)）；**native W8A8 不回退 BF16-dequant**。
7. **验收以 canonical 为准**：[`reference/canonical-test.md`](reference/canonical-test.md)（P42 → token 6127 → argmax 303）。
8. **强开发约束先读 skill**：`.claude/skills/pypto-dev-constraints/`；整网 hang 排查：`.claude/skills/pypto-whole-net-hang-debug/`。
