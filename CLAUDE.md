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

0. **⚠ 开工前必读**：[`postmortems/LESSONS.md`](postmortems/LESSONS.md) —— 触发式教训索引（见铁律 §0）
1. **给别人讲项目**：[`design/00-context-and-goals.md`](design/00-context-and-goals.md) → 两个 `design/*/01-system-design.md` → [`planning/roadmap.md`](planning/roadmap.md)
2. **当前状态**：[`STATUS.md`](STATUS.md)
3. **哪些是确定落地的 / 哪些方向已被否决**：[`progress/landed.md`](progress/landed.md)
4. **被什么卡住**：[`blockers.md`](blockers.md)
5. **接着干什么**：[`planning/handoff.md`](planning/handoff.md)
6. **撞到已知坑**：先按**现象**查 [`postmortems/CASEBOOK.md`](postmortems/CASEBOOK.md)，再进 [`postmortems/`](postmortems/README.md) 全文
7. **部署**：[`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md)

## 分层规则 ★（写东西前先对号入座）

**核心区分**：**确定落地的事实**（T1/T2）与**每次开发的过程**（T3/T4）物理分开，
各有唯一落点和硬预算。超预算 = 有东西该下沉，不是该放宽预算。

| 层 | 唯一落点 | 唯一规则 | 预算 |
|---|---|---|---|
| **T0 必读教训** | [`postmortems/LESSONS.md`](postmortems/LESSONS.md) | 触发式：`你正要做 X → 先记住 Y → 出处`。只收**已确立**结论 | ≤150 行 |
| **T0′ 坑案例集** | [`postmortems/CASEBOOK.md`](postmortems/CASEBOOK.md) | 按**现象**查：背景 / 现象 / 过程 / 处置（✅ 已修 · 🩹 已绕开 · ⏸ 未解）。🩹⏸ 必写「移除代价 / 复发条件」 | 每条 ≤14 行 |
| **T1 当前真相** | [`STATUS.md`](STATUS.md) | 只写**此刻为真**，每条带 sha/digest/门结论。**禁止**历史快照、禁止"曾写…已撤回"、禁止 campaign 叙事 | ≤130 行 |
| **T2 落地台账** | [`progress/landed.md`](progress/landed.md) | 追加式窄格：日期 / 一句话 / pin / 镜像 digest / 门证据 + NO-GO 台账。区分 **SRC**（source-overlay GO）与 **IMG**（immutable-image released） | 每条 ≤3 行 |
| **T3 开发流水** | [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md) | 每 session 一条，细节指向 `benchmark/` 或 0162。也是**完整 pin 时间线**的家 | 每条 ≤40 行 |
| **T4 未决** | [`blockers.md`](blockers.md) | 只放 **open**；症状/根因一句/解除条件/链接。**定案或解决即转 postmortem（负结论也一样）** | 每条 ≤25 行<br>（多缺口 🔴 ≤35） |
| **T5 证据** | [`benchmark/`](benchmark/)、0162 campaign 目录 | 长报告、原始数字、campaign 路径。**不回灌 T1–T4** | 不限 |
| **T6 接力** | [`planning/handoff.md`](planning/handoff.md) | 纯指针 + "现在接什么" ≤3 条 | ≤90 行 |

**判据**：一条内容含"我这次发现 / 曾写 / 已撤回 / 下一位应该" ⇒ 属 **T3 或 T4**，
**不属于 T1**。一条内容是 sha / digest / 门结论 ⇒ 属 **T1 或 T2**。

分区职责（与分层正交）：

| 分区 | 放什么 | 不放什么 |
|------|--------|----------|
| `design/` | HLD/LLD 设计（context→系统设计→详细设计） | 状态、日志、blocker |
| `planning/` | roadmap（规划）、handoff（接力）、活跃 phase | 每日流水（去 archive） |
| `postmortems/` | 已解/已定案问题的五段复盘 + T0 教训索引 + T0′ 坑案例集 | 活跃未分析的 blocker（去 blockers.md） |
| `deployment/` | 纯生产 runbook | troubleshooting 复盘（去 postmortems） |
| `reference/` | canonical 测试、4+1 视图、编程 API、约束、执行主机契约 | 跨仓设计（去 design） |
| `archive/` | session 日志、原型摘要、已完成 phase、交付 | 当前状态 |
| 根 | STATUS / blockers / GLOSSARY / README / CLAUDE | 其他都进分区 |

---

## 同步协议 ★（触发 → 落点）

| 触发 | 改哪个文件 |
|------|-----------|
| phase 状态变化 | [`planning/phases/NN-*.md`](planning/phases/) 的 Status 段 + [`planning/roadmap.md`](planning/roadmap.md) 表 + [`STATUS.md`](STATUS.md) Phase 表 |
| session 末尾 milestone | 追加到 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)（**每日流水 SSOT，不写 STATUS/roadmap**） |
| 新 blocker 发现 | [`blockers.md`](blockers.md)（≤25 行）+ [`STATUS.md`](STATUS.md) §8 摘要一行 |
| blocker **解决或定案**（含负结论） | 从 [`blockers.md`](blockers.md) 删掉 → 新建/更新 [`postmortems/NN-*.md`](postmortems/README.md)（五段模板 [`postmortems/TEMPLATE.md`](postmortems/TEMPLATE.md)）+ 在 `postmortems/README.md` 加一行 + 把可复用教训提炼进 [`postmortems/LESSONS.md`](postmortems/LESSONS.md) + 更 STATUS 摘要 |
| 一个方向被否决（NO-GO） | [`progress/landed.md`](progress/landed.md)「已否决，不要重试」加一行 —— **让下一位不重跑它** |
| 设计变更（架构/接口/数据流） | 对应 [`design/`](design/) 的 HLD 或 LLD |
| 组件 pin 移动（任意 fork push） | [`STATUS.md`](STATUS.md) §1/§2（当前值）+ [`progress/landed.md`](progress/landed.md)（门证据）+ [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md) pin 时间线 |
| 镜像发布（有 manifest digest） | [`progress/landed.md`](progress/landed.md) 表 A + [`STATUS.md`](STATUS.md) §2 + [`deployment/version-matrix.md`](deployment/version-matrix.md) |
| 部署/版本变化 | [`deployment/`](deployment/) 对应 spec |
| 踩到一个**新坑**（不够大、不值得单独成篇） | [`postmortems/CASEBOOK.md`](postmortems/CASEBOOK.md) 加一条（≤14 行，打 ✅/🩹/⏸）+ §0 索引一行。**先查路由表它有没有别的家** |
| 落了一个**绕路**（根因仍在，绕路承重） | 同上，标 🩹 并写清「移除代价 / 复发条件」—— **不写就会有人当冗余删掉** |
| 一条坑值得**开工前就记住** | 再回填 [`postmortems/LESSONS.md`](postmortems/LESSONS.md) 一行触发式索引 |
| 新 **dev workflow 坑 / kernel 硬限制** | **写 sub-repo** `pypto-lib/docs/dev-workflow-gotchas.md` 或 `known-pypto-pitfalls.md`，本仓只在 CASEBOOK 留指针 |
| 新**部署失败** signature | [`deployment/machine-recovery.md`](deployment/machine-recovery.md)「常见部署失败」 |

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

0. **★ 开工前必读 [`postmortems/LESSONS.md`](postmortems/LESSONS.md)。** 那一页是 16 份复盘
   「如何避免」段的触发式索引（≤150 行），按 流程 → 整网 → 部署 → codegen 分段。
   **不读它就动手，等于自愿重犯已经付过学费的错**（本项目已有多次重犯记录，见
   [`postmortems/12-integration-churn-meta.md`](postmortems/12-integration-churn-meta.md)）。
   最低要求：扫一遍左列"你正要做"，命中就读该行；命中"改 orchestration 构造"、
   "删看似多余的同步"、"下死锁结论"、"说某东西 ready"这四类时**必须**读完出处。
   收到新教训 → 提炼一行回填 LESSONS.md，别只写在 milestone 里。
   **另外两个必查点**：① 撞上具体现象 → [`postmortems/CASEBOOK.md`](postmortems/CASEBOOK.md)
   按现象查；② **要删一段看起来冗余的代码 / 同步 / 构造之前**，先查 CASEBOOK §C
   「仍在生效的绕路」—— 🩹 标记的都是承重的，删了 bug 立刻回来。
1. **单卡 ST/UT 保 TP=8 per-rank slice 宽度**：用 `apply_perrank_patch()`，不用 `apply_tp1_patch()`（unslice 只适合 Phase 15 e2e，chunk-follow-slice 的 kernel 会爆）。
2. **Phase 16 三剑合璧**：多卡部署必须 driver 25.5.2 + firmware 7.8.0.7.220 + CANN 9.0.0-beta.1（NOT GA）。见 [`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md) + [`postmortems/01-multirank-ipc-507899-507018.md`](postmortems/01-multirank-ipc-507899-507018.md)。
3. **monkey-patch 后 .pyc stale**：跑过 `apply_perrank_patch`/`cfg.X=Y` 后，下次 fresh run 前 `find <pypto-lib>/models/step3p5 -name "*.py" -exec touch {} +`。
4. **三件套激活**：每个新 shell `source CANN/set_env.sh` + `source activate.sh` + `export PTO_ISA_ROOT=...`。
5. **git push 用 HTTP/1.1**（默认 HTTP/2 在 130s 静默超时）。
6. **生产整网只允许单个 `@pl.program`**（多 program 有 co-prepare 死锁墙，见 [`postmortems/08-multiprogram-coprepare-deadlock.md`](postmortems/08-multiprogram-coprepare-deadlock.md)）；**native W8A8 不回退 BF16-dequant**。
7. **验收以 canonical 为准**：[`reference/canonical-test.md`](reference/canonical-test.md)。**精度准出 = 多步 decode 逐 token vs vanilla（N=128 ≥95%）**；单 token `argmax==303` 仅为首 token 冒烟/liveness。性能优化回归走 `.claude/skills/pypto-perf-regression/`。
8. **强开发约束先读 skill**：`.claude/skills/pypto-dev-constraints/`；整网 hang 排查：`.claude/skills/pypto-whole-net-hang-debug/`。
9. **当前状态不得从本地 branch 名推断**：先读带日期的 [`STATUS.md`](STATUS.md) /
   [`planning/handoff.md`](planning/handoff.md)，再用 GitHub 远端 ref 核对 exact commit。
   `develop/N1/`、旧 phase、旧 benchmark、case-study skill 和本地同名分支都只能作为
   历史证据；若它们与当前 SSOT 冲突，以远端 ref + 当前版本矩阵为准，并立即修正文档。
