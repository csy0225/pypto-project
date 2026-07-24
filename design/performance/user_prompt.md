# 用户提示词（复制即用）

> 推进 step3p5 性能优化专项的现成提示词。所有推进都以
> [`.claude/skills/pypto-perf-regression`](../../.claude/skills/pypto-perf-regression/SKILL.md) +
> 本目录设计文档为单一入口，不用每次重复交代上下文。

---

## 1. 认领并推进一个子任务

把 `<PERF-ID>` 换成要做的子任务（如 `PERF-C1`，清单见 [`README.md`](README.md) 主表 / [`task-tracking.md`](task-tracking.md)）：

```
读 pypto-project/.claude/skills/pypto-perf-regression/SKILL.md 和 design/performance/
（README + 01-system-design + 02-detailed-design + task-tracking），按下面要求推进 <PERF-ID>：

1. 先看 02-detailed-design.md 里 <PERF-ID> 的卡片（问题/shape/如何生效/参考/改法/验证），
   动手前先补一张 step3p5-vs-v4-flash 差异表（差异+理由+改还是留），确认对齐 DeepSeek。
2. 在 task-tracking.md 把 <PERF-ID> 状态置 in_progress + 填 owner。
3. 按 skill 的 Step 0~5 落地：改代码 → liveness 冒烟 → 多步 decode 精度回归
   （N=128 ≥95% vs vanilla，只验第一个 token 不算数）→ DFX 采集对比 baseline。
4. 精度/性能全绿后按 skill Step 4 更新文档（task-tracking + perf-baseline，必要时 canonical/STATUS），
   不要新建散落文档；再按 Step 5 的 HTTP/1.1 协议 commit+push。

铁律：修根因不 work-around；精度只认多步 decode；每个结论要有可证伪的隔离实验支撑。
遇 stall 用 _probe_barrier_scale.py 区分 slow/deadlock。
```

---

## 2. 改完代码后只跑回归

```
我刚改了 step3p5 的 <文件/kernel>，对应 design/performance/task-tracking.md 的 <PERF-ID>。
按 pypto-project/.claude/skills/pypto-perf-regression/SKILL.md 从 Step 0 到 Step 5 完整回归一遍：
环境/pin/清pyc → liveness 冒烟 → 多步 decode 精度（N=128 ≥95% vs vanilla）→ DFX 对比 baseline →
更新 task-tracking + perf-baseline → commit/push。全绿才算完成，任一项红就在 task-tracking 记阻塞原因。
```
