# Phases 阶段

pypto step3p5 项目的活跃 phase 跟踪。每个 phase doc 描述一个相干工作
单元：goal / scope / decisions / tasks / exit / risks / status。

Phase 01-19（pypto kernel 原型，2026 年 6 月完成）的摘要在
[`../archive/prototype-phase-01-19-summary.md`](../archive/prototype-phase-01-19-summary.md)。
Phase 20+ 跟踪 vLLM Ascend 后端集成工作，在这里活跃跟踪。

## Index

| Phase | 标题 | 状态 | 文档 |
|------:|------|------|------|
| **20** | vLLM 后端 monkey-patch —— e2e 流程 | 📐 设计已落 2026-06-22；任务 1.1-1.9 未启动 | [20-vllm-backend-monkey-patch.md](20-vllm-backend-monkey-patch.md) |
| **21** | 精度验证 harness（与 vLLM 原生对比） | 📐 设计已落 2026-06-22；gate Phase 20 | [21-precision-validation.md](21-precision-validation.md) |
| **22** | Perf baseline + 调优 | 📐 设计已落 2026-06-22；被 Phase 26 取代 | [22-perf-baseline.md](22-perf-baseline.md) |
| **23** | 零拷贝 KV-IPC 集成：step 1-5 验证 + 重制定 plan | ✅ **验证完成 2026-07-03**（IPC 主卡点解除） | [23-zero-copy-kv-ipc-validation.md](23-zero-copy-kv-ipc-validation.md) |
| **24** | step 6：整层 live 替换（一 key 整池 map + page_attention） | 🟡 24.1-24.3 ✅（全 45 层 attention live 对齐 baseline，无 OOM）；24.4 整层 ⏸（dense 卡双-worker co-tenancy，MoE 卡 507018） | [24-live-layer-replacement.md](24-live-layer-replacement.md) |
| **25** | step 7：真 module 全网 + whole-model orchestration（Wave-3） | 🟡 设计 kickoff 2026-07-03（host_orch 48 层融合；subsumes 24.4）；实现多周级 | [25-whole-model-orchestration.md](25-whole-model-orchestration.md) |
| **26** | Perf baseline + 调优（原 Phase 22，零拷贝后重测） | ⏸ 待做（gate Phase 25） | 23 doc §5 |
| **27** | N=1 整网融合（单 @pl.program 全 45 层 + tail） | 🟡 **独立攻关线** kickoff 2026-07-09（Phase 0 环境重建中；分支 `feat/whole-net-n1-fusion`，机器 0234；不碰 0162） | [27-n1-whole-net-fusion.md](27-n1-whole-net-fusion.md) |

每 phase 的实时状态见 [`../STATUS.md`](../STATUS.md)。

## 每个 phase 的更新协议

phase 节点变状态时：

1. 更新该 phase doc 的 `## Status` 段。
2. 更新上面 index 表的对应行。
3. 更新 [`../STATUS.md`](../STATUS.md) "Phase 2 sub-phases" 表。
4. 如果浮现新 blocker，加到 [`../blockers.md`](../blockers.md)。
5. 记录新的 pin snapshot 在 phase doc 顶。

## 交叉引用

- [`../STATUS.md`](../STATUS.md) —— 实时状态板
- [`../blockers.md`](../blockers.md) —— 活跃 open issues
- [`../architecture/overview.md`](../architecture/overview.md) —— 跨 5 仓
  + vLLM 系统架构
- `pypto-lib/docs/known-pypto-pitfalls.md` —— kernel / codegen 硬限制
  （在 pypto-lib 仓）
- `pypto-lib/docs/dev-workflow-gotchas.md` —— workflow 坑（在 pypto-lib 仓）
