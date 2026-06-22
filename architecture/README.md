# 架构 Architecture

pypto step3p5 栈的跨仓库 design notes。这些 doc 描述 5 个代码仓怎么拼
在一起，以及 Phase 2 vLLM 集成 interface。

## 内容

| 文档 | 用途 |
|------|------|
| [`overview.md`](overview.md) | 大图：5 仓 + vLLM 各自做什么，数据怎么流 |
| [`vllm-step3p5-mapping.md`](vllm-step3p5-mapping.md) | vLLM 的 `Step3p5Model` 与 pypto 的 `decode_fwd` 之间的 op-级映射 —— Phase 20 monkey-patch 实现必读 |

## 什么时候加新 arch doc

- **新仓库加进项目** → 在 `overview.md` 描述它的角色 + 如果它有自己的
  内部架构，加一个 focused doc。
- **新跨仓 interface 引入** → 那个 interface 一个 focused doc（如
  vllm mapping doc）。
- **非显式 data flow 需要文档化** → focused doc。

避免文档化某个仓内部细节 —— 那些归对应仓的 `docs/`。本目录只放
**跨仓** design 内容。
