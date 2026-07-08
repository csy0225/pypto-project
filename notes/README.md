# PyPTO 学习笔记（Notes）

> 个人学习心得与经验记录。每篇对应 [`pypto_top_level_documents/pypto-runtime-arch-docs/`](../../pypto_top_level_documents/pypto-runtime-arch-docs/) 里的一份权威架构文档，用**彩色 Mermaid 思维脑图 + 速查表 + 学习心得**的形式做提炼——原文档是"规格"，这里是"抓主干 + 记洞察"。

## 📚 目录

| # | 笔记 | 一句话 | 源文档 |
|---|------|--------|--------|
| 02 | [逻辑视图 Logical View](02-logical-view.md) | 抽象机器由哪些概念组成、怎么分层、谁依赖谁 | `02-logical-view.md` |
| 03 | [开发视图 Development View](03-development-view.md) | 概念落到源码里的 10 个模块、依赖 DAG、怎么编 | `03-development-view.md` |
| 04 | [过程视图 Process View](04-process-view.md) | 运行时怎么动：事件循环、状态机、依赖、分布式、延迟预算 | `04-process-view.md` |
| 05 | [物理视图 Physical View](05-physical-view.md) | 跑在什么硬件、怎么部署、怎么扩、坏了怎么办 | `05-physical-view.md` |
| 06 | [PyPTO 编程与部署 API](06-pypto-programming-and-deploy-api.md) | 怎么写 PyPTO 程序(L2/L3)、tensor/tile、API↔模式、buffer/融合/namespace 实战坑、编译部署 | 多源汇总 |

## 🎨 配色约定（贯穿所有笔记）

四类角色统一配色，跨笔记一致：

- 🔵 **蓝** — 地基 / 部署拓扑 / 并发（基础与词汇）
- 🟢 **绿** — 契约 / 网络 / 状态机（中间层与扩展点）
- 🔴 **红** — 装配 / 扩展 / 分布式（组装与消费者）
- 🟣 **紫** — 规格 / 可用性 / 延迟预算（横切与约束）

## 🖥️ 渲染提示

- 脑图用 `mermaid mindmap` + `classDef` 上色。**VSCode Mermaid 插件 / [mermaid.live](https://mermaid.live) 能完整显示图标 + 颜色**。
- mindmap 里的 `::icon(fa ...)` 需要渲染器加载 Font Awesome；GitHub 网页版不加载图标，但结构与配色不受影响。

## 🗺️ 一图看懂四视图关系

```mermaid
graph LR
    L["🔵 02 逻辑视图<br/>是什么"] --> D["🟢 03 开发视图<br/>代码在哪"]
    D --> P["🔴 04 过程视图<br/>怎么动"]
    P --> Ph["🟣 05 物理视图<br/>跑在哪/怎么部署"]

    classDef a fill:#4C6EF5,stroke:#1E3A8A,color:#fff;
    classDef b fill:#12B886,stroke:#0B7285,color:#fff;
    classDef c fill:#FA5252,stroke:#A61E1E,color:#fff;
    classDef e fill:#BE4BDB,stroke:#6B2178,color:#fff;
    class L a;
    class D b;
    class P c;
    class Ph e;
```

---

*相关：架构对比笔记见 [`../architecture/`](../architecture/)；权威架构文档索引见 [`../../pypto_top_level_documents/pypto-runtime-arch-docs/00-index.md`](../../pypto_top_level_documents/pypto-runtime-arch-docs/00-index.md)。*
