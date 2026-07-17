# N=1 案例：完整时间线导航

> 本文只保留时间线总览。按日期细节拆成“早期前史”和“N1 单程序主线”；
> 对根因措辞、保留项和被推翻判断的复核单独放在“关键因果链”。

## 时间线阅读图

```mermaid
flowchart LR
    A["06-15～06-24<br/>环境 / shape / empty-tail"] --> B["07-04～07-09<br/>同卡、TASK 映射、编译与底座漂移"]
    B --> C["07-10<br/>单程序编译、runtime、per-layer alias"]
    C --> D["07-11～07-12<br/>真实 IPC、OOM、exact kernel 映射"]
    D --> E["07-12～07-13<br/>W8A8 数值和 layer boundary"]
    E --> F["07-14～07-15<br/>303、过早闭环、canonical 冻结"]
    F --> G["07-16～07-17<br/>signal 布局 A/B 与 release 审计"]
```

## 按需阅读

| 时间或问题 | 文档 |
|---|---|
| 2026-06-15～2026-07-09：环境、部署、同卡、早期 kernel 与底座漂移 | [早期前史](n1-timeline-early.md) |
| 2026-07-10～2026-07-12：单程序、runtime、per-layer alias、IPC/OOM、exact kernel | [单程序 bring-up](n1-timeline-bringup.md) |
| 2026-07-12～2026-07-14：native W8A8、NaN、layer boundary、首次 303 | [精度主线](n1-timeline-precision.md) |
| 2026-07-14～2026-07-17：概率 stall、kernel 漂移、512B A/B、release | [stall 与 release](n1-timeline-stall-release.md) |
| 为什么某些历史判断被推翻、哪些修改必须保留 | [关键因果链](n1-causal-chains.md) |

## 时间线总览表

| 阶段 | 主要现象 | 排查转折 | 当前应保留的经验 |
|---|---|---|---|
| 06-15～06-24 | 507018、IPC capability、empty-tail | 先区分运行阶段，不把外层错误码当根因 | 环境/shape/kernel 必须分层 |
| 07-04～07-09 | 同卡失败、TASK 映射、TaskMapSize=0、底座漂移 | 建立 exact task → kernel 映射 | 失败 build、TASK 和 binary 必须同轮绑定 |
| 07-10～07-12 | 45 层单程序、alias、真实 IPC、OOM、gate_topk | 先修环境和生命周期，再审计具体 kernel | 不要把 IPC/VA 相关性直接升级为根因 |
| 07-12～07-15 | NaN、303、completion-wave 过早闭环 | native W8A8、index boundary、matched A/B | 精度与稳定性是两个 gate |
| 07-16～07-17 | kernel 位置漂移、32B signal、512B isolation | 回到跨 rank 最早阻塞边界和物理布局 | 512B 是强关联，不是无 PC 的唯一硬件证明 |

## 4.1 一页总览

| 日期 | 被测对象 | 主要现象 | 当时判断/动作 | 后续结论 | 是否保留 |
|---|---|---|---|---|---|
| 06-15～06-24 | 单卡/多卡基础 | 507899、Bootstrap 507018、MoE runtime 507018 | 先补 driver/firmware/CANN；清 stale pyc；对空尾做 dispatch-cut | 环境问题、空 tail 逻辑和后续 gate stall 是不同故障 | 保留环境/shape 检查 |
| 07-04～07-05 | routed kernel + 与 vLLM 同卡运行 | 独立 worker 与 vLLM 同卡 507018；16GB arena OOM | 调整 ring/HBM、换进程组织 | 属于同卡运行前史，不是 standalone 完整 42 个 MoE 层对象的根因 | 保留为分类案例 |
| 07-06～07-07 | EpTpMoE 单块 | `gate_topk` deterministic RUNNING hang | V0 TASK→func 映射，修 mrgsort | 真实状态机 bug，后在 N1 内联副本再次出现 | 修复保留 |
| 07-08～07-09 | multi-program/融合探索 | co-prepare、TaskMapSize=0、compile/device 混淆 | ring sizing、distinct-program sweep、底座升级 | 是架构探索和底座漂移前史；N1 最终只允许单 program | 经验保留 |
| 07-10 | 单程序 45 层 | 编译成功后先报 507899/507018；随后两层 shared-window 实验卡死 | 重建 runtime、关闭 SDMA；按层数截断 | runtime/SDMA 解决环境问题；每层独立窗口解决确定性别名 | 两类修复保留 |
| 07-11 | real-weight IPC | host OOM、arena OOM、S1 task3 | 建 exporter/import_ipc、改 slicing、降 arena | IPC/VA/a2a 归因后来被 exact TASK 推翻 | runtime 支持保留 |
| 07-12 | 完整执行 42 个 MoE 层、原生 W8A8 | `task_id=3`、`aiv0:3` 处于 RUNNING | 用同一次编译产物映射到 `gate_topk` | mrgsort format2 的输入不满足前置条件 | 修复保留 |
| 07-12～07-13 | 精度/NaN | NaN、1e11 幅值、错误 argmax | 层数 bisect、Out 修复、FUSE、op dump | 多个边界 bug；最终 routed expert 漂移由可靠 dump 定位 | 结构修复保留 |
| 07-13～07-14 | 完整 42 个 MoE 层的精度与卡死 | 执行 31 个 MoE 层能完成，完整 42 个 MoE 层卡死；首次得到 303 | 检查通信空间大小、L2 层索引和完成确认波 | 固定字节上限等判断被推翻；L2 索引是真实精度错误；卡死尚未解决 | 分类保留 |
| 07-14 晚 | clean tree 复验 | STALL/CLEAN(303)/STALL | 推翻“A2 已修” | 概率通过不能宣称完成 | 作为核心教训 |
| 07-15 | push/pull 深挖 | S1 RUNNING func28；pull 也会 stall | exact TASK→kernel、超时 A/B、push/pull 重写 | 某轮 kernel 位置不是统一根因；协议/layout 需整体审计 | fixed-slot pull结构保留 |
| 07-15 | 0162 复现 | push+push 6 次 2 clean/4 stall | 排除“只在 0234” | 不是严格 exact-manifest 跨机 A/B | 历史线索 |
| 07-16 | final layout A/B | pull+pull 仍随机停在不同 kernel | 32B signal physical allocation→512B | 0162 fresh pool 收敛；最小 layout 变量 | 512B 保留 |
| 07-17 | release 审计 | 候选 20-run 与 release source SHA 不同 | exact-source 重跑 20/20；三仓 clean-pin smoke | 补齐模型源码 release 证据，区分 old dirty runtime 与 clean pins | release 事实 |
