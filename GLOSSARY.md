# 术语表 / Glossary

> 项目内部约定名、缩写、外部技术名词一览。第一次读 CLAUDE.md 或 docs/ 前先扫一遍。

## 项目仓库 / Project Repos

- **PyPTO** — 整个编译器 / runtime 栈的统称。也指核心仓库 `csy0225/pypto`。
- **Step3p5** — stepfun 自研大模型代号。架构 DeepSeek-V4 风格、MoE、45 层主体 + 3 MTP 层。本项目 bring-up 的目标模型。
- **pypto** — PyPTO 编译器主仓：IR、passes、codegen、runtime API
- **pypto-lib** — 模型实现仓，含 `models/step3p5/`
- **pto-isa** — NPU 指令集抽象层（cross-arch IR）
- **PTOAS** — 汇编器（PyPTO Assembler）：IR → NPU bytecode
- **simpler** — 运行时：chip_process fork、HCCL bootstrap、内存分配、跨卡 IPC。在 pypto 里作为 `runtime` 子模块挂着
- **stepfun/develop** — 5 个 fork 共同的开发分支名

## 华为 Ascend / CANN 术语

- **Ascend / 910B2C** — 华为自研 AI 芯片，本项目用 8 卡机型
- **CANN** — Compute Architecture for Neural Networks，华为 driver + SDK 套件。当前用 `cann-9.0.0-beta.1` 配 driver `25.5.1`
- **AICore** — Ascend 计算核心，矩阵密集计算（含 AIC + AIV）
- **AIC** — AICore Cube：矩阵 / GEMM
- **AIV** — AICore Vector：矢量 / elementwise
- **AICPU** — Ascend AI CPU：控制流、小算子、tiling 计算
- **mixed kernel** — 同时含 AIC + AIV 的融合 kernel（例：`full_fa_fused`）
- **HCCL** — Huawei Collective Communication Library（类似 NCCL）
- **A2A3 / A5** — Ascend 平台代号；本项目主用 a2a3（对应 910B2C）
- **exbus / shmem-exbus** — 跨 die / 跨卡 device-memory 共享通道。`support_shmem_map_exbus` 是 driver 能力位，决定跨卡 IPC 是否可做
- **UB** — Unified Buffer，AICore 内部 SRAM，VEC 指令访问对齐有要求

## 并行 / 分布式

- **TP (Tensor Parallel)** — 把单层张量切分到多卡。本项目 TP=8
- **EP (Expert Parallel)** — MoE 专家切分。本项目 EP=8
- **MoE** — Mixture of Experts，专家混合架构。Step3p5 有 32 routed experts + 1 shared expert per layer
- **SPMD** — Single Program Multiple Data：所有 rank 跑同一段 IR，处理不同数据
- **chip_process** — simpler runtime 在每张 NPU 卡上 fork 出的子进程

## 模型推理阶段

- **prefill** — 处理初始 prompt 的并行计算阶段（计算密集）
- **decode** — 逐 token 生成的串行阶段（带 KV cache）
- **KV cache** — attention 的 key / value 缓存
- **MTP** — Multi-Token-Predict，加速 decode 的 next-token 预测层（本项目 3 层）
- **full / SWA** — full attention vs sliding-window attention（Step3p5 部分层 SWA）

## 错误码 / 常见崩溃

- **507018** — Ascend AICore 错误：`aclrtSynchronizeStreamWithTimeout` 失败、AICore 不可恢复。本项目 bring-up 主要 blocker，根因 = `full_fa_fused` 里 VEC 指令 UB 地址未对齐（subErrType:4 / errcode 0x800）
- **507899** — Ascend IPC 错误：`aclrtIpcMemImportByKey` 失败。本项目 = driver `support_shmem_map_exbus=0`，跨卡 IPC 走不通
- **VEC UB align** — AICore VEC 指令访问 UB 地址未对齐
- **507015** — 紧随 507018 的 device sync 超时

## 项目阶段（简称）

| Phase | 内容 | 状态 |
|---|---|---|
| 01-08 | 设计：config / checkpoint / 单层 / MoE / prefill 单卡 | ✅ |
| 09 | E2E integration + smoke + weight loader | ✅ |
| 10 | TP=8 + EP=8 多卡重构 | ✅ |
| 11-13 | Frontend 联调（pypto IR → bytecode 路径） | ✅ |
| 14 | Codegen pass 完整 | ✅ |
| **15** | **单卡 NPU bring-up（进行中）** | 🟡 26/33 task |
| **16** | **多卡 NPU bring-up（进行中）** | 🟡 卡驱动缺口 |
| 17 | 64K prefill + 16-step decode 端到端 | ⏸ |
| 18 | 性能：l2_swimlane + PMU | ⏸ |

## 任务 / Issue 编号约定

- **TASK-N** — 项目内 backlog 编号（在 dev pod 的 `backlog/` 下，未同步到本 mirror）
- **#NNNN** — upstream `hw-native-sys/<repo>` 的 GitHub issue / PR 编号
- **bring-up** — 模型在新硬件平台首次跑通的过程（区别于性能调优）

## 文件命名约定

- `*.md` — 英文 / 主版本
- `*.zh.md` — 中文翻译（部分文档双语，以 `*.md` 为 canonical）
- `_*.py` — 私有 / WIP / 实验脚本
- `repro_*.py` — bug 复现脚本
- `_compile_*.py` — codegen probe / 编译入口
