---
name: pypto-dev-constraints
description: >
  PyPTO / step3p5-on-Ascend-910B 项目的强开发约束（设计宪法 + 工程经验）。
  写/改/审 pypto kernel、@pl.program、单卡多卡 ST/UT、编译 codegen、
  多卡运行时/通信、W8A8 精度对齐、vLLM 集成、真机部署，或排查
  507018/507899/UB-overflow/const-fold/deadlock 时，必须先读本 skill。
  两层优先级：设计宪法（不可违反，最高优先级）＞ 工程约束（额外经验）。
---

# PyPTO / step3p5 开发强约束

> **怎么用**：动手前先过「第 0 层 设计宪法」——违反它的方案直接推倒重来；再对照「第 1 层 八类工程约束」逐条自检。每条带**为什么**和**出处**；深度看 `pypto-project/notes/06·07` 和引用的源文档。
> **两条元规则**：① 遇问题**修根因、不 work-around**（诊断脚手架只能定位、不能进产品路径）；② 排查四板斧顺序 = DeepSeek 同栈怎么做 → 上游(pto-isa/PTOAS/simpler)是否已修 → kernel 逻辑/自写 → 数据类型。

---

## 第 0 层 · 设计宪法（不可违反，最高优先级）

> 抽自 `pypto_top_level_documents/`。"不变量" = 绝不可违反；"选择" = 已定方向、可能演进。

### ⭐ 九条核心不变量（动手前最易违反、最难调试）

1. **统一 `ISchedulerLayer` 递归契约**：每层实现同一 submit/scope/drain 接口，layer-N Worker 向 layer-(N−1) 递归 submit；加新硬件层只实现接口 + 注册，**不改现有代码**（OCP）。`08-design-decisions §ADR-001/003`
2. **shape vs valid_shape 解耦**：`shape`=存储（**必须 `prod(shape)*sizeof % 512 == 0` 且 ConstInt**），`valid_shape`=逻辑范围（`≤shape`，可运行时 Expr）。512B 只约束 shape。`tensor_valid_shape.md`
3. **sharded_tensor 恒等式**：`shape[i] == shard_shape[i] * rank_shape[i]` 且 `prod(rank_shape)==rank_num`；访问带 `rank_index` 向量，一次一 rank，**无 fused multi-rank 原语**。`sharded_tensor.md §3.1/§5`
4. **buffer 双条件回收**：可回收 ⟺ scope token 已 apply（scope.exit 或 `pl.free`）**且** `ref_count==fanout_count`（fanout 起始=1）。`multi_level_runtime…§6/§8` + `ADR-006`
5. **sharded_tensor 构造 = 全 rank all-to-all blocking barrier**：barrier 返回前任何 remote access 非法（弱同步会 race 物理映射）。`sharded_tensor.md §6.2`
6. **InCoreFunctionGroup 同 cluster 共调**：AIC+2AIV 必须同物理 cluster、共享 TPUSH/TPOP ring；`call_group` 展开 AIC×1+AIV×2，AIV_IDX 由 runtime 注入。`HL…Mixed_Kernel §4` + `machine_hierarchy §3.4`
7. **AIV 双核 split "DUPLICATED by default"**：仅当整条 chain 兼容同一 axis 且无 forbidden（reduction 的 reduce axis forbidden）才 split；两 AIV 核**无跨核通信**，split 不可回 gather。`HL…Mixed_Kernel §9c`
8. **关键路径零 Python**：prefill/decode/KV-cache/radix 查找全 C/C++；Python 不出现在自回归、prefill→decode、KV 访问路径。`pypto_serving_design goal §1`
9. **L3 同构 L2、不改 simpler**：L3 复用 L2 全执行模型（scope+ringbuffer+tensormap+submit），经 ChipBackend adapter；无新概念。`simpler_distributed_runtime_design §1.2`

### P1 递归层级模型
- Linqu **L0(Core)/L1(Die)/L2(Chip)/L3(Host)/L4-6(Cluster)/L7(Global)** 自底向上 enclosure；编译器**必须**给每层函数贴 hierarchy label，即便 runtime 只支持 L0/L2。`machine_hierarchy §1/§2.1`
- **Orchestration/Cluster/InCore** 三级执行模型分层；`role=ORCHESTRATOR`（建 DAG/submit，**从不**算）与 `WORKER`（算，**从不** submit 子任务）严格二分。`§2/§5.7`
- `pl.Level` + `pl.at()` 统一语法；`pl.incore`/`pl.auto_incore` 已弃用。（选择）`§5.5-5.7`

### P2 数据模型契约
- `valid_shape` view/slice/transpose 自动推导；reshape 跨 padded+valid 轴合并时编译器报错强制显式覆盖。`tensor_valid_shape.md`
- `tile_shape` 仅物理 layout hint（`shape[i]%tile_shape[i]==0`），破坏 tile-contiguity 时 warning + 回退 None。（选择）`tensor_layout.md`
- **local tensor 永远本地不可跨节点**；跨节点共享只经 sharded_tensor。`sharded_tensor.md §7`
- collective 作为 `ST.all_reduce(op=…)` typed method，不传裸 buffer/count/comm。（方向已定）

### P3 依赖/生命周期契约
- `pl.free` 幂等、不绕 fanout（`task_freed` flag 防双 increment）。`§7.1/§8`
- 多层 ring stack：每 scope depth 独立 ring，内层 retire 不阻塞外层；规范 id = `TaskKey(scope_level, task_id)`（`task_id` 全局不唯一）。`§13`
- **依赖 RAW-only v1**：前端 owns intra-Submission RAW/WAR/WAW/assemble → `intra_edges`+boundary masks；runtime `producer_index` 单值 RAW-only。前提不变量 = `IMemoryManager` 保证 non-aliasing intermediate memref。`ADR-013` + `07 §7.5`
- Submission DepMode ∈ {BARRIER,DATA,NONE}；NONE=caller-asserted 无外部边；outstanding-submission-window 是 global back-pressure。`ADR-012`

### P4 通信模型契约
- **控制/数据路径分离**：Vertical/Horizontal Channel 载 typed control msg，`IMemoryOps` 载 bulk data（避免 HOL blocking）。`ADR-004`
- **TPUSH/TPOP** tag-based 双通道 ring：单向 SLOT_NUM=8/双向 4；C2P space-free 由 `tfree` 发（**非** `tpop`）；A2A3 ring 在 GM、A5 在 consumer SRAM（kernel 行为平台无关）。`tpush_tpop_isa_design_v3.md`
- IR 着色 **RED=AIC(matmul/cube) / GREEN=AIV(view/element-wise/reduce) / WHITE=控制**；跨色插 TPUSH/TPOP。`HL…Mixed_Kernel §2/§6`
- `call_spmd` 隐式追加 `spmd_idx`/`spmd_size` 两 uint32 参数（用于互操作 AscendC/Triton/CUDA legacy）。`§SPMD`
- 异步引擎（SDMA/RoCE/CCU）`complete_in_future=True`：返回只释放 core、不调 on_task_complete；`waiting_completion_count==0` 才真完成。`runtime_async.md §2`

### P5 扩展性/模块化
- Machine Level Registry + **6 pluggable factories**（Scheduler/Worker/MemoryManager/MemoryOps/Vertical/HorizontalChannel）；topology 是配置非代码；**registry 用前 frozen**。`ADR-002`
- Scheduler 内拆 TaskManager/WorkerManager/ResourceManager + 3 strategy 接口；外部 `ISchedulerLayer` 不变。`ADR-008`
- **`distributed/` header 禁止从 `transport/` TU include**（IWYU+visibility 双护）；transport API 只 `send(peer,MessageType,span<byte>,Timeout)`，payload opaque。`ADR-015`

### P6/P7 serving 目标 + 关键 ADR / 已知偏离
- 无长期上下文（引擎不维护 user/session 状态；唯一持久 = Radix Tree + KV pool）。`serving goal §11`
- TaskState 失败终态 = `ERROR`（Task）/ `FAILED`（Worker），别混。`ADR-016`
- Simulation 三模式 leaf-engine 工厂替换，**无静默 fallback**（PERFORMANCE/REPLAY 未实现则显式 reject）。`ADR-011`
- 故意偏离（Rule 例外，**别当 bug 去"修"**）：intra-node 不加密、无持久化任务状态恢复（crash→Python 幂等 re-submit）、跨节点加密下推 backend。`10-known-deviations.md`

---

## 第 1 层 · 八类工程约束（额外经验）

### A 版本 / 环境绑定
- **多卡 e2e 三件套齐全**：driver `25.5.2` + firmware `7.8.0.7.220` + CANN `9.0.0-beta.1`（**非 GA**）。缺一即 507899/507018/sched=100。`deployment/phase16-three-pillars.md`
- driver+firmware **成对升级**（cap 共同 gate）；**CANN 不升 GA**（GA 的 TDT 不推 AICPU syskernels → BootstrapDispatcher 507018）；升级前备份 beta.1。`version-matrix.md`
- **三件套激活**（每 fresh shell）：`source cann/set_env.sh && source ws/activate.sh && export PTO_ISA_ROOT=ws/pto-isa`（`activate.sh` 不带 CANN env）。`CLAUDE.md 铁律 §4`
- **ptoas-bin ≥ v0.45**（v0.44 有 `pto.tci ui32 {descending=false}` parser bug）；pypto codegen 越过动 MLIR op 的上游 commit 时必须同步 bump。`version-matrix.md`
- CANN 升级后 **clean 重编** pypto+runtime+simpler（`rm -rf build/cp311-* build/cache build/lib`）；**不传 `CMAKE_BUILD_TYPE`**（避 `tensor.h buffer_elems -Werror=unused-variable`）。`machine-recovery.md`
- 0162 是 **netboot/tmpfs**，重启丢 `/usr/local/Ascend`+`.venv311`+仓库副本，持久在 NVMe/`/mnt/persist`。
- 8 卡运行必设 `PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`；simpler 链接必须 `-Wl,--no-as-needed libhcomm.so -Wl,--as-needed`（#1018）。

### B 单卡 ST/UT vs 多卡 shape 铁律
- **单卡 ST/UT 用 `apply_perrank_patch()`（保 TP=8 per-rank slice 8/12/1/1408/160/36），禁用 `apply_tp1_patch()`**（unslice 全宽让 sh_mlp/gate_matmul 爆 L1/UB）。`feedback_single_card_st_shape_iron_rule`
- canonical = **TP=8/EP=8**；TP=1 仅 transient bring-up，**绝不 bake 进 config.py/共享模块**，源改必须 ADDITIVE。`feedback_single_card_vs_multi_card`
- **固定 chunk（`MLP_OUT_CHUNK=128`）** 推荐；**"chunk 跟 slice 走"（`_CHUNK=*_LOCAL`）** 在 unslice 时爆。`CLAUDE.md 顶部表`
- 8000 oracle 占 cards 0-7 时，device 验证必须 **`MOE_ST_DEV_OFFSET=8`** 且确认 fork 到 8-15（否则 reset 打到 oracle 卡）。`moe-block-nextwork-and-constraints.md`
- Phase 15 单卡 e2e 三件套：head_gate bypass + `--tp-world-size 1` + `LAYER_INTER_ROWS_DYN/LAYER_QHIDDEN_ROWS_DYN` TP=1 override。`project_p15_singlecard_e2e_unblock`

### C 编译 / codegen 硬限制

**buffer 容量上限（910C，超即 `AllocateMemoryAddr` reject）** — 层级：L2=一颗 chip / L1=die·L2cache / L0=单 core（`performance-tuning.md:10,169-170`）：

| buffer | 别名 | 物理容量 | 编译报错阈值(有效预算) | 用途 |
|--------|------|---------|----------------------|------|
| **UB** | Vec | **192 KB** | **188416 B (=184 KB)** ⚠ | 向量运算工作集 |
| **L1** | Mat | **512 KB** | 512 KB | cube 左/右操作数 staging |
| **L0A** | Left | **64 KB** | 64 KB | cube 左操作数 |
| **L0B** | Right | **64 KB** | 64 KB | cube 右操作数 |
| **L0C** | Acc | **128 KB** | 128 KB | cube 累加器 |

（UB 物理 192KB 但编译器按 **188416 B(184KB)** 卡，`known-pypto-pitfalls §7:351`）

**对齐规则（4 条，别混——作用对象/数值/硬件/性质各不同）**：

| # | 规则 | 作用对象 | 具体数值 | 硬件 | 性质 / 失败 |
|---|------|---------|---------|------|-----------|
| 1 | **行 32-B 对齐** | intra-UB **Vec / none_box tile 的每一行** | `cols × sizeof(dtype)` 必须 **%32==0**：FP32 cols%8==0(≥8) / BF16 %16(≥16) / INT8 %32(≥32) | UB（AIV 每 micro-op 取 32B=256bit） | `pto.alloc_tile` 静态 reject；**intra-UB VEC tile 静态漏检 → 运行时 `errcode 0x800 subErrType:4` → 507018** |
| 2 | **GM↔UB tile 512-B 对齐** | TLOAD/TSTORE 的 tile（GM↔UB load） | tile 512B 对齐；`tile_shape==valid_shape` 时跳过检查 | GM↔UB DMA | 静态检查 |
| 3 | **tensor 存储 `shape` 512-B 对齐** | tensor 的 **`shape`（存储布局，≠ valid_shape）** | `prod(shape) × sizeof(dtype) % 512 == 0` | GM/DDR DMA 512B 块 | **设计不变量**（P2.2） |
| 4 | **L2 cache line 512-B（性能，非 correctness）** | tensor 的 **trailing（最内）维** | multiple of 512B → **BF16 256 elem / FP32 128 / INT8 512** | L2 | 慢 MTE 路径；`perf_hints.log PH001` 告警 |

> 出处：#1 `known-pypto-pitfalls §1(:54-64)/§2(:97-128)`；#2 `§1:62-69`；#3 `tensor_valid_shape.md`（= 设计宪法 P2.2）；#4 `performance-tuning.md:272-319`。**记忆钩子：32B 只管 UB 里 Vec tile 的行宽；512B 管三处（GM↔UB tile、tensor 存储 shape、L2 cache line 性能）。**

- **`[N,1]`/`[1,1]` FP32 tile 禁用 `pl.slice` 构造**（行字节=4B 违反规则#1，静态漏检→运行时 507018）；用 `pl.row_sum`/`pl.row_max`/`pl.reshape` 或 `row_expand_mul` 广播。`§1`
- kernel 体内 **禁裸 `for`**；用 `pl.range/parallel/pipeline/spmd/unroll/while_`。`§6`
- **`pl.range(常量)` 全 unroll 不复用 SSA buffer → UB overflow**；用 `pld.nranks(ctx)` runtime bound 或 acc 写回 `local`。`§7`
- **barrier `tp_all_reduce` 禁 `tp_chunk=HIDDEN//tp_size`**（TP=1 爆 UB）；用固定 `ar_chunk=HIDDEN//8`。`§7a`
- **`pl.matmul_acc` 小 N=16 丢 K 累加**（gate_logits ~20× 偏小，codegen bug）；gate 走 worker 端预算，勿放回 on-device 小 N matmul。`§8` + `troubleshooting-8001-pypto-bridge.md`
- `pl.dynamic(leading dim)` 跨函数丢父 stride（pos>0 错位）+ 产 phantom int32；model-bound dim 用 config.py **静态 int**。`§3/§4`
- **closure-factory `_build_*` 返回的 `@pl.jit.inline` 不能被外部 `@pl.jit/@pl.program` 调用**；production 把 body **逐字复制进 self.method**（`@pl.program` 内不能实例化另一个 `@pl.program`）。`project_step3p5_closure_factory_not_externally_callable` + `aclgraph-vs-pypto §10`
- **swa_moe 融合 const-fold cascade**：`pl.full([SWA_Q_PAD_ALIGNED − Q_HEAD_BATCH_SWA, …])` 产 IR `Sub` 不折叠 → "must be ConstInt"；改 DSV4-style 固定 const + fillpad/mask。`project_whole_model_pypto_design` + `blockers.md`
- `pl.full` 不能在 Orchestration body（须 InCore/spmd）；`pl.unroll(N)` bound 须 module-const；A2/A3 `tmov` 要求 src/dst 同 shape（不能 slice-assemble 进宽 Vec tile）；`pl.cast(<bool>,INT32)` 在 Ascend **TRUE=-1**；1D `cast+slice` 失败→用 2D `[1,32]` chunk。`troubleshooting-8001` / `moe_shared_swiglu16_wide_tile_clamp` / `moe_gate_topk_tail_precision`

### D 运行时 / 多卡 / 通信
- **多程序 co-prepare N≥6 死锁**（distinct 程序 7 通、8 挂）：PREPARE 阶段 ring `task_window=65536`(2^16) 正解（2^20 会 64GB OOM）；DISPATCH 阶段 wedge = 上游 fork-then-prewarm race，**未解**。`blockers.md`
- 数字 device error **先查 [wiki Device-Error-Codes_zh](https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh)** 再深 debug。`reference_simpler_device_error_codes`
- IPC：`aclrtIpcMemGetExportKey` **只在 dptr==块基址**成功（base+offset→507899）；**一块一 key**，export 后必须 `aclrtIpcMemClose` 收尾（裸指针上 close 会 segfault）；forked chip 场景用 export-side `DISABLE_PID_VALIDATION`+import `ENABLE_PEER_ACCESS`。`troubleshooting-8001` / `phases/22-device-shared-inprocess.md`
- EP-MoE 507018 = `HEAP_RING_DEADLOCK`(orch_error=2) → `PTO2_RING_HEAP≥1GB/ring`；ring→**barrier-mesh** + 固定 ar_chunk。`project_moe_multicard_blocker` / `project_barrier_allreduce_fixes`
- **EP barrier 用 `AtomicAdd`（非 `Set`）**；`ep_all_to_all` 用**对称 fixed-slot a2a**（dst*MAX/src*MAX，read_off=my_rank*MAX，原路径只 rank0 对）；`pub_counts`↔barrier 间加 `pipe_barrier` fence。`pypto_if_int_lt_symbolic_dropped` / `moe-block-nextwork`
- **DeepSeek 不 fuse attention+MoE** → step3p5 用 **Option-C 两独立 program**（TP-attn program → resid1 → EpTpMoE block），别把 TP-attn+EP-MoE 塞一个 chip_orch。`blockers.md`
- gate_topk 必须 **format1 渐进 mrgsort 链**（format2 半块未排序→状态机不终止→挂死）。`troubleshooting-moe-block-8card-gate-topk.md`
- `_expert_routed` partial-tile grouped-GEMM **去掉输入 `valid_shape` + 新变量 `fillpad(gated,0)`**（cube 16×16 fractal 会混无效行）；`tile_valid>0` guard 防空 tile 提交 507018。`blockers.md` / `moe-block-nextwork`
- `add_inout`（非 `add_output`）表达写后读依赖；decode kernel 不能做 prefill 跨 token KV 可见（prefill 走 `attention_full_prefill`）；dummy `seq_lens=ones`（非 zeros，simpler#1023）。`debugging.md §6` / `project_attn_live_prefill_wrong_kernel`
- **禁 `npu-smi set -t reset`**（netboot 机会重启全 16 卡锁死）；**禁 `-9` 强杀** device 进程（无 finalize→card poison→507018）；`pkill -f` 用 `[p]attern` 括号 trick；恢复 8001 顺序 = 先起 8001 等 HCCL init 完成再起 pypto worker。`moe-block` / `troubleshooting-8001`

### E 精度对齐口径
- **oracle = vLLM eager detail dump**；**synthetic golden 会 stale**（误报 FAIL）。L1 `ratio_allclose(atol=0.04,rtol=0.04,max_error_ratio=0.10)`；L2 cos≥0.999+topK overlap≥4/5；L3 greedy top-1≥95%。`phases/21-precision-validation.md`
- **W8A8 不复用 BF16 golden**（`--quantization ascend` 重采）；routed 专家必须 **per-token INT8 dynamic-quant**（input x + clamp 后中间激活两处），**shared 不 quant**（BF16）。`STATUS.md` / `moe_swiglu_missing_int8_requant`
- quant 在**独立 pre-dispatch stage**（`_quant_moe_input` InCore，`pl.at(CORE_GROUP)` + ALL-T ONE-block；`spmd(2)` 写 Out race→49%）；scale per-token `[T,1]` 用 `row_max+reshape`（不 `create_tensor([N,1])`）；`pl.cast(x*127/amax, INT32, mode="rint")`。
- `router_bias` **必须 BF16-round**（vLLM 用 BF16，FP32 loader 让 top-8 尾部错）；shared swiglu16 clamp 拆 **5×`[T,32]`** chunk（宽 `[T,160]` Vec tile miscompile）。`moe_gate_topk_tail_precision` / `moe_shared_swiglu16_wide_tile_clamp`
- **EPS=1e-5**（非 1e-6，对齐 vLLM `GemmaRMSNorm`）；swiglu_limit 用 step3p5 层表（L43=7/L44=16）非 DeepSeek；routed 对齐用 vLLM router dump 的 `topk_ids/topk_weights`。`STATUS.md`
- weight_loader：**45-row norm 按绝对 layer_idx**、**42-row MoE 按 pos=layer-3**，不混；fused 模式传整 stack。`feedback_step3p5_weight_stack_index_class`
- **禁止 silent vanilla fallback mask pypto NaN/error**（`PYPTO_ATTN_AB=1` 只是 debug，不是 ship）；真"整网端到端精度" = **live single-handoff A/B（8001 pypto vs 8000 vanilla token-exact）**（offline chained 缺 KV 无法覆盖 attention-core）。`feedback_no_mask_pypto_errors` / `project_whole_model`
- head_gate ×1 bypass vs vLLM sigmoid 差 ~2× scaling → Phase 21 §2.7 必须标定。`blockers.md`

### F 工程流程 / 同步协议
- monkey-patch 模块全局后必须 `find models/step3p5 -name "*.py" -exec touch {} +`（pyc 序列化 patched 值）。`feedback_stale_pyc_after_monkey_patch`
- **git push 必须 `-c http.version=HTTP/1.1`**（HTTP/2 在 130s 静默超时）；PAT 走 `/data/chensiyu/secrets/github.env`，输出屏蔽 token、不落 `.git/config`。`CLAUDE.md 铁律 §5`
- "update+push fork" 前 audit `git log origin/main..HEAD`，drop 上游已修的 patch（最小 divergence）。`feedback_minimize_divergence_from_upstream`
- **跨仓 push 同步本仓 STATUS.md pin snapshot + archive milestone，两 push 同会话做**；项目跟踪在本仓、代码 reference 在 sub-repo `docs/`、模型在 `models/`。`CLAUDE.md 同步协议`
- launch NPU 任务前 `pkill` + `pgrep` 确认死 + `npu-smi info -t usages` HBM<10%（residual proc→`halMemCtl EACCES`）。`feedback_verify_processes_killed_before_launch`
- V0 定位：`logging.getLogger("simpler").setLevel(15)`（在 `worker.init()` 前）+ `ASCEND_PROCESS_LOG_PATH=<预建目录>`；停机 kernel id → build 的 `chip_orch/kernel_config.py` func_id。`moe-block` / `troubleshooting-moe-block`
- 注释/commit 用**中文**（技术名词保留英文），**不带 emoji、不写 .md 报告文件**（结果走对话）。

### G kernel 写法
- `pl` 唯一 alias；`pl.at(level=CORE_GROUP)` 唯一 level + `name_hint`；不混 `@pl.jit` 与 `@pl.program`；`@pl.jit.inline` 必须返回 value。`pypto-coding-style.md`
- loop 语义：`pl.parallel` 无 carried state / orchestration only；`pl.range` 任意；`pl.pipeline` 必须 inside `pl.at` + `stage=` 必填；`pl.spmd` 自带 InCore、**不能套 `pl.at`**。K-loop 用 `pl.pipeline(stage=2/4)` 不用 `pl.range`（cube 会 stall on load）。
- `pl.matmul_acc` 的 acc 须 `pl.create_tensor` 在 `pl.at` **外**；`pl.split` 仅用于混合 cube+vec region（UP_DOWN/LEFT_RIGHT）。
- `pl.slice(sizes, offsets)` 顺序；`set_validshape`+`fillpad` 配对做 softmax tail-mask；reduction 构造 `[B,1]`（不 `pl.full([N,1])`）。
- **512B L2 cache line**：BF16 trailing ≥256 elem / FP32 ≥128 / INT8 ≥512；kernel 目标 ~50µs（太小 fold/merge/mix，太大 split）。`performance-tuning.md`

### H 集成架构
- **整网执行在 pypto，vLLM 只调度 + KV cache**；准出 = 端到端 + 精度双过。`moe-block-nextwork §8`
- **整模型 monkey-patch 在 `Step3p5Model.forward`（一次 45-layer+lm_head），不 per-layer**（45 次 launch 抹掉融合优势）；`per_layer=True` 只给 Phase 21 精度 diff。`phases/20-vllm-backend-monkey-patch.md`
- **comm option A**：pypto kernel 内用 simpler shmem-IPC（不写 simpler↔HCCL bridge）；被 patch 的 pypto 路径 **`enforce_eager`**（pypto kernel 与 vLLM aclgraph 互斥）。`aclgraph-vs-pypto §D1/D3`
- monkey-patch 留在 pypto-lib（`tools/step3p5/vllm_monkey_patch.py`）经 sitecustomize 注入 stock vLLM，**不 fork vllm-ascend**；MoE routed hook seam = `MoECommMethod._apply_mlp`（不 hook 自由函数）。`moe-block` / `moe-routed-live-wiring.md`
- **program 个数 N 三档**（详见 notes/07）：N=1 整网融合（撞 const-fold 编译墙）＞ N≈few per-block 复用（撞 N≥6 运行时墙、~8 仍可能超）＞ N≈87 每层一个（必死）。whole-decode worker 用 multi-program `DistributedWorker #1706` + **resident `DeviceTensor` 跨 dispatch 串 residual/KV**，不 inline 45 层 body。`project_whole_model`
- **零拷贝 KV-IPC**：45 层合一 buffer → 1 key → 90 VA-map → `DeviceTensor(peer_base+offset)[block]` + `child_memory=True`；**forked chip 的 IPC import 必须在 child 进程 context 内**（父 import 的 ptr 在 child 非法读 0）。`zero-copy-ipc-integration-route.md` / `phases/22`

---

## 动手前 checklist（强制自检）

- [ ] 方案有没有违反第 0 层九条核心不变量？（尤其 512B shape 对齐 / shard×rank 恒等 / ISchedulerLayer 递归 / 双条件回收 / function group 同 cluster / 零 Python 关键路径）
- [ ] 单卡 ST/UT 用了 `apply_perrank_patch`（TP=8 slice）而非 `apply_tp1_patch`？
- [ ] 新 kernel 的 chunk 是固定常量、不跟 slice 走？tile 行字节 32B 对齐？没有 `[N,1]` slice / 裸 for / `pl.range(常量)` unroll？
- [ ] 多卡：三件套版本齐？EP barrier 用 AtomicAdd + 对称 a2a？没有 `-9` 强杀 / `npu-smi reset`？
- [ ] 精度：oracle 是 vLLM dump（非 synthetic）？W8A8 两处 INT8 quant + shared 不 quant + router_bias BF16 + EPS=1e-5？没有 silent fallback mask？
- [ ] 集成：整网在 pypto、vLLM 只调度？monkey-patch 整模型非 per-layer？enforce_eager？
- [ ] 流程：改前 audit divergence？改后 stale pyc touch？push 用 HTTP/1.1？launch 前 pkill+pgrep+npu-smi？
- [ ] 遇 507018/507899 先查 wiki + V0 定位，没在 work-around 绕过根因？

## 详细指南 / 出处索引

- 设计概念：`pypto_top_level_documents/`（machine_hierarchy / tensor_valid_shape / sharded_tensor / HL_* / arch-docs 08-design-decisions·10-known-deviations）
- 编程/坑：`pypto-lib/docs/{known-pypto-pitfalls,pypto-coding-style,performance-tuning,compile-runtime-workflow,debugging}.md`
- 项目部署/流程：`pypto-project/{CLAUDE.md,STATUS.md,blockers.md,deployment/*,phases/*,architecture/*}`
- 串讲笔记：`pypto-project/notes/06`（编程与部署 API）、`notes/07`（per-layer/block/整网融合 + program 个数 + 性能）
- 经验记忆：`~/.claude/projects/-data-chensiyu-hw-project-pypto/memory/{feedback_*,project_*,moe_*}.md`
