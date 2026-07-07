# MoE 单块 8 卡精度收尾 + vLLM+pypto 集成后续工作（含用户约束铁律）

> 2026-07-06 落。承接 `troubleshooting-moe-block-8card-gate-topk.md`。
> 本文是**后续工作的入口 + 用户明确要求的约束铁律**。任何续接会话先读本文 + STATUS.md。

---

## 0. 用户约束铁律（本会话明确提出，必须遵守，不得违反）

1. **整网执行在 pypto**：step3p5 的**整网执行**都放在 pypto 上，vLLM **只负责调度和
   管理 KV cache** 等功能。不要偏离这个目标。
2. **准出标准**：**端到端验证通过 + 精度验证通过**（对齐 vLLM）。两者都要过才算完成。
3. **不要做重复的工作**：复用已验证的组件（gate 修复、barrier-mesh tp_all_reduce、
   shared 路径、torch 参考 harness），不要重造。
4. **507018 先查 wiki 定位**：遇到 507018 先用
   <https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh> 定位
   （`sched_error_code` 分类 + V0 设备日志 + `stuck_task_id`/kernel func_id 映射），
   不要凭猜。
5. **不许绕过 / 不要 work-around**：遇到问题**根因修复**，不能绕过去。诊断脚手架
   （如 `EPMOE_BYPASS_GATE`）只能用于定位，**不能作为产品路径**。
6. **gate_topk 是 MoE 的一部分，不能 bypass**：gate + top-k 必须在 pypto 上算
   （vLLM 不接管路由）。（已按此修复 gate_topk，未 bypass。）
7. **用和 DeepSeek 一样的 push 方式，不能用 pull 重写**：EP dispatch/combine 保持
   DeepSeek 式 push（`pld.tensor.put` / `remote_store` + barrier），**不要**改成 pull。
8. **排查四板斧**：遇到 kernel 问题依次看 —— (a) DeepSeek 同一栈是怎么做的、为什么它没问题；
   (b) 上游（pto-isa / PTOAS / simpler）是否已有针对性修复；(c) kernel 逻辑是否有 bug、
   必要时**自己写 kernel**；(d) 是否是数据类型问题。
9. **协作**：可用 agent team / 反向（红队）agent 从不同角度一起看。
10. **push**：用 PAT `/data/chensiyu/secrets/github.env`，HTTP/1.1，输出屏蔽 token，
    不落 `.git/config`。（fork SSH 在 0162/本地都 publickey 失败；0162 无 GitHub 权限 →
    走 `git bundle` → 本地 → HTTPS(PAT) 推送。）

---

## 1. 当前状态（2026-07-07 更新）

| 项 | 状态 |
|---|---|
| gate_topk 8 卡死锁（507018/sched=100） | ✅ 真解决（DeepSeek 式 format1 mrgsort 链） |
| shared expert 路径数值正确 | ✅ 对 0.12% torch 参考 PASS（含 barrier-mesh tp_all_reduce） |
| routed 路径精度 | ✅ **已解决**（根因=a2a 数据可用性；symmetric fixed-slot 布局修复；device local_routed_x/y/routed_y_buf 全 0.00%） |
| 全 moe_out vs ffn_out（silu 层 L3/L4） | ✅ **PASS**（on-device gate + shared+routed vs vLLM ffn_out，ratio_allclose atol=0.04；L3 swa_moe / L4 full_moe device PASS） |
| swiglu 层 L43 (swiglu7_silu) 精度 | ✅ **PASS**（routed per-token INT8 量化 interm+input；device moe_out vs vLLM ffn_out ratio_allclose atol=0.04；commit `3b236e6`；见 §6） |
| swiglu 层 L44 (swiglu7_swiglu16) 精度 | ✅ **PASS**（两个叠加 bug 均已修：① shared swiglu16 clamp 在整宽 `[16,160]` Vec tile 上被误编译→拆 5 个窄 `[16,32]` chunk，commit `ca5e0b8`；② **gate router_bias FP32-vs-BF16 dtype**——vLLM 用 BF16 跑 router_bias，loader 给 FP32，bias~4.79 主导分数、其 ~0.015 BF16 舍入决定 top-8 尾部→在 gate 内 BF16 round，commit `5b85b34`。device full ffn_out on-device gate PASS 19.22s；CPU topk(sigmoid+BF16(bias))==vLLM 0/16。**非 kernel/sort/precision bug**——vbitsort 是 FP32-exact，gate_w/logits/recovery 全对） |
| gate on-device 精度（全 MoE 层）| ✅ **PASS**（router_bias BF16 修复后：L44 swiglu16 PASS、L43 swiglu7 PASS(21.10s，之前 15%)；silu 层回归验证中）。教训：所有 kernel 侧改动 byte-identical 时，应怀疑**错误的输入**（dtype/加载），而非 kernel。 |
| 端到端（whole-decode 串联 / live A/B） | ⏸ 剩余多周工程（live KV-IPC + 权重驻留 + single-handoff backend），且依赖两条分叉分支的合并决策 |

### ⭐ 2026-07-07 routed 精度根因 + 修复（device 验证）
- **根因（device INT32 offset dump 确认，非之前猜测的 if-predicate）**：`pub_counts` 只把
  count-to-dst 通过 `notify(Set)` publish 到目标 rank dst 的 window → 每个 rank 本地只有
  column-my_rank 有效；`ep_all_to_all` 的 `read_offsets = Σ_{d<my_rank} pub_counts[peer*N+d]`
  累加了 column d<my_rank（在 rank≠0 上全为 0）→ 接收方从 peer 的 send_buf row 0 读取 →
  rank≠0 拿到错误 token。rank0 因空和恰好正确，掩盖 bug 多个会话（历次验证都只看 rank0）。
- **附带确认的两个 pypto/Ascend 行为**：bare-index `pl.read(arr,[my_rank])` 正常工作；
  `pl.cast(<bool 比较>, INT32)` 在 Ascend 上 TRUE = **−1**（不是 +1，全 1 符号扩展）。
- **修复**：symmetric fixed-slot a2a 布局（send dst-block @ `dst*MAX`，recv src-block @ `src*MAX`，
  pull 读 peer @ `my_rank*MAX`，`MAX=T*TOPK=128`，`LOCAL_RECV_MAX=8*128` 精确）。`read_off` 为
  compound-scalar，**不依赖任何 cross-rank column 数据**。全部索引为 loop-var 或 compound-scalar。
  含此前 expert-GEMM fillpad+full-tile 修复。
- **验证（gpu-a910x-0162，8 卡 real W8A8）**：L3(swa_moe) / L4(full_moe) `moe_out` vs vLLM `ffn_out`
  DEVICE PASS（on-device gate mrgsort，非 bypass）。
- **提交**：`moe.py` commit `e82958c`（base 956aede），推送至 `csy0225/pypto-lib` 分支
  **`wip/moe-symmetric-a2a-fix-20260707`**（bundle→本地→PAT/HTTPS）。
- **⚠ 分支分叉（需决策）**：fork `stepfun/develop`（`63dee39`）走的是 co-resident per-rank worker 路线
  （8 commits，含 pypto_moe_backend / pypto_mlp_worker routed op / MoE routed backend hook，
  merge-base `2df96138`）；0162 这条（956aede: gate mrgsort + expert kernel + a2a 修复）是 EpTpMoE
  MoE-block / Option-C 路线。两者互非祖先，**未 force-push**（推到新分支避免覆盖）。live 集成所需
  backend/worker infra 在 fork 那条线上 → **live 整网集成前需先合并两条线（用户决策）**。

代码：`csy0225/pypto-lib` 分支 **`stepfun/develop` = `bb9e683`**（2026-07-07 推送）——已把验证过的 swiglu L43 修复（commit `3b236e6`：routed input+intermediate per-token INT8 动态量化）**合并**到 backend/worker 线（`9336133` = 含 `pypto_moe_backend`/`pypto_mlp_worker` live 集成 infra 的 `63dee39` merge）之上。**两条线已合并、无丢失**（文件不相交：`moe.py` vs `tools/step3p5/*`，merge-base=`e82958c` a2a 修复，clean FF 推送非 force）。分支分叉决策已由用户拍板解决。
harness：`pypto-lib/_stage_moe_block_precision.py`（`--target ffn_out` on-device gate 全量验收；
`--bypass-gate`/`--torch-golden`/`--zero-routed`/`--zero-shared`/`--dump-stages` 诊断）。
可信参考：vLLM `ffn_out` dump（device 全量验收）+ torch BF16（对 ffn_out 差 0.12%）。

---

## 2. 下一步工作（按顺序）

### 2.1 定位并修复 routed 精度（当前唯一硬 blocker）
**已排除**（用 0.12% torch 参考）：act-quant、gate、权重、`moe_parts` dump（不可靠）、
dispatch 读偏移（no-op）、combine push 原语 remote_store→tensor.put（no-op）；
dispatch/combine 行序索引逐行审计一致。→ bug 在 `_expert_routed` 分块 grouped-GEMM
（`pl.parallel`+RECV_TILE=32+`pl.spmd`+`valid_shape`）或 gather，读代码定位不出。

**决定性做法 = 逐级设备 dump**（遵守约束 5「不绕过」、约束 8「自己写 kernel/查 DeepSeek」）：
1. 给 `EpTpMoE.chip_orch`/`host_orch` 加**调试输出** `local_routed_y`（combine 前每
   expert 输出，[LOCAL_RECV_MAX,HIDDEN]）；harness 加 spec + 算 torch per-expert 参考
   （`SwiGLU(x[t]@wg_r[eid])@wd_r[eid]`，按 dispatch 到该 rank 的 token 顺序）。
2. `local_routed_y` 发散 → grouped-GEMM bug（对照 DeepSeek `expert_gate_up`/`expert_down`
   的分块/spmd/valid_shape 写法，约束 8a/8c）；`local_routed_y` 对但 `moe_out` 错 →
   gather 加权/索引 bug。
3. 也可先 dump `local_routed_x`（dispatch 后）确认到达每个 expert 的 token 正确。
4. 修复后：`--zero-shared --torch-golden` routed 隔离 PASS → 全 `moe_out` vs `ffn_out` PASS。

⚠️ 该 dump 是签名改动（chip_orch/host_orch），**在低上下文时干净做**，避免破坏已提交的
gate/shared 修复（基线 = 956aede，可 revert）。

### 2.2 whole-decode 整层串联（#10）
routed 精度过后：attention 程序 + `EpTpMoE` 块顺序执行，单进程跑完 45 层（vLLM idle）。

### 2.3 逐层 device 精度（#11）
逐层对 vLLM dump 比对（先 MLP/MoE 层；attention-core 受 dump 无 KV 限制单列）。

### 2.4 整网 backend + live A/B（#13 / #14）
single-handoff whole-model backend → 8001（pypto 整网）vs 8000（vanilla oracle）
token 级对齐（temp=0 多 prompt）。

---

## 3. 环境 / 运维铁律（本会话踩坑，续接必读）
- 验证机 `gpu-a910x-0162`，cards 8-15，CANN 9.0.0 non-GA；8001 oracle 在 cards 0-7。
- **禁止 `npu-smi set -t reset`**（AMP+HCCS netboot 会重启全部 16 卡 → SSH-key 抹除 → 锁死）。
- **勿 `-9` 强杀 device 上的进程**（无 finalize → card poison → 下一次 507018）；等
  finalize 的 `aclrtResetDeviceForce` 跑完再重启。
- 三件套激活：`source cann/set_env.sh && source workspace/activate.sh && export PTO_ISA_ROOT=...`。
- monkey-patch/flag 测试后清 pyc：`find models/step3p5 -name "*.py" -exec touch {} +`。
- 8 卡 harness 每轮加载 8×~47GB W8A8 bundle（~5 分钟），慢；耐心等 Monitor。
- V0 定位：`logging.getLogger("simpler").setLevel(15)`（在 worker.init 前）+
  `ASCEND_PROCESS_LOG_PATH=<预建目录>`；停机快照 kernel id → 该 build 的
  `chip_orch/kernel_config.py` func_id。

---

## 4. 诚实边界
本会话 gate_topk 死锁真解决、shared 验证正确并入库；**routed 精度（41.8%）未通过**，
是已隔离的 open item（不是伪造通过）。端到端与整网 live A/B 依赖 routed 精度先过。
下一步（逐级 dump）确定、无重复工作、符合全部用户约束。

---

## ⭐ 5. 2026-07-07 会话 — 根因确认 + 下次 SESSION 直接从这里开始

**用系统性「逐 rank × 逐 stage 设备 dump」把多会话遗留的 MoE routed ~41% bug 确定性定位到单一阶段。**

### 已确定修复 ✅（rank0 设备验证，production-ready）
**专家 grouped-GEMM**：根因 = `_expert_routed` gate_up matmul 在 partial tile（tile_valid<RECV_TILE=32）上用**动态 `valid_shape`** → Ascend cube 16×16 fractal 把无效行混进有效行。
- **修法（DeepSeek 对齐，纯 model 侧）**：去掉 4 个 matmul 输入 slice 的 `valid_shape=[tile_valid,...]`（读满 32 行）+ gate_up 加 `gated_m = pl.fillpad(gated_v, zero)`（新变量，不能重赋 gated_v）。
- 结果：rank0 `local_routed_y` **0.00%**，`moe_out` 41%→26%。stage 在 `0162:/tmp/apply_expert_fix.py`（应用到 clean baseline）。

### 🎯 真正根因（唯一剩余 blocker）
**`ep_all_to_all`（moe.py:305-382 的 a2a pull）对「非 0 号接收卡」投递错 token**（rank0 对，rank≠0 错）。
证据（layer3, --bypass-gate, --zero-shared --torch-golden, cards 8-15）：

| stage | rank0 | rank1 |
|---|---|---|
| `recv_x`（a2a 输出，re-pack 前） | **0.00% 对**（验证了 ref 布局） | **68% 错** ← 根因 |
| `local_routed_x`（re-pack 后） | 0.00% 对 | 68% 错（传播） |
| `local_routed_y`（专家） | 0.00% 对 | 19% 错（传播） |
| `routed_y_buf`（combine push） | self 0/20 对 | cross 90/108 错（传播） |

**逐一排除**（全设备验证 no-op）：re-pack、专家 compute、combine（CORE_GROUP fence / put-vs-remote_store / zero-race / gather）、cross-store、src_route_table 3D→2D、DeepSeek combine 重写、上游 combine。就是 **rank≠0 的 a2a pull**。`ep_all_to_all` 的偏移公式（recv_offsets prefix-sum / read_offsets=Σ_{d<my_rank} peer→d / self-copy）逐行看都是 my_rank-无关正确 → 很微妙的 `my_rank≠0` codegen/runtime bug（同「看起来对、设备错」类）。

### 下次 SESSION 直接从这里开始（低上下文最稳妥）
1. **环境**：`ssh infra@gpu-a910x-0162...`；`cd $WS/pypto-lib`（WS=/data/chensiyu/hw_project/pypto/workspace）；`source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib`。cards 8-15 空闲，0-7 是 8000 oracle。**每次 device run 前 frontend-smoke `select_moe_block(3)`**（免卡，省 5 分钟权重加载）。
2. **决定性下一步**：把 rank1 的 `recv_counts / recv_offsets / read_offsets / send_offsets_rank` 作为 **INT32 dump 输出**（仿 harness 现有 per-rank dump 模式），对比手算期望 → 看运行时**确切的错误偏移值**。
   - 偏移值错 → 修 `dispatch_step`(moe.py 857-930)/`ep_all_to_all`(305-382) 的计算。
   - 偏移值对但 recv_x 仍错 → `pld.tile.remote_load` 对 rank≠0 mis-deliver = **上游 pypto/simpler bug**，提 issue（最小复现 = 这个 harness）。
3. 修好 a2a → 所有 rank `local_routed_x` 正确 → `local_routed_y` 正确（专家修复已验证）→ `moe_out` 通过 → **MoE 块精度闭环** → 推进整网集成（whole-decode 串联 + vLLM single-handoff）。

### 关键工件（都在 0162）
- **harness**：`_stage_moe_block_precision.py` 已插桩 `--dump-stages`：per-rank dump `recv_x`/`local_routed_x`/`local_routed_y`/`routed_y_buf` + `_mk_cmp(rank=, permcheck=)` best-match + BAD-ROW-SPLIT(self/cross)。参考布局都已在 rank0 验证（recv_x/local_routed_x rank0 = 0%）。
- **moe.py**：expert-fix + 4 个 dbg 输出（dbg_routed_x/y/ybuf/recv_x）。backups：`/tmp/moe.py.bak_prezerorm`（clean expert-fix，无 dbg）、`.bak_spillfix`（干净基线）、`.bak_prerecvx` 等。
- **完整证据 + 复现 + 每一步 dump 结果**：memory `moe_routed_bug_expert_partial_tile.md`（含 START HERE 段）+ backlog `task #4`。
- **agent team** `vllm-pypto-e2e`（reverse-review / hw-analyst / sw-analyst / upstream-scout）本会话产出：hw-analyst 的 cube fractal 机制（解开专家 bug）、sw-analyst 的 vllm_routed_experts + a2a 对称性分析、reverse-review 的 combine 审计、upstream-scout 的 pypto #1588。team session-scoped，下次重开。

### 诚实边界
MoE 块精度**未闭环**（a2a rank≠0 bug 待修），整网集成**未开始**（依赖此）。专家修复已确定。根因已确定性定位到单一阶段（ep_all_to_all a2a-pull for rank≠0），下一步明确（INT32 offset dump）。

---

## ⭐ 6. 2026-07-07（下半会话）— swiglu 层 L43/L44 精度 = routed 专家漏 per-token INT8 动态量化

> a2a routed bug 解决后，遗留 swiglu 变体层精度不对齐 vLLM。本节记录根因 + 修复 + device 证据。

### 根因（device 确认，非 clamp kernel bug）
- **不是 clamp bug**：device clamp == BF16-clamped-torch（L43 `--torch-golden --zero-shared --bypass-gate` `moe_out` PASS）。公式 `silu(gate).clamp(max=7)*up.clamp(±7)` 在 pypto device / torch / vLLM-ascend `SwigluStepAndMul` 三处逐字一致。
- **真因**：pypto routed 专家**漏了 vLLM W8A8 oracle 的 per-token INT8 动态量化**。oracle（`quant_apply_mlp` else 分支，`w1_offset` 未 thread → 非 antiquant）对 **输入 x 和 clamp 后的中间激活都做 `npu_dynamic_quant`**（per-token amax/127, round, ±127, dequant），再走 INT8 gmm。pypto 只做 FP32 silu/clamp → BF16 → BF16 matmul，两处量化都缺。
- **为何 silu 层过、swiglu 挂**：silu（40/42 层）中间激活分布平滑，INT8≈BF16，量化无损；swiglu 被 clamp 到 ±7 后是双峰分布（少数 ~±49、大量小值），per-token scale≈0.386 把小值量化成 0 → 27%+ 偏差，在 output ch4094 汇聚成尖峰（max\|diff\|=162）。
- **铁证**：pypto 自己的 `_moe_ref_dynamic(routed_w8a8_dynamic=True)`（量化 input+interm）对齐 vLLM ffn_out 到 **0.9995**（全 45 层含 L43/L44）。

### ⚠ 只量化 routed，**shared 专家不量化**（ckpt 验证 + 用户确认）
W8A8 index.json：`layers.{43,44}.share_expert.{gate,up,down}_proj.weight` = 3 keys、**0 个 scale/offset → 纯 BF16 无量化**；routed 专家 1728 个 scale/offset → W8A8。所以 shared（Step3p5MLP，独立于 FusedMoEBlock）**不做动态量化**，保持 clamp(swiglu16)+BF16（现状正确）。**不要给 shared 加量化**。

### swiglu 位置 + 中间 dtype（对齐 vLLM）
swiglu 夹在 gate_up（up）与 down 之间：`x → gate_up → swiglu → h → down`。vLLM：swiglu 本身 **BF16** 算，输出 h 再**量化成 INT8** 喂 down（INT8 gmm）。pypto：silu/clamp 用 FP32→BF16(h_bf16)，再 per-token 量化到 INT8 精度→反量化回 BF16→BF16 down（权重两边都反量化，等价；对齐 `_moe_ref_dynamic` 已验证 0.9995）。

### 修复（`moe.py` `_expert_routed`，DeepSeek `expert_routed.py:132-151` 模式）
- **interm-quant**（clamp 后中间激活）：`pl.spmd` **over tokens**（不能用 CORE_GROUP 读 vec bridge → 94.5% 乱码）；per-token amax over 全 INTER=1280；`pl.cast(x*127/amax, INT32, mode="rint")`（不能用 cast-to-INT8-round）；dequant→BF16。gated by 编译期 `_routed_swiglu_step`。
- **input-quant**（gate_up 前）：预算 per-token scale `[RECV_TILE,1]`（amax over HIDDEN，token-spmd），gate_up K-loop 内 on-the-fly 量化 x（materialize x_q[32,4096]=256KB 会爆 UB 188KB）。scale create_tensor 必须在 tile 级（同 h_bf16），否则 SSA "used outside defining scope"。
- 关键坑：`[N,1]` scale 用 reduction 产出（`row_max`+reshape）+ `row_expand_mul` 广播（RMSNorm/FA 已验证安全），不用 slice 出 `[N,1]`（撞 32B 对齐 fault）。

### device 证据（L43 `--target ffn_out --bypass-gate`，cards 8-15）
| 版本 | bad ratio | max\|diff\| |
|---|---|---|
| 基线（无量化） | 27.6% | 162 @ch4094 |
| v1/v2（CORE_GROUP + cast-INT8-round，两个 bug） | 94.5% | 乱码 |
| v3（interm-quant，spmd-over-tokens + INT32-rint） | **15.86%** | **2** ← 尖峰消除 |
| 隔离（device interm-q vs torch interm-q，routed） | **PASS** | — device 量化逻辑正确 |
| + input-quant（完整修复） | device 验证中 | — |

### 下一步
- L43 完整修复过 → L44（同 routed 修复 + shared 不量化，无需改代码）→ 整网串联 + vLLM handoff。
- 卡免 codegen 探针（省 5 分钟权重加载）：`ir.compile(select_moe_block(43), backend_type=_backend_for_platform("a2a3"), platform="a2a3")`。
- backups：`/tmp/moe.py.bak_predynquant`（clean 基线）、`.bak_preinputquant`（仅 interm-quant）。memory `moe_swiglu_missing_int8_requant.md`。

---

## ⭐ 7. 2026-07-07（续接会话 team `vllm-pypto-e2e`）— stepfun/develop 全仓回归 PASS + 验证过的 commit + 仓库对齐确认 + monkey-patch 归属决策

### 7.1 验证过的 commit 组合（本次回归确认，全部已在 fork stepfun/develop）

| 仓库 | 验证过的 HEAD | 内容 | fork 对齐 |
|------|--------------|------|-----------|
| **pypto-lib** | **`1a6c6342`** | MoE 全修复链：`956aede`(gate mrgsort)→`e82958c`(对称 a2a)→`93361336`(merge backend/worker 线)→`3b236e6`(routed INT8 动态量化)→`2b00bec`(shared swiglu16 clamp 5×[T,32])→`1a6c634`(router_bias BF16) | ✅ ls-remote 确认 = fork |
| **pypto** | `be90f992` | DeviceTensor.__getitem__ slicing + distributed_runner import_ipc glue（Phase 24 zero-copy） | ✅ 上次会话已 push；import_ipc 已 landed 上游 `edb0adf5` |
| **pto-isa** | `e25732f0` | 未改 | ✅ ls-remote 确认 = fork |
| **PTOAS(src)** | `da011a3d` / bin `v0.45` | 未改 | ✅ ls-remote 确认 = fork |
| **simpler(runtime)** | `1aa6efb4` | import_ipc device-IPC key import（`c236194`/rebased）+ comm PID-whitelist fix `25a0544` | ✅ 上次会话已 push；import_ipc 已 landed 上游 `c2361943` |

**回归证据（device，cards 8-15，`_stage_moe_block_precision --target ffn_out` on-device gate vs vLLM ffn_out，atol=0.04）**：L3 swa_moe silu PASS(20.4s) / L4 full_moe silu PASS / L43 swa_moe swiglu7 PASS / L44 full_moe swiglu16 PASS —— 覆盖全部 4 个 program-class 变体。红队 review：5 个修复无 CRITICAL/HIGH（routed INT8 仅 gate routed、shared 保持 BF16 无量化；无 silu 回归；无 [N,1] slice / 跨 rank 列读；5-chunk clamp 边界正确）。

### 7.2 仓库对齐确认（用户要求）
- **需同步 push 的：无**。5 仓验证过的 HEAD 全部已在 fork stepfun/develop（pypto-lib/pto-isa/PTOAS 经 ls-remote 直接确认；pypto/simpler 上次会话已 push 且 import_ipc 已 landed 上游）。
- **未提交项（非「验证过的 commit」，是 WIP 集成脚手架）**：pypto-lib 有 8 个 untracked 新工具（`whole_decode_compare.py` / `moe_block_specs.py` / `pypto_weight_ipc.py` / `vllm_dense_mlp.py` / `vllm_shared_mlp.py` / `pypto_dense_mlp_backend.py` + test + report）——整网集成用，待 e2e 跑通后一并 commit；pypto 有 `M runtime`（submodule 指针 bookkeeping）。
- **上游 divergence**：我们落后 origin/main 大量 commit（DSV4/Qwen3 驱动）；import_ipc 已 landed 上游。**re-foundation on origin/main 是后续清理项**（非本目标阻塞）。建议后续 bump ptoas-bin v0.45→v0.48（拉入 tile-addr 对齐 verifier #875 + identity-tmov clamp co-factor #876）。

### 7.3 整网 decode 集成新增 de-risk（2026-07-07）
- **整网 45/45 层 COMPILE PASS（Option C 解耦）**：融合 swa_moe 在 L3 编译失败（`attention_swa.py:479` EP 分布式 lowering 无法常量折叠 `pl.full([32−12,HEAD_DIM])`）；改用「TP-attention 程序（`_build_tp_attention_{swa,full}_program`）→ resid1 → `select_moe_block` EpTpMoE block」解耦，全网 45 层编译通过。
- **整网 dense 前缀 TP=8 device 运行 PASS**：L0 full_dense / L1,L2 swa_dense 8 卡 DEVICE PASS（rc=0）。
- **47.46GiB 单 key IPC = WORKS**：`aclrtIpcMemGetExportKey(48GiB) rc=0` + `ImportByKey` 同一 VA + 首尾 readback OK → live 权重零拷贝驻留最硬 gate 解除（splitter 不需要）。

### 7.4 目标聚焦：decode（先不动 prefill）
- **本阶段目标 = 跑通整网 decode**（vLLM 只调度 + KV）。**prefill 暂不动**（TASK-29 prefill MoE `moe_gate_up` 5MB L1 overflow 是 prefill-only，decode 路径不触发；后续再按 DSV4 prefill 模式移植）。
- decode 剩余（多周 live 工程）：whole-decode worker 用上游 `pl.submit`/multi-program DistributedWorker(#1706，已在 HEAD，解 §10 nesting + co-tenancy）串起 45 层 → 权重/KV 双 IPC pool（47GiB 单 key 已验证）→ wire `_pypto_full_forward`（当前 raise）→ live A/B 8001 vs 8000 token 级。离线串联受 dump 缺 KV 限制（attention-core 只能 live 验），真整网精度对齐须走 live single-handoff。

### 7.5 vLLM monkey-patch 归属决策（用户询问：挂 vllm-ascend 还是 pypto-lib）
**决策：monkey-patch 留在 pypto-lib（`tools/step3p5/vllm_monkey_patch.py`），经 sitecustomize（`make_vllm_sitecustomize.py`）注入 stock vLLM，不挂 vllm-ascend、不 fork vllm-ascend。** 理由：
1. **单一事实源**：patch 依赖的 pypto 模型程序 + backend infra（`pypto_weight_ipc`、whole-decode worker、`pypto_*_backend`）全在 pypto-lib，co-locate 便于与被依赖代码同版本演进。
2. **不背 vllm-ascend fork 维护**：vllm-ascend 上游变动极快（scout 报告 ~百级新 commit），fork 进去 = 沉重同步负担 + pypto 发布被耦合。
3. **零改动部署**：sitecustomize + PYTHONPATH/env 注入 stock vLLM，无需重建 vLLM，部署 + 回滚都轻。
4. **patch 面小而隔离**：只 hook `Step3p5Model.forward` / `Step3p5ForCausalLM.compute_logits`（版本敏感的仅是这两个符号名）；patch 机制（install/uninstall）与 pypto runner 已在 `vllm_monkey_patch.py` 内分离。
- **维护铁律**：(a) pin patch 针对的 vLLM-Ascend 版本；(b) patch 只读 `vllm.model_executor.models.step3p5` 符号，vLLM 改名时只更新薄 patch shim（不动 pypto runner）；(c) 若 vllm-ascend 后续提供一等 custom-backend/plugin 注册 API，则改走该 API 注册（比 monkey-patch 干净），但注册器 + runner 仍**从 pypto-lib 发布**（entry-point / sitecustomize 加载），仍不 fork vllm-ascend。
