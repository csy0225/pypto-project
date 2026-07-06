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

## 1. 当前状态（2026-07-06）

| 项 | 状态 |
|---|---|
| gate_topk 8 卡死锁（507018/sched=100） | ✅ 真解决（DeepSeek 式 format1 mrgsort 链） |
| shared expert 路径数值正确 | ✅ 对 0.12% torch 参考 PASS（含 barrier-mesh tp_all_reduce） |
| routed 路径精度 | ⏸ **未过，41.8% 符号翻转**，隔离到 `_expert_routed` grouped-GEMM |
| 全 moe_out vs ffn_out | ⏸ ~41%（由 routed 拖累） |
| 端到端（whole-decode 串联 / live A/B） | ⏸ 待 routed 精度过后 |

代码：`csy0225/pypto-lib` 分支 `wip/moe-gate-fix-20260706`（commit 956aede）。
harness：`pypto-lib/_stage_moe_block_precision.py`（`--bypass-gate`/`--torch-golden`/
`--zero-routed`/`--zero-shared`/`--target moe_parts_shared|moe_parts_routed`）。
可信参考：torch BF16（对 vLLM `ffn_out` 差 0.12%）。

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
