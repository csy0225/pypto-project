# NEXT SESSION — N=1 整网 W8A8 端到端**精度对齐** vs vLLM（Blocker B 已解，进入精度阶段）

> 直接把最底部 code block 当第一条消息粘贴。自包含。更新于 2026-07-12（本 session 尾）。
> **运行环境：0234 机器，通过本地 tmux `pypto-ascend-0:0` 登陆**（8 卡 0-7；781GB RAM；driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1）。
> 编辑机 `b-csy-develop`（无 python，NFS 与 0234 共享，编辑即时可见）。分支 `pypto-lib feat/whole-net-n1-fusion`。

---

## ⛔ 用户硬约束（不可违背，勿走弯路）

- **必须用 IPC 共享显存机制**做端到端，**KV cache 和权重都走 IPC**。**不许 H2D 绕路**、不许换非-IPC 方案。（权重+KV IPC 已 device 跑通，见下。）
- **必须用真实权重加载**（真 W8A8 checkpoint，非 dummy）。**真权重调试，不走其他弯路。**
- 遇到问题只能**解决它**，不能绕开（work-around）。诊断脚手架只能定位、不能进产品路径。
- **correctness 和 speed 都要**：既要跑出正确结果、也要推进到底完成目标；别用"correctness"当借口停在半路，也别为"快"造出错误的精度数字。
- **对齐 DeepSeek/Qwen**：遇问题先看 DeepSeek v4/Qwen 实现 + 历史开发文档，尽量对齐；step3p5-vs-DeepSeek 差异必须论证（只在"性能更好"时保留）。
- **架构优先**：coding 前先系统分析 + 整体设计。**严格遵守 SKILL.md**（`pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`）；不满足约束可能是设计不合理需重设计。
- **⚠ 历史文档可能 stale，先核对当前代码再下结论**（本 session 就踩过：attention_full.py "Phase 15 BYPASS" 注释已 stale，实际 head-gate 已 landed）。

## ⭐ 一句话状态（2026-07-12）

**本 track = N=1 整网融合（offline）**：`whole_decode_faithful_real` —— **全部 45 层（42 MoE + 3 dense/swa）内联进一个 `@pl.program`**，真 W8A8 权重经 **IPC** 加载，harness `tests/step3p5/_stage_whole_faithful_real_ipc.py`，分支 `feat/whole-net-n1-fusion`。
> ⚠ **别和另一个 track 搞混**：`NEXT-SESSION.md` 是 **G5b / per-layer 逐层 golden + live vLLM token-exact** track（harness `_stage_whole_decode_run.py`，未 commit 在 0162 working tree，本分支无此文件）。两者都朝 INT8-native 收敛但**目标/harness/分支都不同**。

**功能 bring-up 已完成（device 验证 + 推送）**：
- **Blocker B 解除**（`4bede85`）：全 42 MoE 层真 W8A8、**权重+KV 双双走 IPC**（`--kv-ipc`，`c61046b`）、8 卡 → `REAL_WEIGHT_IPC_RUN_CLEAN` ~3.4s，无 stall。根因是 `gate_topk` mrgsort 级联 bug（**非**文档旧假设的 IPC-VA 冲突，已 device 证伪），已对齐 DeepSeek format1 链修复。
- **gate_topk 修复 device 数值验证 PASS**（`_probe_gate_sort` `b92031f`，vs torch.topk，7.33s）。
- **head-gate 确认已 landed**（不是旁路——旧注释 stale，已更正）。

**⭐ 现在进入：整网 token-exact 精度对齐 vs vLLM。头号阻塞见下。**

## ✅ 头号精度阻塞已解（2026-07-12）：monolithic 整网 per-layer gate_r —— on-device head-gate（路径 a）

**已解**：`matmul_acc N=16` 丢 K 累加的 codegen bug（当年把 head-gate 移 worker 的原因）**现栈已修**（device probe `_probe_matmul_acc_n16` PASS + full-chain `_probe_head_gate_full` PASS）。据此在 `attention_full.py` + `attention_swa.py` **Scope 1.f 恢复 on-device head-gate**：`gate_logits = normed_all @ w_g`（K-chunk matmul_acc, N=16）→ `sigmoid` → `gate_exp = gate_score @ R`（N-chunk）；Scope 3.a o_proj 乘 `gate_exp`。`gate_r` 槽改承载 **block-diag R 常量**（`R[h,h*HEAD_DIM+d]=1` 实头，**layer-independent → 喂一次全 45 层通用**），每层从自己 `normed_all` 自算 gate → **monolithic 整网可 token-exact**（不再需 per-layer dispatch / resident-DeviceTensor 喂 gate_r）。对齐 vLLM `modeling_step3p5` L489 + L527-531。harness 填 R（实头=HQ//HEAD_DIM：full=8/swa=12）。`whole_decode_faithful_real` **TP=8 COMPILE OK**（attention inline 从 `._func` 重导，无需 regen）。memory `n1_head_gate_ondevice_restored_l1_nan` / `step3p5_head_gate_uses_normed_hidden`。

## ⛔ 新头号阻塞（L1 暴露）：整网 attention/decode 的 pre-existing NaN

L1 ctx=1 A/B 首跑：pypto worker `--hidden-token 6127 --kv-ipc` **RUN_CLEAN 3.59s 但 `next_hidden=nan / logits=nan / argmax=0`**（vLLM golden：tid=6127「北京」→ next=**303**「，」）。**NaN 不是 gate**——`w_g` padding 已 zero-pad（`weight_loader._slice_g_proj` L594）→ sigmoid(0)=0.5 finite；KV pool `torch.zeros`（`pypto_weight_ipc.export_from_checkpoint` L394）。**真相**：旧代码 dummy `gate_r=0` 让 `attn_out*0=0` **静默屏蔽了 attention 里 pre-existing 的 NaN**（SKILL 禁止的 silent-mask）；恢复真 gate 后 NaN 流出。疑似 `g5b_swa_multientry_kv_nan_root_cause` 同族，但本 harness seq_lens=ones（16 行全 ctx=1），那条 seq_len=0 路径不应触发。

**下一步**：per-layer golden bisect 定位首个 NaN 层/算子（faithful 单-dispatch 无逐层输出 → 加逐层 dump 旋钮，或用 G5b `_stage_whole_decode_run.py --golden-decode-pos` 另一 harness）→ 修 → 重跑 L1（tid 6127 期望 303）。

**bisect 结果（2026-07-12）—— NaN 定位到 MoE 路径，attention 干净**：`P_FAITHFUL_MOE_LAYERS=0`（仅 3 dense/swa attention 层，0 MoE）跑出 **FINITE**（`next_hidden=502.0 / logits=9.03 / argmax=27527`，无 NaN）→ **attention 路径（含恢复的 on-device head-gate）确认干净**；full-run NaN 源在 **42 层 INT8 W8A8 routed-MoE**（gap-5 territory）。head-gate 改动已确认正确。**bisect 二**：`P_FAITHFUL_MOE_LAYERS=1`（3 attention + 1 MoE）→ **NaN**，即**单个 INT8 MoE 层就复现**（非跨层累积；MoE 输入 post-norm O(1)，非 502）。→ 根因在 INT8 routed-expert 计算（gap-5，whole-net 内联 MoE 与 standalone moe.py INT8 kernel decoupled）。

**下 session 从这里继续**：8 hold-mode exporter + `--reuse-exporters`（免 15min 重载，每次 ~5min）；对单层 MoE 加内部 dump 定位算子（routed grouped-GEMM INT8 dequant / shared-expert / input per-token quant，疑 cast→INT8→cube gap-5 链）→ 修 → 重跑 L1（tid 6127 期望 argmax=303）。

## 🎯 精度对齐方式（三档，从易到难）

**L0 单卡算子（已做）**：`_probe_gate_sort`（gate_topk sort vs torch.topk，PASS）。gate_matmul 单卡 unsliced 会 Mat/Vec 溢出（`test_gate.py`，pre-existing，单卡 shape 铁律），验 gate 用 sort-only probe。

**L1 ctx=1 单 token device-vs-vLLM（脚手架已建，gated on head-gate 路径 a/b）**：`_stage_whole_faithful_real_ipc.py --hidden-token <id>` 已把 `embed(token)` 灌进 `current_hidden` row0 + pos-0 identity rope（cos=1/sin=0）。原理：1-token prompt vLLM 首 token = argmax(lm_head(hidden(pos0)))，等价 ctx=1 self-attn（rope pos0=identity），**不需 prefill KV / KV bridge**。流程：`vllm serve` 起 oracle → `curl /v1/completions prompt=[token0] max_tokens=1 temperature=0` 拿 golden token → kill vLLM 腾卡 → 起 exporters + `_stage_whole_faithful_real_ipc --hidden-token token0` 拿 pypto argmax → 比对。**但 L0 之后 head-gate 仍 dummy gate_r → 必须先解 per-layer gate_r（路径 a/b）才 token-exact。**

**L2 整网 token-exact / decode-step golden（终极）**：多 token 需 **vLLM→whole-net KV bridge**（vLLM 分页 KV 池 → 整网 flat KV，见 memory `g5b_kv_bridge_not_pure_reshape` / `g5b_kv_is_bf16_not_int8`(KV=bf16 1head/rank) / `g5b_import_ipc_facade_missing`(pure-python CTRL_IMPORT_IPC 已 device 证)）。或 **live A/B**（8001 pypto 整网 vs 8000 vanilla，co-tenancy `SIMPLER_COMM_NO_HCCL=1`，见 memory `project_g4_cotenancy_hccl_conflict`）。**这套是 G5b track 的机器（0162 working tree，本分支需 port）。** 判据：L1 per-layer hidden `ratio_allclose(atol=0.04)`；L2 logits cos≥0.999+topK overlap≥4/5；L3 greedy top-1≥95%。**oracle = vLLM eager dump，synthetic golden 会 stale。**

## 🖥 环境 / vLLM oracle 启动（本 session 验证可用）

- **三件套激活**（每 fresh shell，`activate.sh` 不带 CANN env）：
  `source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib`（`WS=/data/chensiyu/hw_project/pypto/workspace`）。
- **vLLM W8A8 oracle（0234 可跑，本 session 验证）**：
  `vllm serve <W8A8ckpt> --served-model-name step3p5 --trust-remote-code --quantization ascend --tensor-parallel-size 8 --enable-expert-parallel --enforce-eager --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85`
  → health=200，greedy chat golden 可得。占 8 卡 0-7，~5-6min load。
  - **⚠ acl 坑**：vLLM(`vllm_ascend`) 要 `import acl`（在 `/usr/local/Ascend/cann/python/site-packages`）。**跑 vLLM 前不要 export pypto 的 PYTHONPATH**（`$WS/pypto/python:...` 会 shadow 掉 CANN site-packages → `ModuleNotFoundError: acl`）。先 `source cann/set_env.sh`（它把 acl 加进 PYTHONPATH），**不** export pypto PYTHONPATH，再 `vllm serve`。
  - vLLM 与 pypto **同卡** → 要么先 vLLM 出 golden 再 kill 腾卡跑 pypto（offline A/B），要么 co-tenancy `SIMPLER_COMM_NO_HCCL=1`（live A/B）。
- **W8A8 ckpt** = `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`（arch `Step3p5ForCausalLM`，45 层，embed=`model.embed_tokens.weight` 在 shard 00048，非量化）。
- **8 卡 pypto env**：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
- **exporters（IPC 权重/KV）**：8 个 `--export-rank r --dev r` 常驻 hold（~15min cold load，jfs warm 快）；worker 用 `--reuse-exporters` 秒级 attach（免重载）。**exporter survive worker 的 force-reset**（分开进程），但 reset 后 pool 可能失效——保险起见重跑一轮。

## 🐞 Debug 方式（本 session 验证有效）

- **数字 device error 先查 [wiki Device-Error-Codes_zh](https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh)**，别凭空猜。`507018` 是泛化 host 码，看 `orch_error_code`/`sched_error_code`/`sub_class` 定真因。
- **device stall 快照（本 session 定位 gate_topk 的关键）**：
  1. harness 里 `logging.getLogger("simpler").setLevel(15)`（→ simpler info_v=0，放开 `LOG_INFO_V0`）。
  2. `export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_PROCESS_LOG_PATH=<预建目录>`（device slog 落文件；否则不写盘）。
  3. 读 `<dir>/debug/device-*/device-*.log` 找 `log_stall_diagnostics`：`TASK ... state=RUNNING kernels=[aic:-1 aiv0:N] running_on=[core=X(aiv0)]` = 卡住的 kernel+核；`SUMMARY completed=c/t`。root-owned，用 tmux(root) grep。
  4. `orch_error=8`=`TENSOR_WAIT_TIMEOUT`（producer 永不完成/kernel hang），`sched=100 detail=1`=S1 running-stalled。定 kernel 后对 `next_levels/<orch>/orchestration/chip_orch.cpp` 的 `rt_submit_*(task_id, ...)` 数 task 号→kernel。
- **单算子精度 probe 范式**（避开整网/matmul 溢出，隔离验证）：module-level `@pl.jit` + `from golden import TensorSpec, run_jit, ratio_allclose, topk_pair_compare` + torch golden_fn（见 `_probe_gate_sort.py`）。⚠ `@pl.jit` 的 shape/const 注释必须 module-level（closure-local const 报 `Undefined variable`）；`torch.full(FP32_NEG_INF)` 溢出，用 `-1e30`。
- **VA 布局诊断**（本 session 加的，comm_hccl.cpp:703 `domain_alloc_via_ipc` 打 `[base,base+size)`）：证明 comm window 不撞 IPC 池。若再疑 VA，用它 + 校验 `pypto_weight_map.rank0.json`（offset 512 对齐/无重叠/在池内）。
- **launch 前**：`pkill -f '[_]stage_whole'` + `rm -f /tmp/n1_weight_ipc/{STOP,*.rank*}`；**禁 `-9` 强杀 device 进程 / `npu-smi set -t reset`**（netboot 机锁死全卡）；`npu-smi info -t usages -i <c>` 确认 HBM<10%。stale pyc：monkey-patch 后 `find models/step3p5 -name '*.py' -exec touch {} +`。
- **每次 device run 慢**（exporter 全 ckpt load ~15min；compile 42 层 ~4min）。

## ⭐⭐ 铁律（勿再踩）

1. **单卡 ST/UT shape**：`apply_perrank_patch()`（保 TP=8 per-rank slice），不用 `apply_tp1_patch()`。gate/gate_matmul 是 replicated（全 288 expert），单卡跑 gate_matmul 会 Mat/Vec 溢出——验 gate 用 sort-only probe。
2. **gap-5 坑**：in-kernel `pl.cast(bf16/fp32,INT8)` 喂 cube 可能静默错；照抄 DeepSeek cast 链 + create_tensor 位置 + scope。
3. **ccec ND2ND**：scale slice 必须 contiguous row-slice + reshape；a2a3sim compile 过 ≠ device ccec-clean（必须真 device 跑）。
4. **push**：PAT `/data/chensiyu/secrets/github.env` + `git -c http.version=HTTP/1.1`，输出屏蔽 token。pypto-lib 的 `.git/objects` 是 root-owned（worker 以 root 跑过）→ commit/push 走 tmux(root)；pypto-project 是 chensiyu-owned，可直接。跨仓 push 同步 STATUS pin。
5. **文档 stale 风险**：核对当前代码再下结论（head-gate 就栽在 stale 注释）。

## 本 session commits

pypto-lib `feat/whole-net-n1-fusion`：`4bede85`(gate_topk mrgsort 修复→Blocker B 解除) `c61046b`(KV via IPC `--kv-ipc`) `b92031f`(gate_sort device probe PASS) `<head-gate 注释更正+ctx=1 A/B 脚手架>`。
pypto-project `main`：`e9af803`/`09a4e11`/`9f12dac`/`665431c`/`326d94a`(STATUS/blockers/NEXT-SESSION 更正)。

---

```
继续 step3p5 **N=1 整网 W8A8 端到端精度对齐 vs vLLM**（本 track = whole_decode_faithful_real，45 层内联进一个
@pl.program，真 W8A8 权重+KV 经 IPC，harness tests/step3p5/_stage_whole_faithful_real_ipc.py，分支 feat/whole-net-n1-fusion）。
⚠ 别和 NEXT-SESSION.md 的 G5b/per-layer 逐层-golden+live track 搞混。用户硬约束：IPC(KV+权重)+真实权重、不走弯路、
correctness>speed、对齐 DeepSeek、架构优先、严格遵守 pypto-dev-constraints SKILL、历史文档可能 stale 先核对代码。

功能已通（device+推送）：Blocker B 解除（gate_topk mrgsort 修复，非 IPC-VA 冲突）；权重+KV 双 IPC 8 卡 device-clean；
gate_topk device 数值验证 PASS；head-gate 确认已 landed（旧"BYPASS"注释 stale）。

⭐ 头号精度阻塞 = monolithic 整网的 per-layer gate_r：head-gate 靠 worker 预算 gate_r=expand(sigmoid(RMSNorm(hidden_L)@w_g_L))
逐层喂入，activation-dependent；45 层塞一个 dispatch，caller 只能算对 L0，L1-44 喂 dummy → 除 L0 外 head-gate 全错 → 不可能 token-exact。
两条解法：(a)【优先，最贴合 N=1】先验 N=16 matmul_acc codegen bug 现栈是否还在（当年因它把 head-gate 移 worker；写 N=16 matmul_acc
probe vs torch，仿 gate.py:138-149），若已修则把 on-device gate_logits+sigmoid+block-diag-expand 加回 attention(+swa)→整网自算 per-layer
gate_r→token-exact；若还在则修 matmul_acc(N=16) 上游或换避开小-N 写法。(b) per-layer dispatch(resident-DeviceTensor,SKILL §H,G5b 路子)。

精度对齐三档：L0 单算子 probe（gate_sort 已 PASS，仿 _probe_gate_sort.py）；L1 ctx=1 单 token A/B（--hidden-token 脚手架已建：
embed(token)灌 current_hidden row0+pos0 identity rope，vLLM 1-token prompt 首 token 作 golden，不需 KV bridge，但 gated on head-gate 路径 a/b）；
L2 整网 token-exact（多 token 需 KV bridge / live A/B，G5b 机器需 port）。判据 L1 ratio_allclose(0.04)/L2 cos≥0.999/L3 top1≥95%，oracle=vLLM eager dump。

环境：0234 tmux pypto-ascend-0:0（8 卡）；三件套 source cann/set_env.sh+activate.sh+PTO_ISA_ROOT+PYTHONPATH。
vLLM oracle：source cann/set_env.sh（勿 export pypto PYTHONPATH，否则 import acl 失败）→ vllm serve <W8A8ckpt> --quantization ascend
--tensor-parallel-size 8 --enable-expert-parallel --enforce-eager --trust-remote-code --port 8000（占 8 卡，先出 golden 再 kill 腾卡跑 pypto）。
debug：数字 error 先查 wiki Device-Error-Codes_zh；device stall 快照 = setLevel(15)+ASCEND_GLOBAL_LOG_LEVEL=1+ASCEND_PROCESS_LOG_PATH→
读 device-*/log 的 log_stall_diagnostics 找卡住 kernel/核。禁 -9/npu-smi reset；push 走 HTTP/1.1+PAT（pypto-lib root-owned 走 tmux root）。
全部读上面 ⛔/⭐/🎯/🖥/🐞/铁律。每完成关键节点更新 pypto-project STATUS/blockers/NEXT-SESSION + push（同 session）。
```
