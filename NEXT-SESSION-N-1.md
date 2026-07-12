# NEXT SESSION — N=1 整网 W8A8 端到端集成收尾（IPC 真权重 + KV，解 Blocker B）

> 直接把最底部 code block 当第一条消息粘贴。自包含。更新于 2026-07-12（本 session 尾）。
> **运行环境：0234 机器，通过本地 tmux `pypto-ascend-0:0` 登陆**（8 卡 0-7 空闲；781GB RAM / 575GB free；driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1）。
> 编辑机 `b-csy-develop`（无 python，NFS 与 0234 共享，编辑即时可见）。分支 `pypto-lib feat/whole-net-n1-fusion`。

---

## ⛔ 用户硬约束（不可违背，勿走弯路）

- **必须用 IPC 共享显存机制**做端到端，**KV cache 和权重都走 IPC**。**不许 H2D 绕路**、不许换非-IPC 方案。
- **必须用真实权重加载**（真 W8A8 checkpoint，非 dummy）。
- 遇到问题只能**解决它**（IPC 机制内解），不能绕开。（Blocker B 已解，且根因**不是** IPC VA 冲突而是 gate_topk mrgsort，见下。）

## ⭐ 一句话状态（2026-07-12 更新）

**本 track = N=1 整网融合（offline）**：`whole_decode_faithful_real` —— **全部 42 层 MoE 内联进一个 `@pl.program`**，真 W8A8 权重经 **IPC** 加载，harness `tests/step3p5/_stage_whole_faithful_real_ipc.py`，分支 `feat/whole-net-n1-fusion`。
> ⚠ **别和另一个 track 搞混**：`NEXT-SESSION.md` 是 **G5b / per-layer 逐层 golden + live vLLM token-exact** track（harness `_stage_whole_decode_run.py`，在别的分支，本分支没有此文件）。两者都朝 INT8-native 收敛但**目标/harness/分支都不同**：本 track 是 offline IPC device 跑通，G5b track 是 live 数值对齐。

**✅ Blocker B 已解除（2026-07-12，commit `4bede85`）**：N=42 全 42 层真 W8A8 IPC，8 卡 → `REAL_WEIGHT_IPC_RUN_CLEAN` 3.48s，无 stall。
**根因不是 IPC VA 冲突（文档旧假设已被 device 证伪）**，而是 `gate_topk` 的 mrgsort 级联 bug（format2 二路归并被喂"各含 2 段"的半块 → SKILL "format2 半块未排序→状态机不终止→挂死" → AICore 挂死）。修复 = 对齐 DeepSeek v4 gate.py 渐进 format1 链，把 format2 二路改成 `mrgsort(block_len=256)`（4×256→1×1024 全排序）。
**剩余 = ① KV cache 也走 IPC（用户硬约束，当前 harness 喂 dummy KV）；② 整网精度 vs vLLM（gate 修复的数值正确性 + decode-step golden 或 live A/B）。**

## ✅ 已完成（device 验证，勿重做）

1. **W8A8 kernel（moe.py）DONE**：真 INT8×INT8→INT32 + dequant，照抄 DeepSeek v4，dispatch-side 量化（Option A，用户拍板）。
   - `_expert_routed` INT8 权重 + per-output-channel FP32 scale；gate/up `matmul(out_dtype=INT32)`+`matmul_acc` → `col_expand_mul(row_expand_mul(cast(acc,FP32), x_scale[T,1]), w_scale[1,N])`；中间 `h_i8` = DeepSeek cast 链（FP32→INT32 rint→FP16 round→INT8 trunc，`pl.at(CORE_GROUP,"routed_h_quant")`）；down INT8×INT8+dequant。
   - `_quant_moe_input` = **scheduled InCore + `pl.range`（非 spmd）+ 双输出 tuple return**（三 codegen 坑终解）。
   - dispatch_step：INT8 recv_x + 并行 `[.,SCALE_W_PAD=8]` scale 窗口融进同一 a2a barrier；repack scalar un-pad col-0 → `local_routed_x_scale [1,LOCAL_RECV_MAX]`。
2. **ccec TLOAD 修复（commit 1379ce2）**：expert 消费的 scale 改**非-padded contiguous** `[1,LOCAL_RECV_MAX]` + `[1,RECV_TILE]` ND2ND **row-slice+reshape**（DeepSeek recv_scale_dq 模式）。a2a 窗口仍 `[.,SCALE_W_PAD]`。
3. **✅✅ Stage C 精度 PASS（device）**：`_stage_moe_block_precision --layer 3 --dev-offset 0 --ckpt <W8A8> --bypass-gate --torch-golden` → **`'moe_out' PASS ratio_allclose(atol=0.04,rtol=0.04,max_error_ratio=0.1)` 27.19s**。INT8×INT8 W8A8 MoE-block 数值正确（vs torch W8A8-dequant = vLLM 同款数学），无 fractal-32 静默错。**精度验证达成（MoE kernel 级）。**（此路径用 H2D 权重只为隔离 kernel 精度，**不是** e2e 方案。）
4. **A5 整网 INT8 编译 DONE（TP=8）**：`whole_decode_faithful_real` inlined MoE 经 `tools/step3p5/_a5_int8_transform.py` 转 INT8（in-kernel per-token quant，DeepSeek cast 链）。`_probe_whole_faithful_canonical --layer-name whole_decode_faithful_real -d 0-7` → COMPILE OK。
5. **INT8 loader DONE**：`weight_loader.py int8_routed=True` → INT8 权重 + `KEY_MOE_W_{GATE,UP,DOWN}_R_SCALE`（`.squeeze(-1)` → `[N_MOE,EXPL,1280]`/`[.,4096]`）。host 验证 shape/dtype PASS。
6. **IPC exporter INT8 DONE**：`pypto_weight_ipc.py` `_dtype_for`/`_torch_dtype` 加 int8/float16；`export_from_checkpoint(int8_routed=True)`；池 47→**25.35 GiB/rank**。
7. **N=1 IPC 权重机制 device 证实（上上 session）**：`P_FAITHFUL_MOE_LAYERS=0`（跳 MoE）+ IPC 真 W8A8 + heap=2GB → `RESULT=REAL_WEIGHT_IPC_RUN_CLEAN`。即 attention+dense 全链（import_ipc + getitem + rt.run + tp_all_reduce）经 IPC 跑通。**IPC 机制本身可用**，只 MoE dispatch collective 卡。

## ✅ Blocker B — 已解除（2026-07-12，commit `4bede85`）

> **历史修正**：本节旧内容假设 Blocker B = "IPC 池 VA `0x12c1c0000000` 与 comm-window/arena VA 冲突"，并规划了一整套 VA-instrument / VA-placement 攻坚顺序。**该假设已被 device instrument 彻底证伪，勿再沿此方向。**

- **现象（旧）**：`_stage_whole_faithful_real_ipc -d0-7` → compile OK → import_ipc OK → rt.run → 前 4 chip 执行（`completed=4/32`）→ 507018 stall @ `stuck_task_id=0x100000003`。
- **VA 证伪（device 实测）**：在真正的 MoE 路径 `comm_hccl.cpp:703 domain_alloc_via_ipc` 加 VA 诊断（旧诊断加错在 `:431` 未走路径；`LOG_INFO_V0` 被默认 `info_v=5` 压掉，`logging.getLogger("simpler").setLevel(15)` 放开）→ 42 层仅 1 个 MoE comm domain window 在 `0x12c041600000`+396MB，8 卡一致，**整段在池 `0x12c1c0000000` 下方无重叠**。IPC 映射表（`pypto_weight_map.rank0.json`）48 key 全 512 对齐/无重叠/`max_end==pool 25.35GiB`/都在池内 → **VA 表完全正确，无冲突**。exporter 侧 `PYPTO_WEIGHT_IPC_VA_SHIFT_GB` 无效是因为它改的是 exporter 池，与真根因无关（commit `7765a0e` 是失败实验，可 revert）。
- **真根因 = `gate_topk` mrgsort 级联 bug**：device stall 快照（`ASCEND_GLOBAL_LOG_LEVEL=1`+`ASCEND_PROCESS_LOG_PATH` → `ascend_*/debug/device-*/`）显示 `task_id=3 state=RUNNING kernels=[aic:-1 aiv0:3] core=28(aiv0) fanin 3/3` 永不完成 = **`gate_topk` AIV kernel 挂死**（`orch_error=8` TENSOR_WAIT_TIMEOUT + S1 running-stalled；60s 超时排除"慢"）。`gate.py` SCORE_PAD=512 级联 `sort32→16×64 → mrgsort(block_len=64)→4×256 → mrgsort(srt[:,0:512],srt[:,512:1024])`——最后 format2 二路归并被喂两个"各含 2 段"的半块（非单段），违反 format2 前置 = SKILL "format2 半块未排序→状态机不终止→挂死"。**N≤2 clean / N=42 hang**（编译规模触发）。
- **修复**（对齐 DeepSeek v4 `gate.py` 渐进 format1 链，scale 到 512）：format2 二路 → `mrgsort(block_len=256)`（4×256→1×1024 全排序）。`gate.py` + `decode_layer.py` 10 处去重内联 MoE gate 全改。
- **验证**：全 42 MoE 层真 W8A8 INT8 IPC 池（25.35GiB/rank）8 卡 → `REAL_WEIGHT_IPC_RUN_CLEAN` 3.48s，无 stall。（输出为 0 因 harness 喂 dummy hidden/KV——device 执行路径已干净。）
- **工具**：harness 加 `--reuse-exporters`（8 exporter 常驻、survive force-reset、bisect 秒级 attach，免 15min 重载）。

### 剩余（下 session，本 N=1 track）

1. **KV cache 也走 IPC**（用户硬约束，当前 harness 喂 dummy KV）：KV bridge 见 memory `g5b_import_ipc_facade_missing`（dense L0 import vLLM KV + attention 读真 KV rc=0 已 device 证实）+ `g5b_kv_is_bf16_not_int8`（KV=bf16、1 head/rank、纯 layout reshape）+ `g5b_kv_bridge_not_pure_reshape`（per-layer feed，MAX_SEQ 不动 flash-attn scratch）。把权重 IPC 与 KV IPC 一起接进 `_stage_whole_faithful_real_ipc.py`。
2. **整网精度 vs vLLM**：gate 修复的数值正确性需 gate-exercising 验证（Stage C 之前 `--bypass-gate` 没验过 gate）；整网需 decode-step golden（重生成 vLLM W8A8 eager dump）或 live A/B。**注意**：G5b track（`NEXT-SESSION.md`）发现整网 token-garbage 的另一个 bug = BF16-dequant 在 L17 大残差幅值下精度退化——那是 **live/per-layer track 的问题，与本 N=1 track 的 gate_topk 是两回事**，别混。

## 🎯 e2e 跑通后：整网精度验证 vs vLLM

- ⚠ 现有 vLLM dump 是 **18-token PREFILL**，kernel 是 BATCH=16 **decode-step**——需 **decode-step golden**（重生成 vLLM W8A8 eager dump `--quantization ascend`）或 **live A/B**（8001 pypto 整网 vs 8000 vanilla；live 路径 co-tenancy 用 `SIMPLER_COMM_NO_HCCL=1`，见 memory `project_g4_cotenancy_hccl_conflict`）。
- MoE-block 精度已 PASS（Stage C）是整网精度的强证据；L1 per-layer hidden ratio_allclose(atol=0.04)。

---

## ⭐⭐ 关键 device 事实 / 铁律（勿再踩）

1. **单卡 ST/UT shape 铁律**：`apply_perrank_patch()`（保 TP=8 per-rank slice），**不用** `apply_tp1_patch()`。
2. **gap-5 坑**：in-kernel `pl.cast(bf16/fp32,INT8)` 喂 cube 可能静默 ~98% 错。**照抄 DeepSeek cast 链 + create_tensor 位置 + scope 就避坑**（本 session 已验证整网编译过 + MoE-block 精度 PASS）。
3. **三 codegen 坑**（`_quant_moe_input`）：终解 = InCore(pl.range) 双输出 tuple return。
4. **ccec ND2ND**：scale slice 必须 contiguous row-slice `[1,RECV_TILE]`+reshape。a2a3sim compile 过 ≠ device ccec-clean（必须真 device 跑）。
5. **每次 device run 慢**（8 rank sequential load 全 checkpoint 后 slice；~15min）。W8A8 ckpt = `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`。
6. **环境三件套**：`source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib`（WS=/data/chensiyu/hw_project/pypto/workspace）。
7. **launch 前**：`pkill -f '[_]stage_whole'` + `rm -f /tmp/n1_weight_ipc/STOP /tmp/n1_weight_ipc/*.rank*`；**禁 `-9` 强杀 / `npu-smi set -t reset`**（netboot 机锁死全卡）。stale pyc：monkey-patch 后 `find models/step3p5 -name '*.py' -exec touch {} +`。
8. **8 卡 env**：`PTO2_RING_HEAP=... PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
9. **push**：PAT `/data/chensiyu/secrets/github.env` + `git -c http.version=HTTP/1.1`，屏蔽 token。

## 📌 用户提示词要点（贯穿开发，务必遵守）

- **不走弯路，严格按要求**：e2e 必须 IPC（KV + 权重）+ 真实权重。Blocker B 只能解不能绕。
- **对齐 DeepSeek/Qwen**：遇问题先看 DeepSeek/Qwen 实现，尽量对齐；查历史开发文档避免重复造轮子。
- **架构优先**：coding 前先有整体架构脉络，思考后再落地。
- **step3p5-vs-DeepSeek 差异必须论证**：DeepSeek 为什么没遇到？能否搞成一样？只在"我们性能更好"时保留差异。
- **地址对齐 / padding / shape / dtype / layout** 都检查（Blocker B 正是 VA 地址问题）。
- **遇决策找另一个 agent 讨论**；重要的是**完成目标**。**严格遵守 SKILL.md**（`pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`），不满足约束可能是设计不合理需重新设计。

## 本 session commits（feat/whole-net-n1-fusion）

pypto-lib：`cd3ef0d`(A kernel) `a293fe7`(A5 transform+gen) `132fedc`(StageB wire) `32b59d3`(scale squeeze+IPC int8) `6404385/6fe58ee/517dd6e`(Stage C harness) `1379ce2`(ccec scale fix + Stage C PASS) `7765a0e`(VA-shift 实验，失败可 revert) **`4bede85`(gate_topk mrgsort 修复 → Blocker B 解除，N=42 IPC device-clean)**。

---

```
继续 step3p5 **N=1 整网 W8A8 offline 端到端集成收尾**（本 track = `whole_decode_faithful_real`，42 层 MoE 内联进一个
@pl.program，真 W8A8 权重经 IPC，harness `tests/step3p5/_stage_whole_faithful_real_ipc.py`，分支 feat/whole-net-n1-fusion）。
⚠ 别和 `NEXT-SESSION.md` 的 G5b/per-layer 逐层-golden+live-vLLM track 搞混（那是别的分支、别的 harness）。
用户硬约束：必须 IPC 共享显存（KV + 权重都走 IPC）+ 真实权重，不走弯路、不许 H2D 绕路。

✅ Blocker B 已解除（commit 4bede85）：全 42 层真 W8A8 IPC 8 卡 REAL_WEIGHT_IPC_RUN_CLEAN 3.48s。根因不是 IPC VA 冲突
（device instrument 证伪：comm window 0x12c041... 在池 0x12c1c0... 下方无重叠），而是 gate_topk mrgsort 级联 bug
（format2 二路被喂"各含 2 段"半块→挂死），已按 DeepSeek format1 链修（format2→mrgsort(block_len=256)）。

第一步：KV cache 也走 IPC（用户硬约束，当前 harness 喂 dummy KV）—— 借 memory g5b_import_ipc_facade_missing /
g5b_kv_is_bf16_not_int8 / g5b_kv_bridge_not_pure_reshape，把权重 IPC + KV IPC 一起接进 _stage_whole_faithful_real_ipc.py。
第二步：整网精度 vs vLLM —— gate 修复的数值正确性需 gate-exercising 验证（Stage C 之前 --bypass-gate 没验 gate）+
decode-step golden 或 live A/B。（注意 G5b track 的 L17 BF16-dequant 精度退化是那个 track 的 bug，与本 track gate_topk 无关。）

机器 0234 经 tmux pypto-ascend-0:0（8 卡；exporter reset 后仍可 --reuse-exporters attach）；编辑机 b-csy-develop 共享 NFS；
分支 pypto-lib feat/whole-net-n1-fusion。全部读上面 ⛔/⭐/✅/铁律/提示词要点。
环境三件套：source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh &&
export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib（WS=/data/chensiyu/hw_project/pypto/workspace）。
每完成关键节点更新 pypto-project STATUS/phases + push（同 session）。
```
