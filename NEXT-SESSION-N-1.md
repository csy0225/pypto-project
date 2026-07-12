# NEXT SESSION — N=1 整网 decode 集成 + 端到端精度对齐 vs vLLM

> 直接把最底部 code block 当第一条消息粘贴。自包含。更新于 2026-07-12（本 session 尾）。
> **运行环境：0234 机器，通过本地 tmux `pypto-ascend-0:0` 登陆**（8 卡 0-7；781GB RAM；driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1）。
> 编辑机 `b-csy-develop`（**无 python，有 npu-smi，能直连 github**；NFS 与 0234 共享，编辑即时可见）。分支 `pypto-lib feat/whole-net-n1-fusion`。

---

## 🎯 总目标（north star，勿只盯单步）

**完成 step3p5 整网 decode 阶段的集成 + 端到端精度对齐 vs vLLM**（真 W8A8 + IPC，token-exact）。
本 track = N=1 整网融合（offline）：`whole_decode_faithful_real` —— **全 45 层（42 MoE + 3 dense/swa）内联进一个 `@pl.program`**，真 W8A8 权重+KV 经 **IPC** 加载，harness `tests/step3p5/_stage_whole_faithful_real_ipc.py`，分支 `feat/whole-net-n1-fusion`。
> ⚠ **别和另一个 track 搞混**：`NEXT-SESSION.md` 是 **G5b / per-layer 逐层 golden + live vLLM** track（harness `_stage_whole_decode_run.py`，本分支无此文件）。目标/harness/分支都不同，但 L2/live 阶段可复用其 KV-bridge / co-tenancy 机器。

### 里程碑路线图（当前进度）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 单算子 probe | matmul_acc N=16 / head-gate / gate_topk vs torch | ✅ PASS |
| M1 功能 bring-up | 42 MoE 真 W8A8 + 权重+KV 双 IPC 8 卡 dispatch-clean | ✅（Blocker B 解） |
| M2 per-layer gate_r | monolithic 整网自算逐层 head-gate（on-device，token-exact-capable） | ✅（路径 a，本 session） |
| **M3 单层 MoE 数值正确** | **单层 INT8 routed-MoE 有效行 NaN 修掉 → finite** | **⛔ 当前卡这里** |
| M4 L1 ctx=1 token-exact | 全 42 层放开，`--hidden-token 6127` → **argmax=303** vs vLLM | ⏸ gated on M3 |
| M5 L2 多 token / decode-step | vLLM→whole-net KV bridge 或 live A/B（8001 vs 8000），多 token token-exact | ⏸（需 port G5b 机器） |
| M6 整网 decode 集成落地 | 接入 serving 路径（live single-handoff），端到端精度双过准出 | ⏸ |

**判据**：L1 per-layer hidden `ratio_allclose(atol=0.04)` / L2 logits cos≥0.999+topK overlap≥4/5 / L3 greedy top-1≥95%。**oracle = vLLM eager dump，synthetic golden 会 stale。**

## ⛔ 用户硬约束（不可违背，勿走弯路）

- **必须用 IPC 共享显存机制**做端到端，**KV cache 和权重都走 IPC**。**不许 H2D 绕路**、不许换非-IPC 方案。（权重+KV IPC 已 device 跑通。）
- **必须用真实权重加载**（真 W8A8 checkpoint，非 dummy）。**真权重调试，不走其他弯路。**
- 遇到问题只能**解决它**，不能绕开（work-around）。诊断脚手架只能定位、不能进产品路径。
- **correctness 和 speed 都要**：既要跑出正确结果、也要推进到底完成目标；别用"correctness"当借口停在半路，也别为"快"造出错误的精度数字。
- **不能只盯一个子目标**——总目标是整网 decode 集成 + 端到端精度对齐；单步（如 MoE NaN）修完要立刻推进下一里程碑。
- **对齐 DeepSeek/Qwen**：遇问题先看 DeepSeek v4/Qwen 实现 + 历史开发文档，尽量对齐；step3p5-vs-DeepSeek 差异必须论证（只在"性能更好"时保留）。
- **架构优先**：coding 前先系统分析 + 整体设计。**严格遵守 SKILL.md**（`pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`）；不满足约束可能是设计不合理需重设计。
- **⚠ 历史文档可能 stale，先核对当前代码再下结论**。

## ✅ 已解决并推送（pypto-lib `f07da3b`）

- **M1 Blocker B（gate_topk mrgsort）+ 权重+KV 双 IPC 8 卡 device-clean**（`4bede85`/`c61046b`/`b92031f`）。gate_topk 修复对齐 DeepSeek format1 链，device 数值验证 PASS（vs torch.topk）。
- **M2 整网 per-layer gate_r 结构性阻塞已解 —— on-device head-gate（路径 a）**：
  - `matmul_acc N=16` 丢 K 累加的 codegen bug（当年把 head-gate 移 worker 的原因）**现栈已修**（`_probe_matmul_acc_n16` + full-chain `_probe_head_gate_full` device PASS）。
  - `attention_full.py`/`attention_swa.py` **Scope 1.f 恢复 on-device gate**：`gate_logits = normed_all @ w_g`（K-chunk matmul_acc N=16）→ `sigmoid` → `gate_exp = gate_score @ R`（N-chunk）；Scope 3.a o_proj 乘 `gate_exp`（2 scope 分离控 UB）。
  - `gate_r` 槽改承载 **block-diag R 常量**（`R[h,h*HEAD_DIM+d]=1` 实头；**layer-independent → 喂一次全 45 层通用**），harness 填 R（实头=HQ//HEAD_DIM：full=8/swa=12）。→ monolithic 整网每层从自己 `normed_all` 自算 gate → **token-exact-capable**（不再需 per-layer dispatch / resident-DeviceTensor）。
  - 对齐 vLLM `modeling_step3p5` L489（g_proj on **post-input-layernorm** hidden，非 raw）+ L527-531。`whole_decode_faithful_real` **TP=8 COMPILE OK**（attention inline 从 `._func` 重导，无需 regen）。
  - memory `n1_head_gate_ondevice_restored_l1_nan` / `step3p5_head_gate_uses_normed_hidden`。

## ⛔ M3 当前阻塞：单层 INT8 W8A8 routed-MoE 的**有效行**计算产生 NaN

**L1 ctx=1 A/B 现状**：pypto worker `--hidden-token 6127 --kv-ipc` **RUN_CLEAN ~3.6s 但 `next_hidden=nan / logits=nan / argmax=0`**（vLLM golden：tid=6127「北京」→ next=**303**「，」）。

**已完成的 bisect（旋钮 `P_FAITHFUL_MOE_LAYERS`，`decode_layer.py:19182`，默认 42，emit N 层 MoE）**：
- `P_FAITHFUL_MOE_LAYERS=0`（仅 3 dense/swa attention 层）→ **FINITE**（`next_hidden=502.0 / logits=9.03 / argmax=27527`）→ **attention 路径 + on-device head-gate 确认干净**。
- `P_FAITHFUL_MOE_LAYERS=1`（3 attention + **1 MoE**）→ **NaN** → **单个 INT8 MoE 层就复现**，非跨层累积。MoE 输入 post_attention_layernorm 归一化（O(1)，非 502），不是输入幅度问题。

**已排除**：
- **不是 gate/head-gate**（bisect MoE=0 干净）。
- **不是 A-operand fractal-32 padding**（gap-5 经典坑）：试过把 `routed_x_quant`/`routed_h_quant` 的 `x_i8`/`h_i8` padding 行 [tile_valid:RECV_TILE] 用 `fillpad(set_validshape(cast_tile,tile_valid,K), PadValue.zero)` 置零（对 cast **TILE**，非 tensor —— set_validshape 只吃 TileType）→ 编译+跑通**仍 NaN** → 已 revert（tree clean f07da3b）。amax 是 per-row（`row_max` over K → `[1,RECV_TILE]`），有效行 scale 干净，padding 置零证实无效。
- **不是 stale 权重格式**：exporter `int8_routed=True` → INT8 pool 25.35GiB；`w_g` padding zero-pad（`weight_loader._slice_g_proj` L594）；KV pool `torch.zeros`（`pypto_weight_ipc.export_from_checkpoint` L394）。

**关键结构事实**：whole-net 内联 MoE 是 **INT8-native in-expert quant**（`decode_layer.py` `routed_x_quant`/`routed_h_quant` scope：`xe_amax→scale→cast INT8 trunc→matmul out_dtype=INT32→dequant col/row_expand_mul`），`_quant_moe_input` 次数 **decode_layer.py=0 vs moe.py=2** —— 即 whole-net 内联 MoE 与 **standalone-validated `moe.py` INT8 kernel（dispatch-side quant, Option A）DECOUPLED**（旧 in-expert 路子）。只有 **2** 处 `routed_x_quant`（base + real builder），routed expert body 是共享 inline（**非 42 份**）→ 定位/修改是**单点**。

**剩余嫌疑（有效行计算，按可疑度）**：
1. **routed INT8 gate/up/down dequant**：`gate_2d = col_expand_mul(row_expand_mul(cast(gate_acc,FP32), x_scale_dq), wg_scale_row)`（`decode_layer.py:~20113`）—— 查 IPC 里 `moe_w_gate_r_scale`/`moe_w_up_r_scale`/`moe_w_down_r_scale` 是否含 0/inf/NaN（weight_loader dequant / exporter INT8 scale 逻辑）。
2. **combine** routing weight × routed_y + shared。
3. **shared expert**（BF16 swiglu；layer-0=full_moe_silu_silu 用 silu）。
4. **dispatch a2a**（comm windows / CSR）。

**M3 攻克手法（下 session，用户已批准集中攻）**：
- **手法 A（推荐，最快隔离 权重 vs 代码）**：把真 IPC INT8 routed 权重+scale 喂进 **standalone `moe.py` MoE-block harness**（已 device-validated 的 kernel）跑单 MoE block。standalone **finite** → whole-net 内联副本 bug（→ 手法 C）；standalone **也 NaN** → 权重/scale bug（→ 查 weight_loader/exporter INT8 scale dequant）。
- **手法 B（per-op dump 仪表化）**：real builder routed expert 加中间 Out（gate_2d / h_bf16 / routed_y / shared_out）经 host_orch + harness 拉回，逐 stage 看谁先 NaN。monolithic 无中间输出，必须显式加 Out。
- **手法 C（根治，A5 大改）**：把 `moe.py` 的 `_quant_moe_input` + dispatch-side INT8 recv（Option A，已 validated）经 `tools/step3p5/_gen_faithful_real.py` regen 应用到 whole-net 内联 MoE（替换 decoupled 旧副本）。工作量大但根治。
- **收尾即推进 M4**：修完 `P_FAITHFUL_MOE_LAYERS=1` finite → 逐步放开 2/4/…/42 → 全量 L1 A/B（tid 6127 期望 **argmax=303**）→ 进 M5（多 token / KV bridge / live A/B）。

## 🎯 精度对齐三档（对应 M1/M4/M5）

- **L0 单算子 probe（M0 ✅）**：`_probe_matmul_acc_n16`、`_probe_head_gate_full`、`_probe_gate_sort` 全 PASS。范式见 🐞。
- **L1 ctx=1 单 token A/B（M4，脚手架已建可跑）**：`_stage_whole_faithful_real_ipc.py --hidden-token <id>` 把 `embed(token)` 灌进 `current_hidden` row0 + pos-0 identity rope（cos=1/sin=0），seq_lens=ones（16 行全 ctx=1，避开 seq_len=0 NaN）。vLLM 1-token prompt 首 token = argmax(logits(pos0))，等价 ctx=1 self-attn，**不需 KV bridge**。**当前卡 M3 MoE NaN**，修完即通。
- **L2 整网多 token / decode-step golden（M5，终极）**：多 token 需 vLLM→whole-net KV bridge（分页池→整网 flat KV，memory `g5b_kv_bridge_not_pure_reshape` / `g5b_kv_is_bf16_not_int8` / `g5b_import_ipc_facade_missing`）或 live A/B（8001 pypto vs 8000 vanilla，co-tenancy `SIMPLER_COMM_NO_HCCL=1`，memory `project_g4_cotenancy_hccl_conflict`）。**这套是 G5b track 机器（0162 working tree），本分支需 port。**

## 🖥 环境 / vLLM oracle 启动（本 session 验证可用）

- **三件套激活**（每 fresh shell，`activate.sh` 不带 CANN env）：
  `source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib`（`WS=/data/chensiyu/hw_project/pypto/workspace`）。
- **vLLM W8A8 oracle（0234 可跑）**：**⚠ 先 `unset PYTHONPATH` 再 `source cann/set_env.sh`**（顺序关键——先 source 再 unset 会把 acl 抹掉 → `ModuleNotFoundError: acl`），**不 export pypto PYTHONPATH**，再：
  `vllm serve <W8A8ckpt> --served-model-name step3p5 --trust-remote-code --quantization ascend --tensor-parallel-size 8 --enable-expert-parallel --enforce-eager --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85`
  → 占 8 卡 0-7，~5-6min load。
  - **⚠⚠ vLLM 跑在 0234，查询必须在 tmux 0234 里 `curl localhost:8000`**。**从 b-csy-develop 的 Bash `curl localhost:8000` 命中的是它自己的 nginx → 404**（本 session 踩过，别误判 vLLM 挂了）。
  - 1-token golden：`_l1_ab_vllm.py --word 北京`（取首 token 作 tid，查 vLLM greedy next-token id+text）。本 session：tid=6127→303。
  - vLLM 与 pypto **同卡** → 先 vLLM 出 golden 再 kill 腾卡跑 pypto（offline A/B）。kill：`pkill -f "[v]llm serve"; pkill -9 -f "[E]ngineCore"`。
- **W8A8 ckpt** = `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`（arch `Step3p5ForCausalLM`，45 层，embed 在 shard 00048 非量化）。
- **8 卡 pypto env**：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
- **exporters（IPC 权重/KV）—— bisect 高效设置**：默认 worker mode 自己 launch 8 exporter + 跑完写 STOP（每次 15min 冷载）。**多次 bisect 用 hold-mode**：先手动起 8 个常驻 `--export-rank r --dev r --kv-ipc`（一次 15min 冷载，写 `ready.rank{r}`+`pypto_weight.key.rank{r}` 后 hold 等 STOP），再多次 `--reuse-exporters` worker 秒级 attach（每次只 compile+run ~5min）。收尾 `touch /tmp/n1_weight_ipc/STOP` 或 `pkill -f export-rank` 释放卡。
  - **⚠ `/tmp` 每机独立**：exporter 写 0234 的 `/tmp/n1_weight_ipc/`，**从 b-csy-develop `ls /tmp/...` 看不到**（本 session 误以为 0 ready）。查 ready keys 必须在 **tmux 0234**：`ls /tmp/n1_weight_ipc/ready.rank* | wc -l`。

## 🐞 Debug 方式（累积经验）

- **数字 device error 先查 [wiki Device-Error-Codes_zh](https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh)**。`507018` 是泛化 host 码，看 `orch_error_code`/`sched_error_code`/`sub_class` 定真因。
- **device stall 快照**（定位卡住 kernel/核，本 session 定位 gate_topk 用过）：harness `logging.getLogger("simpler").setLevel(15)` + `export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_PROCESS_LOG_PATH=<预建目录>` → 读 `<dir>/debug/device-*/device-*.log` 的 `log_stall_diagnostics`（`TASK state=RUNNING kernels=[...] running_on=[core=X]`）。`orch_error=8`=TENSOR_WAIT_TIMEOUT，`sched=100`=running-stalled。root-owned，tmux(root) grep。
- **NaN/精度 bisect（本 session 定位 MoE 用的）**：`P_FAITHFUL_MOE_LAYERS=N` 层数二分（0=纯 attention / 1=单 MoE / …/42）隔离 attention-vs-MoE、单层-vs-累积。monolithic 无中间输出 → 要 op 级须显式加 Out dump（手法 B）。
- **单算子精度 probe 范式**（隔离验证）：module-level `@pl.jit` + `from golden import TensorSpec, run_jit, ratio_allclose` + torch golden_fn（见 `_probe_head_gate_full.py`/`_probe_matmul_acc_n16.py`）。⚠ `@pl.jit` 的 shape/const 必须 module-level；per-K-chunk cast 避免 `[T,HIDDEN]` FP32 UB overflow（256KB>184KB）；2-scope 分离让 K-loop buffer 先释放再做 N-chunk expand；小 N=16 输出 matmul_acc 现栈已正确累加 K。
- **launch 前**：`pkill -f '[_]stage_whole'` + `rm -f /tmp/n1_weight_ipc/{STOP,ready.rank*,pypto_weight.*}`（在 0234）；**禁 `-9` 强杀 device 进程 / `npu-smi set -t reset`**（netboot 机锁死全卡）；`npu-smi info -t usages -i <c>` 确认 HBM<10%。stale pyc：改 models 后 `find models/step3p5 -name '*.py' -exec touch {} +`。
- **每次 device run 慢**：exporter 全 ckpt load ~15min（用 hold+reuse 省重复冷载）；compile 42 层 ~4min，少层更快。

## ⭐⭐ 铁律（勿再踩）

1. **单卡 ST/UT shape**：`apply_perrank_patch()`（保 TP=8 per-rank slice），不用 `apply_tp1_patch()`。gate_matmul 单卡 unsliced 会 Mat/Vec 溢出——验 gate 用隔离 probe。
2. **gap-5 坑**：in-kernel `pl.cast(bf16/fp32,INT8)` 喂 cube 可能静默错；照抄 DeepSeek cast 链 + create_tensor 位置 + scope。**但本 track 的 MoE NaN 已证不是 A-operand padding**（经典 gap-5 fix 试过无效），是有效行计算。
3. **ccec ND2ND**：scale slice 必须 contiguous row-slice + reshape；a2a3sim compile 过 ≠ device ccec-clean（必须真 device 跑）。
4. **push**：PAT `/data/chensiyu/secrets/github.env` + `git -c http.version=HTTP/1.1`，输出屏蔽 token。**github 从 0234 连不通**（直连 130s 超时；proxy `deploy.i.shaipower.com/httpproxy` 返回的 proxy 需 auth → 407）；**从 b-csy-develop 的 Bash 能直连 github**（NFS 共享同一 repo；commit 在 tmux(root) 做完后 `git push` 从 b-csy-develop 跑）。pypto-lib `.git/objects` root-owned → commit 走 tmux(root)，push 可从 b-csy-develop（读 objects OK）。pypto-project chensiyu-owned 可直接。跨仓 push 同步 STATUS pin。
5. **文档 stale 风险**：核对当前代码再下结论（head-gate 就栽在 stale 注释）。
6. **别只盯单步**：M3 修完立刻推进 M4→M5→M6，总目标是整网 decode 集成 + 端到端精度对齐。

## 本 session commits

pypto-lib `feat/whole-net-n1-fusion`：`f07da3b`（on-device head-gate 恢复 + `_probe_matmul_acc_n16`/`_probe_head_gate_full`/`_l1_ab_vllm` + harness 填 block-diag R + 清 stale 注释）。
pypto-project `main`：`4b045b8`/`b7e8986`/`e05b59d`/`612a830`/`fe8750c`（STATUS pin + blockers §1 + NEXT-SESSION：gate 解除 + L1 NaN bisect + A-operand disproven）。

---

```
继续 step3p5 **N=1 整网 decode 集成 + 端到端精度对齐 vs vLLM**（总目标；不是只修一个 bug）。当前挡路的一步 =
单层 INT8 routed-MoE 有效行 NaN，修完立刻推进：全 42 层 L1 token-exact → L2 多 token（KV bridge / live A/B）→ 整网 decode 集成落地。
本 track = whole_decode_faithful_real（45 层内联一个 @pl.program，真 W8A8 权重+KV 经 IPC，harness
tests/step3p5/_stage_whole_faithful_real_ipc.py，分支 feat/whole-net-n1-fusion，pypto-lib HEAD f07da3b）。
⚠ 别和 NEXT-SESSION.md 的 G5b/per-layer track 搞混（但 L2/live 可复用其 KV-bridge/co-tenancy 机器）。
用户硬约束：IPC(KV+权重)+真实权重、遇问题只解不绕、correctness 和 speed 都要、不能只盯单步（总目标是整网 decode 集成+端到端精度）、
对齐 DeepSeek、架构优先、严格遵守 pypto-dev-constraints SKILL、历史文档可能 stale 先核对代码。

里程碑：M0 单算子 probe ✅ / M1 功能 bring-up（42 MoE+双 IPC 8 卡 dispatch-clean，Blocker B 解）✅ /
M2 per-layer gate_r（on-device head-gate 路径 a：matmul_acc N=16 bug 现栈已修 probe PASS，attention_full/swa Scope 1.f
gate_logits=normed_all@w_g(K-chunk)→sigmoid→gate_exp=gate_score@R(N-chunk)，gate_r 槽承载 layer-independent block-diag R
→ 整网自算逐层 gate → token-exact-capable，TP=8 COMPILE OK）✅ / M3 单层 MoE 数值正确 ⛔当前 / M4 L1 token-exact ⏸ /
M5 L2 多 token（KV bridge/live A/B）⏸ / M6 整网 decode 集成落地 ⏸。

⭐ M3 = 单层 INT8 MoE 有效行 NaN：L1 ctx=1（--hidden-token 6127，vLLM golden next=303）RUN_CLEAN 但 logits=nan。
bisect P_FAITHFUL_MOE_LAYERS=0（仅 attention）→FINITE(argmax=27527)；=1（+1 MoE 层）→NaN → 单个 INT8 MoE 层就复现。
已排除：gate/head-gate（MoE=0 干净）、A-operand fractal-32 padding（gap-5 经典 fix 试过无效已 revert）、
stale 权重格式（INT8 pool、w_g zero-pad、KV zeroed）。关键：whole-net 内联 MoE 是旧 in-expert INT8 quant，
与 standalone-validated moe.py（_quant_moe_input dispatch-side，count decode_layer.py=0 vs moe.py=2）DECOUPLED；
只 2 处 routed_x_quant（base+real builder），routed body 共享 inline（非 42 份）→ 单点定位。剩余嫌疑（有效行）：
routed INT8 gate/up/down dequant（查 moe_w_*_r_scale IPC 值 finite）/ combine routing weight / shared expert / dispatch。
攻克：(A 推荐) 真 IPC INT8 权重喂 standalone moe.py MoE-block harness 隔离 权重-vs-代码；(B) real builder routed expert 加
per-stage Out dump 逐段看谁先 NaN；(C 根治) 用 moe.py dispatch-side quant 经 _gen_faithful_real.py regen 替换内联旧 MoE。
修完 P_FAITHFUL_MOE_LAYERS=1 finite → 放开 42 → 全量 L1（argmax=303）→ 进 M5。

环境：0234 tmux pypto-ascend-0:0（8 卡）；三件套 source cann/set_env.sh+activate.sh+PTO_ISA_ROOT+PYTHONPATH。
⚠ /tmp 每机独立——查 ready keys / curl vLLM 必须在 tmux 0234（b-csy-develop 的 localhost:8000 命中自己 nginx→404）。
vLLM oracle：unset PYTHONPATH 再 source cann/set_env.sh（否则 import acl 失败），vllm serve <W8A8ckpt> --quantization ascend
--tensor-parallel-size 8 --enable-expert-parallel --enforce-eager --trust-remote-code --port 8000（占 8 卡，先出 golden 再 kill）。
bisect 高效：先起 8 hold-mode exporter（--export-rank r --dev r --kv-ipc，一次 15min 冷载），再多次 --reuse-exporters worker
（P_FAITHFUL_MOE_LAYERS=N 变体，每次 ~5min）；收尾 touch /tmp/n1_weight_ipc/STOP 释放卡。
debug：数字 error 先查 wiki Device-Error-Codes_zh；stall 快照 setLevel(15)+ASCEND_GLOBAL_LOG_LEVEL=1+ASCEND_PROCESS_LOG_PATH；
NaN bisect 用 P_FAITHFUL_MOE_LAYERS 层数二分。禁 -9/npu-smi reset。push 走 HTTP/1.1+PAT，从 b-csy-develop 直连 github
（0234 连不通），pypto-lib commit 走 tmux root。全部读上面 🎯/⛔/✅/🖥/🐞/铁律。每完成关键节点更新 pypto-project STATUS/blockers/NEXT-SESSION + push（同 session）。
```
