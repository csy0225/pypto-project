# NEXT SESSION — N=1 整网融合 kickoff prompt

## ⏱ 2026-07-11 进展更新（Task#2 增量② 核心达成 + 剩余硬 gate）

**✅ 已完成并推送（pypto-lib `be10ba0`，分支 `feat/whole-net-n1-fusion`）**：新 program
`whole_decode_faithful_real`（`WholeDecodeFaithfulReal`）—— **真实 per-layer 权重 + full+swa 完整路由**
（11 full-MoE 层 L4,8,…,44 走 full attn；31 swa）。norm[45] kernel-index 绝对 L；attn/dense/MoE
全部 host-slice 单层（探针 `_probe_single_layer_inline` 证明 `pl.inline` 接受实参 leading dim <
标注）；MoE experts[42] host-slice by pos；`chip_orch` 仅 `layer_idx→norm_layer_idx`。生成器
`tools/step3p5/_gen_faithful_real.py`（保留 reuse-one-slab 干净基线）。**smoke rc=0 + compile rc=0 +
8 卡 device DISPATCH_CLEAN 285s（dummy，88 pass-block，无 507018/stall）**。旧基线不回归。
phase27 已记录、memory `n1_real_per_layer_builder_device_clean` 已存。

**⛔ 剩余两个硬 gate（本 session 无法关闭，不是本程序的问题）**：
1. **Task#4 逐层对齐 vLLM**：vLLM eager dump 是 **18-token PREFILL**，decode kernel 是 **BATCH=16 单
   token** → shape 不兼容，kernel-vs-dump 无法直接逐层比。**需生成 decode-step golden**（1 tok/step，
   batch≤16）或走 live 路径对齐。（真实权重 harness `_stage_whole_faithful_real_weights.py` 已写好，
   但它在驱动进程 8× 全量 jfs 重读 checkpoint 太慢——要改 per-rank forked-load 对齐生产；且它也不解决
   数值对齐。ckpt 路径 `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp` 正确。）
2. **Task#5 live A/B（IPC + single-handoff）**：memory `hccl_single_world_cotency_blocks_forked_worker`
   —— whole-decode worker 的 HCCL world 与同卡 vLLM 冲突（`HcclCommInitRootInfo failed:7`，一卡一个
   HCCL world）。需 (a) 上游 simpler IPC-only control comm，或 (b) in-process 复用 vLLM comm（Phase 22）。

**下个 session 起点建议**：(1) decode-step golden 生成 → 用 `whole_decode_faithful_real` 逐层 device 对齐；
或 (2) 推进 Task#5 的 HCCL 共存上游方案。分支仍用 `feat/whole-net-n1-fusion`。

---

> 直接把下面代码块粘贴为 `/goal` 执行。自包含。

```
继续 step3p5 N=1 整网融合的 runtime/集成阶段。Task#1(scheduler-timeout)已解决,本 session 攻 Task#2 增量②(MoE 层真实权重 + 逐层对齐 vLLM) + Task#3(IPC + live A/B)。

## 起点(已完成,勿重做)
分支: pypto-lib `feat/whole-net-n1-fusion`(csy0225 fork)。**用这个分支,不是 stepfun/develop。**
- HEAD = `7b9693b`。关键 commit: `1a68c47`(Task#1 scheduler-timeout 修复) + `7b9693b`(Task#2 增量① dense-prefix 真实索引)。
- pypto-project main HEAD = `24fd754`(phase27 已含完整 Step-2 蓝图 + 策略决定)。
- ✅ Task#1 已解决: N=1 整网 scheduler-timeout 根因 = 42 个 MoE 层复用一套 comm window → 同一 SSA 跨 chip_orch submission 别名 → 违反 RAW-only-v1 non-aliasing(P3/ADR-013)。修复 = 每层 distinct 窗口(`_L{pos}`,严格 N=1 不用 multi-program)+ 删 dead 共享 alloc(MaterializeCommDomainScopes 会报错)。全 45 层 8 卡 device RUN_CLEAN 45.89s。诊断 env 门控 `P_FAITHFUL_MOE_LAYERS`(默认 42)。
- ✅ Task#2 增量①: dense-prefix(L0/L1/L2)`full_chip_orch`/`swa_chip_orch` 单 layer_idx→3 scalar,host_orch 传真实三元组 L0=(0,0,0)/L1=(1,0,1)/L2=(2,1,2)。8 卡 DISPATCH_CLEAN 无回归。

## 环境铁律(0234 已修复态,保持不变)
- CANN 保持 beta.1: `/usr/local/Ascend/cann-9.0.0-beta.1`(新装的 cann-9.0.0 是坏的 HCCL ABI,勿用)。
- simpler 必须 SDMA-OFF: `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=OFF`。任何重编启跑 device 前先 `allreduce_distributed -p a2a3 -d 0-7 --mode twophase` 复验 = `max|out-expected|=0`。
- 换 pod/换 CANN 后必须 clean 重编: `build_runtimes --platforms a2a3`(`--clone-protocol` 已废弃),否则单卡 AICPU 507018。
- 三件套(每 fresh shell): `source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa`($WS=/data/chensiyu/hw_project/pypto/workspace)。
- 开发/编译只在 0234(tmux `pypto-ascend-0`),绝对不碰 0162。8 卡跑设 `PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
- git push 用 PAT `/data/chensiyu/secrets/github.env` + `-c http.version=HTTP/1.1`,输出屏蔽 token。
- launch device 前 `pgrep -af '[p]ython -m tests'` 确认无残留 + `npu-smi info` HBM=0(AICore 85-93% 是 idle 噪声,非卡死)。

## 本 session 目标(按顺序)

### Task#2 增量②: MoE 42 层真实 per-layer 权重 + 逐层对齐 vLLM
锁定策略(phase27 已定,勿再权衡): **norm 走 full-stack + 真实绝对 `norm_layer_idx=L`**(KV-cache base = norm*cache_rows 必须绝对 L); **attn + MoE 权重走 host-side slicing**(host_orch 侧 `stack[pos]` 切单层 slab)。

已确认事实(勿重复排查):
- faithful `chip_orch`(decode_layer.py L20546) 的 attention 参数(m_wq/wk/wv/wo/k_cache 等)**是 vestigial 未用**;body(L20604-20748)只用 `current_hidden`(=swa_attn_only 输出)做 post-RMSNorm(`post_rms_weight[layer_idx]` @L20650)+ MoE(gate/dispatch/expert/combine,MoE 权重无 kernel 索引)+ residual。→ chip_orch 的 `layer_idx` 纯 = norm 索引。
- 逐层索引映射(绝对层 L): norm_idx=L; attn type-local = full `L//4`(0..11)/ swa `L-(L//4+1)`(0..32); mlp=L(dense 0/1/2); moe pos=L-3。
- pre-existing 坑: `LAYER_HIDDEN_ROWS_DYN=49152`(=12×HIDDEN)被 attention_swa 复用于 swa wq/wk/wv,但 swa 有 33 层,真实需 `33×HIDDEN=135168`。→ 喂真实 swa attn 权重前必须核对 loader `KEY_WQ_SWA[33,HIDDEN,...]` 并调 config const(或 host-slice attn 单层规避)。

改动清单:
1. `swa_attn_only_orch`(L20890): `layer_idx`→`(norm_layer_idx, attn_layer_idx)`,inline call `(layer_idx,layer_idx)`→`(norm_layer_idx, attn_layer_idx)`。
2. `chip_orch`(L20546): `layer_idx`→`norm_layer_idx`(rename,post_rms 用);MoE 权重参数保持单层,host 侧预切。
3. host_orch(L20940): MoE 段 42 层每层 call 传真实 `norm=L`(chip_orch)+`(norm=L, attn=swa-local)`(swa_attn_only);MoE 权重 args 改 `[tp,42,...]` 栈 + 逐层 `[r,pos]` slice;swa attn 栈 33 层。
4. harness `tests/step3p5/_stage_whole_faithful_device.py`(L102-192): 全栈 shape([tp,45/42/33,...]),用 `weight_loader` 真实权重填充(而非 dummy)。
5. weight_loader(L747-864)已就绪(3-class 栈 + W8A8 dequant,无需改)。真实权重: `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp/`。
6. 验证: smoke rc=0 → 8 卡 device 逐层 hidden vs vLLM eager dump(L1 ratio_allclose atol=0.04)+ final logits argmax 一致。**oracle 用 vLLM detail dump,synthetic golden 会 stale**。注意 vLLM dump 是 prefill(18 tok),需 decode-step golden(见 memory `vllm_golden_dumps_are_prefill_not_decode`)。

建议先加 flag/新 builder,保当前 per-layer-window reuse-one-slab 干净基线不回归。

### Task#3: 47GiB 单 key 权重 IPC + wire `_pypto_full_forward` + live A/B
gate 在 Task#2。`vllm_monkey_patch.py:233` 的 `_pypto_full_forward` 仍 D3 fail-closed stub;整模型 monkey-patch(非 per-layer)+ enforce_eager;live A/B 8001(pypto 整网)vs 8000(vanilla)token-exact。

## 先读(对齐状态再开工)
memory: `n1_whole_net_scheduler_timeout_fixed_perlayer_windows`、`n1_multicard_ipc_fix_sdma_workspace_off`、`step3p5_real_w8a8_weights_path_newpod`、`vllm_golden_dumps_are_prefill_not_decode`;`pypto-project/phases/27-n1-whole-net-fusion.md`(§Step-2 增量①+策略决定+refined 蓝图);严格遵守 `pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`。

## debug/开发约束
遇问题先看 DeepSeek/Qwen 实现怎么做(DeepSeek 用 multi-program 每层独立 scope 天然避 comm-window 别名;我们 N=1 单 program 是用户指定方向,不回退 multi-program),再查历史开发文档避免重复造轮子;多看地址对齐/padding/shape/dtype/layout。开始 coding 前先有整体架构脉络再落地,不满足约束可能是设计不合理需重设计。每完成关键节点更新 phase27 + push。
```
