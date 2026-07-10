# 下一阶段启动提示词 —— G1 Option-C 整网 decode 接管（续）

> 新 session 直接把下面 code block 作为第一条消息粘贴即可。自包含，不依赖记忆。
> 生成于 2026-07-10，承接 G1 device Pass 1/2（见 `archive/milestones-2026-Q2.md` "2026-07-10 (续)"）。

## worker 备份位置（in-tree 未提交，别丢）
- 工作树：`0162:/data/chensiyu/hw_project/pypto/workspace/pypto-lib/_stage_whole_decode_run.py`
- 持久备份（NFS）：`.../workspace/g1_worker_pass2_20260710_173720.patch` + `.../workspace/_stage_whole_decode_run.py.g1_pass2_bak_20260710_173720`

---

```
继续 pypto + vLLM 集成，目标不变：live single-handoff A/B —— 8001(pypto)跑完整 45 层
step3p5 decode，token-exact vs 8000(vanilla)。有进展及时同步到 pypto-project。

## 运行环境（权威，别搞错）
- 全部开发/验证在 0162：`ssh 0162`，repo `/data/chensiyu/hw_project/pypto/workspace/pypto-lib`，
  分支 `stepfun/develop @47c260e3`。**不要 rebase**。**不要用本地 b-csy-develop / feat/whole-net-n1-fusion**。
- fresh shell 三件套：`source /usr/local/Ascend/cann/set_env.sh`（非-GA symlink，**不是 beta.1**）
  `&& source /data/chensiyu/hw_project/pypto/workspace/activate.sh
  && export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa`。
- device 跑必设：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
- 8000 oracle 在 cards 0-7（别碰）；pypto 跑 cards 8-15，worker 用 `--tp 8 --dev-offset 8 -p a2a3`
  （**必须 -p a2a3；默认 a2a3sim 会要没装的 g++-15**）。
- cards 8-15 的 Aicore=100% 是 sticky 计数器假象，非 poison（L3 allreduce golden 已证）；
  真健康看 rc / "No process in device"。launch 前 `pgrep -af "[c]hip_process"` 确认空。
- **禁** `-9` 强杀 / `npu-smi set -t reset`。

## 约束（硬性）
- 架构走 Option-C 多程序（TP-attn program + select_moe_block moe_block 解耦），不走 n1-fusion。
- 动手前读 skill：`pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`。
- 启动 4-agent team：reverse-review / hw-analyst / sw-analyst / upstream-scout。
- debug 四板斧：先看 DeepSeek/Qwen（models/deepseek/v4）→ 上游是否已修 → 我们 kernel → dtype；
  多查地址对齐/padding/shape/dtype/layout。507018/507899 先查 simpler wiki。
- 精度 oracle = vLLM eager dump（synthetic 会 stale）；真 token-exact 靠 live A/B。
- 同步协议：进展改 STATUS.md/archive/milestones-2026-Q2.md，commit + push
  （`git -c http.version=HTTP/1.1 push`，PAT 在 /data/chensiyu/secrets/github.env，屏蔽 token）。

## 已完成（先核对，别重做）
- worker `_stage_whole_decode_run.py --worker`（in-tree 未提交，+~508 行；备份见本文件顶部）：
  N=7 Option-C DistributedWorker（dedup by id(select_moe_block)+kind）、3-scalar layer_idx
  （norm=绝对 / attn=type-local / mlp=dense-order）、worker 自补 post-RMSNorm(EPS=1e-5,Gemma +1.0)
  + next_hidden=resid1_fp32+moe_out、per-layer gate_exp 经 gate_r 槽、real W8A8 加载。
- Pass 1（synth 机制）device rc=0：4 层链全 5 步派发无 507018。
- Pass 2（真 W8A8）device rc=0：47GB 载入 + moe_out 0→3.5（真 MoE 专家跑通）。日志 /tmp/pass2_realw.log。
- 已修 5 bug：gate_exp L96 shape、double _expand_tp、w_g 非连续切片、
  fork 后共享内存必须 prepare() 前 share_memory_()、synth shape。
- 已同步 pypto-project（2e4a2ae）。

## 关键契约（读码已证，别信旧 memory）
- select_moe_block 返回的 EpTpMoE 不自带 norm/residual，只出 moe_out → worker 必须自补。
- gate_r 是真 per-head gate 乘子（非 ×1 旁路）→ 喂 sigmoid(RMSNorm(hidden)@w_g)，不是 ones。
- weight_loader：45-row norm 按绝对 layer_idx，42-row MoE 按 pos=layer-3，不混。

## 下一阶段任务（按顺序）
1. 接 dense/attn 真 W8A8 wiring（当前 dense/attn 仍 synth，只 moe_block 真；证据：Pass 2 dense/attn
   输出 ≈ Pass 1 synth 30.9/44.8/59.8）。改 build_dense_inputs(~605)+_moe_attn_sh(~437) 吃 bundle；
   per-rank gate（每 rank 从 bundles[r][KEY_WG][attn_li] 算）。重跑 --ckpt
   /mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp 4 层 → 成功信号 = 每层
   next_hidden 不再等于 synth 值。
2. torch-ref 逐层对拍：复用已 device-PASS 的 ST torch ref（test_decode_layer_full_dense_st /
   _swa_dense_st / test_decode_layer_moe_st），喂相同输入对拍。离线 attention 自洽可比（同 synth KV），
   vLLM-parity 才受 KV 阻塞。ratio_allclose atol=0.04 → 坐实 3-scalar split 多层正确性。
3. 修 L43/L44 standalone SplitIncoreOrch（扩 45 前）：根因 = _quant_moe_input(moe.py:1801) 用 pl.spmd
   在 InCore helper 泄漏 InCoreScopeStmt（#1828 拒，只 (7,0)/(7,16) 走到）。修法 Option C：
   @pl.function(InCore) → @pl.jit.inline，保留 pl.spmd body（对齐 DeepSeek gate.py + step3p5 自己的
   _expert_routed；数值与 L43 device-PASS 版 3b236e6 字节一致；可能 drop return x_out）。改后回归：
   standalone select_moe_block(43/44) 编译 + moe_block L44 精度 ST device + reverse-review。
4. 扩到 45 层链 device 跑通 + torch-ref 全层过 → G1 完成。
5. G2 _pypto_full_forward live wiring（vllm_monkey_patch.py:233，fail-closed）：常驻 DistributedWorker
   holder + import KV/权重 pool + 45 层 dispatch loop（常驻 DeviceTensor residual handoff）+ 读 live
   forward_context slot_mapping/block_table 进 attn args + final hidden copy 回。
6. G3 HBM 共存/gap-5：vLLM W8A8(24G)+pypto BF16(47G)=OOM → (a) standalone 先验；(b) gap-5 in-kernel
   INT8×INT8→INT32 dequant（primitive device-validated，但 cast→int8→cube codegen 上游未修 gated OFF；
   BF16-dequant 是工作路径）。
7. G4 co-tenancy 507018 测试 → G5 翻 8001 full + live A/B token-exact。

先起 team + 读 skill + ssh 0162 核对 worker 现状和 Pass 2 日志，再从任务 1 开始。
注意 sw-analyst 上个 session 卡过几次，盯紧增量、别空转 20 分钟；必要时 lead 直接接管跑 device。
```
