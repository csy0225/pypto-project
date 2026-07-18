# Phase 28 — N=1 整网 → vLLM live single-handoff 集成

> 承 Phase 27（N=1 整网融合 `whole_decode_faithful_real`，分支
> `feat/whole-net-n1-fusion`，最终 standalone release 在机器 0162）。本 phase 把已
> canonical 20/20、每次 `argmax=303` 的 N=1
> whole-net decode **接进 vLLM serving 路径**（M5/M6：live single-handoff A/B）。
> 严格遵循 SKILL §H：**N=1 单 `@pl.program` 是唯一生产形态**（多程序永久排除）。
>
> **组件 pin（2026-07-18）**：standalone 当前 stable 已升级为
> rank-local single-submit：pypto-lib
> `sync/whole-net-mtp3-53a6732@3af13f4facbe8db5cd4a6c769e8b9e07e351c7b9`
> + pypto
> `fix/n1-inline-orchestration-helpers@e49ce111c1503f4fb3e898af4223560cab907a62`
> + simpler/runtime `36957c6b56700ecba3aeb8dbbedd6240594e01de`。程序为
> `models.step3p5.decode_layer_single_chip:whole_decode_faithful_real_single_chip`，
> whole-net 底座仍是 `dispatch fixed-slot pull + combine pull`、native W8A8
> IPC weights、KV IPC。0162 P42 repeat20 全部 `argmax=303`，日志
> `workspace/logs_n1/single_submit_cleanup_p42_repeat20_20260718_174948`。
>
> **提交边界**：`0e7a0fdd` 只发布 standalone 核心的
> `decode_layer.py`、`moe.py`、`_gen_faithful_real.py`。下表中的
> holder/sidecar/KV importer/容器 backend 属于 2026-07-15 live 集成工作区记录，
> **不在该 release commit 内**；下一 session 必须先整理、复核并单独提交这些 live
> 组件，不能把 standalone commit 当作完整 live 集成交付。

## Goal

真实 vLLM 请求的 decode step 走 pypto N=1 whole-net（token-exact vs vanilla 8000）。
准出 = live A/B **L3 greedy top-1 ≥ 95%**（L1 hidden atol=0.04 / L2 cos≥0.999 辅证）。

## 架构（live single-handoff）

```
vLLM(8001, mode=full, enforce_eager)          pypto sidecar (co-resident, cards 0-7)
  Step3p5Model.forward (patched)                WholeDecodeHolder (build+prepare 一次)
    embed(input_ids) -> hidden [T,HIDDEN]         resident weights(IPC) + gate_r
    rank0: socket send hidden ───────────►        recv hidden -> set_hidden
                                                   holder.run() = whole-net 45L + tail
    rank0: recv next_hidden ◄───────────          send next_hidden
    tp.broadcast(next_hidden, src=0)
  compute_logits (patched) = norm + lm_head
```

- **co-tenancy**：sidecar 设 `SIMPLER_COMM_NO_HCCL=1`（跳过 HCCL control comm，file_barrier+IPC
  不变），与 vLLM 的 HCCL world 同卡共存。见 `deployment/cotenancy-simpler-no-hccl.md`。
- **seam**：patch `Step3p5Model.forward`→sidecar（返回 next_hidden，pre-norm）；`compute_logits`
  用 vLLM validated tail（norm+lm_head）。whole-net 内部 lm_head 只作 debug argmax。

## 已落地组件（本 session，offline/standalone 验证）

| 组件 | 文件 | 验证 |
|------|------|------|
| resident holder | `pypto-lib/tools/step3p5/whole_decode_holder.py` | 2026-07-15 工作区 device compile OK；**未包含在 `0e7a0fdd`** |
| sidecar + 协议 | `pypto-lib/tools/step3p5/whole_decode_sidecar.py` | 2026-07-15 工作区 offline `--selftest` PASS；**未包含在 `0e7a0fdd`** |
| monkey-patch full seam | `pypto-lib/tools/step3p5/vllm_monkey_patch.py::_pypto_full_forward` | py_compile OK；collective fallback + rank0 drive/broadcast |
| 容器 self-contained backend | `pypto-lib/tools/step3p5/pypto_whole_decode_backend.py` | 2026-07-15 工作区 py_compile + status() OK；**未包含在 `0e7a0fdd`** |
| co-tenancy patch | `pypto/runtime/.../comm_hccl.cpp` `SIMPLER_COMM_NO_HCCL` | 重编 exit 0；flag-OFF 无回归 + flag-ON bootstrap device-validated |
| KV-bridge importer | `pypto-lib/tools/step3p5/pypto_kv_ipc.py` | 2026-07-15 工作区 selftest PASS；**未包含在 `0e7a0fdd`** |

## Stage 4 设计 — per-layer KV + KV-bridge（下 session 执行；先设计）

**根因边界（本 session 读码坐实）**：`whole_decode_faithful_real` 现在 **45 层共享 ONE
`k_cache/v_cache [KV_CACHE_ROWS_DYN=4096, HEAD_DIM] bf16`** → ctx=1-only（每层对自身
position-0 K/V 自注意力，无 history；argmax==303 数值成立）。real multi-token decode 须
per-layer KV + 导入 vLLM 已 prefill 的 KV。

**设计决策**：
1. **KV 来源 = vLLM prefill 的 paged KV（IPC 导入，非 pypto 自算 prefill）**：vLLM 做
   prefill 填充其 paged KV pool；pypto decode 经 IPC 导入 rank-r 的 KV。phases/20 §G FINAL
   device-verified：vLLM KV = **bf16、1 KV head/rank、per-layer flat `[nb*bs, head_dim]`**
   （**不是 int8** → 无 dequant，纯 reshape）。
2. **per-layer KV 接法 = 导入 vLLM per-rank KV 大池（ONE key）+ 按 map offset 切每层**（选 (b)，
   对齐 vLLM 布局；不用 90 个 per-layer arg 的选 (a)）：
   - vLLM 侧：port `pypto_kvpool_backend.py`（从 0162/stepfun-develop）→ patch
     `_allocate_kv_cache_tensors` = 45 层 K/V 进 ONE buffer → `aclrtIpcMemGetExportKey` ONE key +
     offset map（`pypto_kvpool.key.rank{r}` + `pypto_kvpool_map.json.rank{r}`：`L{i}.K/V`→
     {offset,nbytes,shape}）。
   - pypto 侧：写 `pypto_kv_ipc.py::KvIpcMap`（**镜像 `pypto_weight_ipc.py::WeightIpcMap`**：
     `from_files(key,map,rt,worker_id)`→`import_ipc`→`peer_base`；per-layer `DeviceTensor(peer_base+offset,
     [nb*bs, head_dim], bf16)`）+ `build_stacked_kv(maps, layer)`→per-layer StackedDeviceTensor(8 shards)。
   - whole-net host_orch：把每层 attention 的 k_cache/v_cache 从共享单 buffer 改成 per-layer KV
     DeviceTensor（generator `_gen_faithful_real.py` — docstring 已述"each layer method receives its
     OWN per-layer KV cache"，per-layer 线程已存在，改 binding 即可）。
3. **ABI**：`config.KV_CACHE_ROWS_DYN = nb*bs`（live num_blocks*block_size，须对 running 8001 pin）→
   recompile。`block_table`/`slot_mapping`/`seq_lens` 每 step 从 live forward_context 经 socket 传
   （sidecar 协议已支持 meta_* tensor）。
4. **attention read**：现 flat `k_cache[slot_mapping]` 读；vLLM paged `[nb,bs,1,hd]` flatten `nb*bs`
   后 block_table 索引等价（phases/20 per-op 已 token-exact）→ 大概率兼容，**device 验证确认**。

**buffer/边界注意（用户强调）**：
- 每层 KV = vLLM per-rank pool 的 distinct offset slice（vLLM 分配 45 层 KV 互不别名）。
- dtype **bf16**（vLLM KV 是 bf16，无 quant/dequant → 与"禁 bf16-dequant 历史版本"不冲突：那条针对
  **权重**；KV 本身在 vLLM 就是 bf16）。
- 512B 对齐：每层 K/V row = head_dim(128)×2 = 256B；nb*bs 是 block_size(128) 倍数 → 池对齐。
- 多 batch：block_table `[BATCH, MAX_BLOCKS_PER_SEQ]` per step；单 batch T=1。
- 内存初始化：vLLM prefill 已写 KV；pypto decode 只读 context + 写当前 step 的新 K/V（add_inout）。

**为何必须单独验证（不能盲写）**：num_blocks/KV 精确 shape 必须对 live
vLLM pin；per-layer KV model change 会修改 whole-net KV 接口，必须先重跑
standalone canonical 回归，再做 live HBM 与 token-exact 验证。现有 standalone
release 已 20/20，但不能自动证明新 KV ABI 正确。

## Stage 5b runbook — live A/B（下 session，gated on Stage 4 + HBM closure）

1. 部署 `pypto_whole_decode_backend.py` → 容器 `/logs/pypto_patch/` + sitecustomize 加
   `_load_backend("PYPTO_WHOLE_DECODE", ...)`。
2. 起 8001 mode=full（`PYPTO_WHOLE_DECODE=1` + sock 路径；profiling 期 sock absent → collective
   fallback 存活）→ health=200。
3. 起 sidecar：`python -m tools.step3p5.whole_decode_sidecar --serve --sock /logs/pypto_whole_decode.sock
   -d 0-7 --kv-ipc --ckpt <w8a8>`（`SIMPLER_COMM_NO_HCCL=1`）。
4. 3-prompt greedy(temp=0) A/B vs 8000 vanilla → L3 top-1≥95%。
5. hazard：pypto AICore timeout → `aclrtResetDeviceForce` 卡级 nuke 同卡 vLLM（见
   `deployment/cotenancy-simpler-no-hccl.md §hazard`）；停 sidecar 后须 restart 8001。

## Stage HBM — 3-way 冗余权重消除设计（2026-07-17，machine-independent，先设计）

**实测 footprint（0234/0162 一致，per card 64GB）**：

| 常驻块 | 大小/卡 | 来源 |
|--------|---------|------|
| pypto exporter 池（native W8A8 IPC） | **25.35 GiB** | `WeightIpcExporter.export`，45 层 packed，routed INT8 + FP32 scale |
| vLLM 常驻 W8A8 | ~24 GiB | vLLM 自身 model loader（HF `[out,in]` W8A8_DYNAMIC） |
| whole-net run working set | ~16 GiB | ≈4×`PTO2_RING_HEAP` |
| **合计** | **~65 GiB > 64** | 三方 OOM 207001（device-proven 2026-07-15） |

**关键判定（读码坐实，`weight_loader.py`）**：pypto↔vLLM 权重**布局不同 → 不能纯 IPC remap**：

- attention `_slice_q/kv/o/g_proj` = **transpose(0,1)** + TP=8 per-rank slice + contiguous（HF `[out,in]` → pypto `[HIDDEN, LOCAL]`）；
- MoE routed `_transpose_routed_block` = **transpose(-2,-1)** + EP-slice（每卡 36 expert）+ contiguous；
- g_proj zero-pad 到 16；全部 pack 进 ONE 连续池按 offset 寻址。
- vLLM 保留 HF `[out,in]` 供其自身 W8A8_DYNAMIC kernel。

→ 直接 import vLLM 常驻权重不可行（转置/切分/pad/pack 全不同）；on-device repack 需第二 buffer（正是要消除的 26GB，自相矛盾）。

**消除方案（选定，反向）**：whole-net **替换** vLLM 的 45 层 decode forward，故 vLLM **不需要**其 decode-layer 权重常驻。vLLM 仅保留 tail：`embed_tokens [VOCAB,HIDDEN]` + `final_norm` + `lm_head [VOCAB_LOCAL,HIDDEN]`（`compute_logits` 用）。

- **tail footprint 精算**（VOCAB=128896, HIDDEN=4096, TP=8, bf16）：per-rank vocab-parallel embed 0.123 + lm_head 0.123 + norm(fp32) 16KB = **0.25 GiB**；worst-case replicated embed = **1.1 GiB**。
- 消除后预算：exporter 25.35 + tail 0.25 + run 16 ≈ **41.6 GiB < 64** ✅（~22 GiB headroom）。
- 唯一权重副本 = pypto exporter 池（native W8A8，满足"禁 bf16-dequant"）。
- **buffer 边界**：decode 权重只在 pypto 池；vLLM tail 与 pypto 池不同 VA、无别名；embed 读 / lm_head 写各自独立 buffer。

**待落地（vLLM 侧，gated on 选定 substrate 机器 + 用户 steer）**：让 vLLM 不 materialize / 加载后释放 45 层 decode module params（只保留 embed+norm+lm_head）。三条候选：(a) patch model loader 跳过 decode-layer 权重加载 + 不构造 decode module（config `num_hidden_layers` 影响 embed/tail 索引，需谨慎）；(b) 加载后 `del` decode 权重 tensor + 触发 allocator 回收（Ascend allocator 释放行为待验证）；(c) 构造期 stub decode module（param=空 view）。**先在 substrate clean 机器上量 vLLM tail-only 实际驻留，再选 (a)/(b)/(c)。**

## Status

- 2026-07-16：Phase 27 standalone gate 已关闭；0162 canonical P42
  **20/20 PASS**（pull + pull，native W8A8/KV-IPC，`argmax=303`），final
  smoke PASS；standalone random stall 不再是当前 gate。
- Stage 4（per-layer KV bridge + whole-net model-side KV binding）仍未执行；
  当前需从 live vLLM paged KV pool 导入每层 BF16 KV，并传递真实
  `block_table`/`slot_mapping`/`seq_lens`。
- 当前 live HBM blocker 是 vLLM 常驻 W8A8 权重、exporter whole-net
  INT8 权重和 runtime working set 的重复占用；须先消除 redundant weights
  或实现共享/in-place 权重方案。
- Stage 5b（live A/B）仍未执行；live single-handoff token-exact
  **尚未完成**。不得把 standalone 20/20 写成 serving 已完成。
- live 组件当前还缺一份经过 review 的独立 commit/pin；先整理提交，再开始
  per-layer KV 与 HBM 攻坚。
