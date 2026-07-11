# Phase 20 — vLLM 后端 monkey-patch（e2e 流程）

> **Component pin snapshot（doc 创建时，2026-06-22）**
>
> | 仓库 | 分支 | Pin | 备注 |
> |------|------|-----|------|
> | pypto-lib | `stepfun/develop` | `a6b5faa`（pre-commit） | 加这文件的 commit 之后会更新 |
> | pypto | `stepfun/develop` | `b00c8b23` | DFX env hooks + repros |
> | pto-isa | `stepfun/develop` | `e25732f0` | = origin/main |
> | PTOAS | `stepfun/develop` | `da011a3d` | binary v0.45 |
> | simpler（submodule） | — | `a6e06406` | Phase 16 4 patches |
> | vLLM（参考） | stepcast/develop | `0e0901376` | 路径 `<workspace>/.../ascend/vllm/` |

## 目标

End-to-end decode 流程：vLLM offline_inference 调用走 **pypto step3p5
decoder kernel**（通过 `pypto.ir.compile` 编出），替代上游 torch eager
的 `Step3p5DecoderLayer.forward`。用户的 opt-in 入口是一次 Python 调用：

```python
import pypto.step3p5.vllm_backend as B
B.install()
from vllm import LLM, SamplingParams
out = LLM(model="<jfs>/step3p5_flash_release_hf_mtp3_bf16/").generate(
    "Hello", SamplingParams(max_tokens=16)
)
```

Phase 20 要让这条路径**不崩**，能返回任意 16 个 token。精度跟上游 vLLM
对齐是 Phase 21；perf 测量是 Phase 22。

## Scope

**In:**
- 整模型 monkey-patch 在 `Step3p5Model.forward`（一次融合 `decode_fwd`
  kernel call per forward，而不是 45 个 per-layer call）。
- 单卡（TP=1）路径，用 `apply_tp1_patch` 风格的 unsliced widths ——
  Phase 15 e2e 已经证明 dense layer 这条路径能跑通。
- Mixed-mode MoE：dense 层走 pypto；MoE 层（21/45）走 vLLM
  `FusedMoEBlock` 回退（host 侧切换）。
- HF safetensors 真权重 load，从
  `<jfs>/step3p5_flash_release_hf_mtp3_bf16/` 读。
- KV cache 和 attention metadata 从 vLLM forward context 桥接过来。
- `install()` / `uninstall()` / `is_installed()` public API。

**Out（延后）:**
- 多卡 canonical TP=8（Phase 22 + barrier all_reduce gate）。
- 全 pypto MoE（Phase 22 + MoE 507018 gate）。
- per-layer monkey-patch 粒度（保留 escape hatch 给 Phase 21 精度 diff
  harness，详见下面 "Per-layer escape hatch"）。
- MTP 集成（Phase 23+）。
- Tokenizer / sampler —— 已在 vLLM，不动。

## 关键决策

| # | 决策 | 原因 |
|---|------|------|
| D1 | **整模型 patch** 在 `Step3p5Model.forward`，不 per-layer | pypto `decode_fwd` 是一个融合 45-layer + lm_head 程序；45 次 launch 会抹掉融合优势 |
| D2 | **Comm option A**：pypto kernel 内部用 simpler shmem-IPC comm；vLLM 的 `tp_group` 在 pypto kernel 内不用 | 不用写 simpler↔HCCL bridge；pypto kernel 是 self-contained |
| D3 | Phase 20 **mixed-mode MoE** | MoE device 507018 是独立硬 blocker；不在 e2e 关键路径上 |
| D4 | 代码在 `pypto-lib/models/step3p5/vllm_backend/`；vLLM 仓不动 | 可逆，不污染 fork |
| D5 | 保留 per-layer hook 表面给 Phase 21 用（Phase 20 不用） | 允许 Phase 21 做 layer-by-layer hidden_states diff 而不用重架构 |

## 交付物

```
pypto-lib/models/step3p5/vllm_backend/
├── __init__.py              # public API: install(), uninstall(), is_installed()
├── install.py               # monkey-patch dispatcher，存原始版本待 uninstall
├── weight_translate.py      # vLLM nn.Module -> pypto bundle dict
├── kv_bridge.py             # vLLM kv_cache layout -> pypto k_cache/v_cache view
├── attn_meta_bridge.py      # vLLM AttentionMetadata -> pypto seq_lens/block_table/slot_mapping
├── compile_cache.py         # rank-aware 编译后 kernel cache（编一次，runtime_dir 复用）
├── mixed_moe.py             # MoE 层回退到 vLLM 的 FusedMoEBlock
├── config_align.py          # 校验 vLLM hf_config 与 pypto config.py 常量匹配
└── README.md
```

加退出测试：`pypto-lib/tests/step3p5/test_vllm_backend_e2e.py`。

## 任务

| # | 任务 | 输出 | 估时 |
|---|------|------|------|
| 1.1 | `config_align.py` —— assert pypto config.py vs vLLM `hf_config`（NUM_HIDDEN_LAYERS=48, HIDDEN=4096, NUM_HEADS_FULL=64, NUM_KV_HEADS=8, INTERMEDIATE=11264, VOCAB=128896, BLOCK_SIZE=128 等）；mismatch 报 diff | 干净 import | 1 d |
| 1.2 | `weight_translate.py:vllm_to_pypto_bundle(model)` —— 走 `model.named_parameters()`，按 `weight_loader.expected_shapes()` key 重 group；处理 `qkv_proj` Q/K/V 切分、`gate_up_proj` 切分、MoE `experts.w13_weight` 切分、vocab parallel embedding gather、lm_head TP 切 | 30-key bundle dict | 5 d |
| 1.3 | `kv_bridge.py:make_kv_views(vllm_kv_cache_layer_i)` —— vLLM `[num_blocks, block_size, num_kv_heads, head_dim]` BF16 → pypto `[KV_CACHE_ROWS_DYN, HEAD_DIM]` 零拷贝 `.view()`；assert layout stride 兼容 | (k_view, v_view) per layer | 3 d |
| 1.4 | `attn_meta_bridge.py:extract_pypto_meta(attn_metadata)` —— 从 vLLM `AttentionMetadata` 拿 `seq_lens / block_tables / slot_mapping`，dtype/shape 转 pypto 合约（INT32, flat block_table） | meta dict | 2 d |
| 1.5 | `compile_cache.py:get_compiled(rank, world_size)` —— 首次 call 时 `ir.compile(Step3p5DecodeFwd, distributed_config=DistributedConfig([rank,...]))`，持久化 `runtime_dir`；后续 call 复用 `.so`/`.bin` | callable + `runtime_dir` | 2 d |
| 1.6 | `mixed_moe.py:hybrid_forward_dispatcher(model, positions, hidden_states)` —— 全是 dense 层 → 调 pypto `decode_fwd`；有任何 MoE 层 → 对 dense 块调 pypto，再回 Python 对每个 MoE 层调 `Step3p5MLP.forward` / `FusedMoEBlock.forward`；复用 vLLM `Step3p5Model.forward` 骨架配 hot-swappable dispatcher | dispatcher | 3 d |
| 1.7 | `install.py:install()` —— 备份 `Step3p5Model.forward._orig = Step3p5Model.forward`，set 新 forward 调 `hybrid_forward_dispatcher`；patch `vllm.entrypoints.api_server` init 在 `PYPTO_STEP3P5_BACKEND=1` 时 lazy 调 `B.install()` | install/uninstall API + idempotent | 2 d |
| 1.8 | `tests/step3p5/test_vllm_backend_e2e.py` —— `B.install(); LLM(model="step3p5-flash-...").generate("Hello", SamplingParams(max_tokens=16, temperature=0.0))`；assert 返回 16 个 token id 不崩 | e2e smoke | 2 d |
| 1.9 | bring-up debug + 0162 device 0 首次绿 | Phase 20 准出 | 2 d |

## 准出条件

```bash
# 干净 0162 shell:
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa
export PYPTO_STEP3P5_BACKEND=1
cd <workspace>/pypto-lib
python -m tests.step3p5.test_vllm_backend_e2e \
    --model-path <jfs>/step3p5_flash_release_hf_mtp3_bf16/ \
    --prompt "Hello, the future of AI is" \
    --max-tokens 16 \
    -p a2a3 -d 0
# 期望：exit 0；打印 16 token id；无 fault；无 507018。
```

Token 输出**不**要求严格匹配上游 vLLM；Phase 21 负责精度。Phase 20 只
证明管道通。

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| vLLM `qkv_proj` 权重 layout（concat 顺序 Q,K,V）跟 pypto 分开的 `wq/wk/wv` 不对应 → 静默错误输出 | 高 | 任务 1.2 必须用合成测试验证维度切分再 integrate；打印 expected vs got shape |
| KV cache layout 不兼容零拷贝 → 每次 call 要 repack | 中 | Phase 20 接受拷贝；Phase 22 perf round 再优化 |
| vLLM `get_rope` 内部缓存 cos/sin 是全局；pypto kernel 期望 `[SEQ, ROTARY_DIM]` per-rank | 低 | 直接读 vLLM `RotaryEmbedding.cos_sin_cache`，喂进去 |
| `Step3p5Attention.use_head_wise_attn_gate=True` 但 pypto head_gate 走 × 1 旁路 | 数值（Phase 21 处理） | 在代码里 doc；Phase 21 §2.7 标定 |
| 首次编译 ~30s，vLLM 可能 warmup 超时 | 低 | `runtime_dir` cache `.so`/`.bin`；pre-compile 脚本 |

## Per-layer escape hatch

虽然 Phase 20 在整模型层 patch，install 层 expose：

```python
B.install(per_layer=True)  # 只 Phase 21 用
```

切到 patch `Step3p5DecoderLayer.forward`。每层 call 调一个更小的编译
kernel（`decode_layer_full_dense` 等，Phase 19 ST 验过）。慢一点但允许
hook 做 per-layer hidden_states diff 给精度验证用。

## Status

- 2026-06-22：设计已落（本 doc）。
- 任务 1.1-1.9 未启动。
- 关键路径。预计到 e2e 绿 ~3-4 周。

## References

- `pypto-lib/docs/known-pypto-pitfalls.md` —— kernel 硬限制
- `pypto-lib/docs/dev-workflow-gotchas.md` —— dev workflow 坑
- [`21-precision-validation.md`](21-precision-validation.md) —— Phase 21
  接力
- vLLM 参考：`<vllm_repo>/vllm/model_executor/models/step3p5.py` HEAD
  `0e0901376`
- pypto-lib 参考：`models/step3p5/decode_fwd.py:198 _build_decode_fwd_program`，
  `models/step3p5/weight_loader.py:197 expected_shapes`

---

## G2/G3/G5 live single-handoff 实现蓝图（2026-07-11，co-tenancy gate 已清后落地）

> 前置 gate **G4 co-tenancy 已解**（`SIMPLER_COMM_NO_HCCL=1`，见
> [`../deployment/cotenancy-simpler-no-hccl.md`](../deployment/cotenancy-simpler-no-hccl.md)）。
> 运行时机制已 device 验证 co-resident：NO_HCCL + resident prepared-rt 跨 step 复用 +
> real-weight dispatch（L0 torch-ref 1.000）。以下是把 offline worker 接进 live 8001 的具体设计。

### 架构决策：whole-decode ≠ per-op 拓扑

- **per-op live（phase 24，已工作）**：8 个独立 per-rank ChipWorker（`--device 8+r --rank r`），
  每个 vLLM rank-r 进程经 Unix socket `pypto_mlp_rank{r}.sock` 连自己的 worker，**vLLM 做 all_reduce**
  （`pypto_dense_mlp_backend.py:57 _WorkerClient` / `:137 forward` per-rank）。
- **whole-decode（本 phase）**：需 TP all_reduce + EP a2a **在 pypto 内部**跨 8 卡 → 必须是**一个
  DistributedWorker**（1 host 进程 + 8 chip children co-prepare，共享 comm domain），**不是** 8 个独立
  ChipWorker。→ 不能照搬 per-op 的 8-socket 模型。

### 单 handoff 数据流（TP hidden 层间 replicated 这一事实是关键）

vLLM TP 下每个 decoder-layer 边界的 hidden 在 8 rank 间 **replicated**（RMSNorm replicated、
attn/MLP shard 后 all_reduce 回 replicated）。pypto whole-decode worker 内部自己 TP 分片
（`_expand_tp` 把 [1,...] 复制到 [tp,...]）、自己 all_reduce、出 replicated 结果。所以整步只需
**一次** full-hidden handoff：

1. **sidecar** = 一个常驻 pypto DistributedWorker 进程（8 chip children on cards 8-15，`SIMPLER_COMM_NO_HCCL=1`），
   module-global 持 `rt`（manual `__enter__/__exit__`，resident `--steps` 机制已验证）。暴露 socket
   `pypto_whole_decode.sock`：收 full hidden `[BATCH,HIDDEN]` → 跑 45 层 → 回 full next hidden。
2. **`_pypto_full_forward`（vllm_monkey_patch.py:233，8 rank 都跑）**：driver = rank-0 连 sidecar 发
   hidden、收结果；其余 7 rank 经 `tensor_model_parallel_broadcast`（src=0）拿同一结果（因结果 replicated）。
   或 8 rank 各连 sidecar 拿同一 replicated 结果（sidecar 广播回）。**推荐 rank-0 drive + broadcast**（少 7 次拷贝）。
3. **final hidden** → vLLM `compute_logits`（tail 现委托 vLLM，可留；或后续接 pypto rms_lm_head）。

### G3 真 KV import（复用 phase 24 零拷贝 KV-IPC）

pypto sidecar 的 chip-r 需读 vLLM rank-r 的 paged KV。**每 rank 一条 IPC**（8 export / 8 import，
与 phase 24 per-op 已 token-exact 的 `attn_setup import_ipc` 同）：
- vLLM rank-r：`_allocate_kv_cache_tensors` 产「一 buffer + 一 key」（phase 23/24 的整池 map，
  避免 per-tensor MemPool OOM）；45 层合一 buffer → 1 key → sidecar chip-r `rt.import_ipc(key)` →
  `DeviceTensor(peer_base+offset)[block]`，`child_memory=True`。
- **forked chip 的 IPC import 必须在 child 进程 context 内**（父 import 的 ptr 在 child 读 0）。
- attn args（slot_mapping / block_table / seq_lens）从 live `forward_context` 取，每步随 hidden 一起
  发给 sidecar（现 offline worker 用 dummy KV → L17 residual NaN，input-independent；真 KV 消除之）。

### 实现顺序（每步可 device 验证）

1. **resident holder 重构**：把 `_stage_whole_decode_run.py::_run_worker` 的 build+prepare+dispatch
   抽成 `WholeDecodeHolder`（`build()` / `__enter__` prepare / `decode(cur, kv_args, fwd_ctx)->next` /
   `__exit__`）。offline worker 改用它跑通（standalone device rc=0）= 回归不退化。
2. **sidecar 进程 + socket 协议**：holder 包成常驻服务（收 hidden+attn_args → decode → 回 next）。
   先 dummy-KV standalone 验证 socket round-trip。
3. **G3 KV-IPC**：sidecar chip-r import vLLM rank-r KV（先 offline exporter/worker 对拍 bad_ratio=0，
   再 live）。
4. **`_pypto_full_forward` 接线**：install() lazily 建 sidecar client；rank-0 drive + broadcast；
   读 forward_context 进 attn_args；final hidden 回。
5. **G5 live A/B**：8001 `PYPTO_STEP3P5_PATCH_MODE=full`（先起 8001 等 HCCL init，再起 sidecar，
   sidecar 设 `SIMPLER_COMM_NO_HCCL=1`）；3-prompt vs 8000 token-exact；swa/MoE(L43/L44) 数值在此定论。

### 已验证可复用积木

- co-tenancy：`SIMPLER_COMM_NO_HCCL=1` + 重编 a2a3 runtime（simpler `878f3742`）。
- resident rt 复用：`_stage_whole_decode_run.py --steps N`（device ✓ co-resident ✓）。
- 47GiB 单 key 权重 IPC（STATUS 2026-07-07 ④）；phase 24 KV-IPC token-exact；tail live（phase 24）。
- 启动顺序 + 停机：[`../deployment/troubleshooting-8001-pypto-bridge.md`](../deployment/troubleshooting-8001-pypto-bridge.md)。

### G3 真 KV-IPC 具体 spec（2026-07-11 读码定位，device-gated 下一步）

worker 当前 KV = **dummy zeros**（`_stage_whole_decode_run.py:1025-1026`
`k_cache=v_cache=torch.zeros(1, MAX_SEQ_DEFAULT, HEAD_DIM, bf16)`，per-rank，build 时 baked、
fork-inherited），attn args 也是 dummy（`:1022-1024` seq_lens=ones / block_table=zeros /
slot_mapping=arange）。attention 消费点：`_TorchRefChain._attn` 附近的 `attention_full_inline`
调用（`:328-348` `k_cache_full=kc, v_cache_full=vc, seq_lens/block_table/slot_mapping`），
KV_HEADS_LOCAL=1（TP=8 per-rank）。

**G3 要改的三点**：
1. **KV 来源**：dummy zeros → 从 vLLM rank-r 的 paged KV pool **IPC import**（`rt.import_ipc(key)` →
   `DeviceTensor`）。vLLM paged 布局 `(num_blocks, block_size, 1, head_dim)` flatten =
   `[num_blocks*block_size, head_dim]`（phase 24 已 token-exact 证）。**每层一份**（45 层各自 KV；
   现 worker 单 k_cache 复用 → 需扩成 per-layer KV 输入，或一大池按 layer 切）。
2. **KV-rows ABI（最硬 gate）**：`MAX_SEQ_DEFAULT`（k_cache 行数）**必须 == vLLM `num_blocks*block_size`**
   （远大于 dummy 默认）。编译期常量 → sidecar build 时须按 live vLLM 的真实 num_blocks 定。
3. **attn args per step**：seq_lens/block_table/slot_mapping 从 live `forward_context` 取，随 hidden
   经 socket 发给 sidecar（协议从 fixed `[BATCH,HIDDEN]` 扩成 length-prefixed + attn-args blob）；
   sidecar 每 step `copy_` 进 sh 的对应输入。

**验证路径**：先 offline（仿 phase-24 `_stage_attn_e2e.py exporter/worker`：独立进程 export 合成 KV
→ sidecar import → decode → 对 torch golden `bad_ratio=0`），再 live（vLLM `_allocate_kv_cache_tensors`
整池 export → sidecar chip-r import rank-r KV）。forked chip 的 import **必须在 child context 内**。
真实 num_blocks + 布局只能对 **跑起来的 8001** 定 → 这步是 device-gated，须 live vLLM 迭代。

### G5 live 落地路径（2026-07-11 定位容器 patch 加载机制）

**容器无 pypto-lib mount** —— vllm-8001 只 mount `/logs`（host `/mnt/nvme1/.../step3p5_910b_w8a8_v001/`）。
patch 走 **`/logs/pypto_patch/sitecustomize.py`**：按 env var（`PYPTO_DENSE_MLP_BACKEND` /
`PYPTO_ATTN_BACKEND` / `PYPTO_KVPOOL`）加载**自包含**后端文件（各含 `maybe_autoload()` + `status()`）。
- **G5 packaging（net-new）**：`vllm_monkey_patch.py`（含新 `_WholeDecodeClient` + `_pypto_full_forward`）
  import 了 `models.step3p5.config`（BATCH/HIDDEN）——容器里没有。须做**自包含** `/logs/pypto_patch/
  pypto_whole_decode_backend.py`：内联 BATCH/HIDDEN 常量 + 客户端 + `maybe_autoload()`（调
  `install(mode="full")` 等价逻辑），+ sitecustomize 加一行 `_load_backend("PYPTO_WHOLE_DECODE", ...)`。
- **`_pypto_full_forward` 已加 profiling fallback**（sidecar-absent → 响亮回退 original forward + 计数，
  非 silent mask；once sidecar up → pypto 路径）→ 解 mode=full startup profiling 的 chicken-and-egg。
- **live 顺序**：起 8001 mode=full（profiling 走 fallback 存活）→ 等 ready → 起 sidecar
  （`SIMPLER_COMM_NO_HCCL=1 --serve`，先 dummy-KV 验证 plumbing 连通，再 G3 真 KV）→ 送 prompt。
- **里程碑拆分**：(G5a) plumbing 连通（sidecar 收到 live 请求 + vLLM 出 token，验证 embed/broadcast
  API，dummy-KV 数值 garbage）；(G5b) 真数值 token-exact（gate 在 G3 真 KV）。
- **hazard**：pypto AICore timeout → `aclrtResetDeviceForce` 卡级 nuke vLLM（见
  `../deployment/cotenancy-simpler-no-hccl.md` §hazard）；live 前确保 vLLM stream idle + pypto 不 timeout。

### G5a live 落地进度 + 硬经验（2026-07-11 首次实跑 mode=full）

已实装 + 部署（备份 `workspace/g5_{whole_decode_backend,start_8001_full,start_sidecar}_*`）：
- **自包含容器后端** `/logs/pypto_patch/pypto_whole_decode_backend.py`（内联 BATCH=16/HIDDEN=4096，
  无 pypto-lib import）+ sitecustomize 加 `_load_backend("PYPTO_WHOLE_DECODE", ...)`。**device 实测：
  8 rank 全部 autoload + install `Step3p5Model.forward -> sidecar` 成功**。
- **mode=full launch** `/logs/start_8001_full.sh`（`PYPTO_WHOLE_DECODE=1` + `PYPTO_WHOLE_DECODE_SOCK=
  /logs/pypto_whole_decode.sock` + util 0.5 + enforce-eager）。

**踩过的坑（下次直接避开）**：
1. **socket 必须在 `/logs`（共享 mount），不能 `/tmp`**：容器 `/tmp` 与 host 不共享；per-op 也用 `/logs`。
2. **fallback 必须 COLLECTIVE**：`_pypto_full_forward` 8 rank 都跑。原版只 rank-0 查 sidecar + fallback，
   rank≠0 直奔 `tp.broadcast` → rank-0 fallback 不 join broadcast → **HCCL broadcast 超时 error 9 / TP
   deadlock（shm_broadcast "No available block in 60s"）**。修：**每 rank `os.path.exists(sock)` 一致判定**，
   absent（profiling 期）→ 全 rank fallback 到 original forward（不 broadcast）。已修 backend + pypto-lib 版。
3. **sidecar sock profiling 期必须 absent**：startup `_dummy_run` 会调 patched forward；sock 在 → 尝试连
   死 sidecar → broadcast 分叉。启 8001 前 `rm -f` sock，让 profiling 走 fallback，8001 ready 后再起 sidecar。
4. **停 8001 要 `pkill -9 -f VLLM::EngineCore`（不只 `vllm serve`）**：只杀 serve 会**孤儿化 EngineCore
   子进程**，多个 EngineCore 抢 cards 8-15 → shm_broadcast 争用 hang。杀后 `npu-smi` 确认 HBM 归 0。

**下一步**：clean boot（cards 空 + collective fallback + sock absent）→ profiling 走 fallback → ready →
起 sidecar（`/logs` sock）→ 送 prompt 验证 plumbing 连通（rank-0 drive + broadcast，dummy-KV garbage
数值但验 embed/broadcast API + socket 路径）→ 即 G5a。G5b token-exact gate 在 G3 真 KV。

### G3 真 KV — export 侧已存在（pypto_kvpool_backend），只剩 sidecar import（2026-07-11 定位）

**vLLM KV export 侧已建好，可直接复用**：`/logs/pypto_patch/pypto_kvpool_backend.py`（Phase 24.1，
`PYPTO_KVPOOL=1` 启用）已 patch `_allocate_kv_cache_tensors` → 把所有目标 attention 层的 K/V 分配进
**一个 MemPool 的一个 buffer**（块基址可导出）→ `aclrtIpcMemGetExportKey` 导出**单 key** + offset map
（`/logs/pypto_kvpool.key` + `/logs/pypto_kvpool_map.json`，每层 K/V 的 offset）。→ 解 per-tensor
MemPool 207001 OOM。

**G3 剩 = sidecar import 侧**（net-new，但 export 免写）：
1. 8001 mode=full launch 加 `PYPTO_KVPOOL=1`（+ 与 whole-decode 后端共存；vLLM 导出 KV 池 key/map）。
2. **sidecar 每 rank `rt.import_ipc(key)`**（import_ipc 已在 stepfun/develop `1aa6efb`）→ 按 map offset
   切每层 K/V `DeviceTensor` → 替换 worker build 里的 dummy `k_cache/v_cache=zeros`（`_stage_whole_decode_run.py:1025`）。
   forked chip 的 import **必须在 child context 内**。
3. **KV-rows ABI**：sidecar k_cache shape 行数 = vLLM `num_blocks*block_size`（从 map/vLLM 配置读），
   layout 对齐 vLLM paged `(num_blocks, block_size, 1, head_dim)` flatten（phase 24 per-op 已 token-exact）。
4. **attn args per request**：sidecar 协议扩 length-prefixed，随 hidden 收 slot_mapping/block_table/seq_lens
   （从 live forward_context）→ 每 step copy 进 sh。
5. sidecar `--layers 0..44`（全 45 层，真 W8A8 `--ckpt`，sharded ~6GB/卡 + vLLM util 0.5 ~28GB fits 64GB）。
→ 然后 G5b：3-prompt A/B vs 8000 token-exact。**这步须对 running 8001 device 迭代（KV layout/num_blocks
只能对活的 8001 定）= 专门 session**，但 export 免写 + G5a plumbing 已通 → 范围收敛到 sidecar KV-import。

### G3 KV export ABI — 已从 live 8001 pin 死（2026-07-11，PYPTO_KVPOOL=1 + mode=full 实跑）

8001 mode=full + `PYPTO_KVPOOL=1` boot READY_200，KV export 成功（8 rank 各 `consolidated 45 layers ->
ONE buffer ... ONE key exported; map entries=90`）。map（`/logs/pypto_kvpool_map.json.rank{0-7}`，
root-owned）实测布局：
```
{"rank":0, "pool_base":<VA>, "pool_bytes":15019868160 (14324 MiB), "num_layers":45,
 "map":{"L0.K":{offset:0,          nbytes:166887424, shape:[166887424]},
        "L0.V":{offset:166887424,  nbytes:166887424},
        "L1.K":{offset:333774848,  ...}, "L1.V":{offset:500662272, ...}, ... 90 entries (45×K/V)}}
```
- **每 K/V per layer = 166887424 bytes, flat, stride 166887424**；single buffer 14324 MiB/rank；
  **一个 IPC key/rank**（`pypto_kvpool.key.rank{r}`）。
- KV-rows ABI：166887424 B / head_dim(128) = **1,303,808 KV slots**（若 INT8 kv_cache；= num_blocks×block_size，
  block_size=128 → num_blocks=10186）。sidecar k_cache `MAX_SEQ_DEFAULT` 须 = 该值（编译常量）。
- → **G3 sidecar import 无运行时未知量了**：每 rank `rt.import_ipc(key.rankR)` → 按上表 offset 切每层
  K/V `DeviceTensor`（166887424 B）→ 喂 `k_cache_full/v_cache_full`（替 dummy zeros），配 live
  forward_context 的 block_table/slot_mapping/seq_lens。剩纯实现 + token-exact 验证（对 8000）。

### G3 import mechanism = 复用 `WeightIpcMap`（2026-07-11 确认，proven building block）

sidecar KV-import 不用从零写：`tools/step3p5/pypto_weight_ipc.py::WeightIpcMap`（`:358-414`）已实现**正是
需要的零拷贝 import 模式** —— `rt.import_ipc(key, worker_id=r)` → `peer_base`（int）→
`DeviceTensor(peer_base + offset, shape, dtype)` per map offset（`device_tensor.DeviceTensor`）。已在 47GiB
权重池 + per-op `_stage_attn_worker.py::attn_setup`（phase 24 token-exact）验证。`import_ipc` API 在
`distributed_runner.py:1073`（`def import_ipc(self, key, *, worker_id=0) -> int`），已在 stepfun/develop。

**→ G3 剩纯装配（全部 building block 已 proven + reusable，零未知）**：
1. 8001 launch `PYPTO_KVPOOL=1`（export 已验证）。
2. sidecar 仿 `WeightIpcMap` 建 `KvIpcMap`：读 `pypto_kvpool.key.rankR` + `pypto_kvpool_map.json.rankR`
   → per-layer K/V `DeviceTensor`（offset 见上表）。**在 chip child context 内 import**。**已写：
   `tools/step3p5/pypto_kv_ipc.py::KvIpcMap`（AST-verified；`from_files(key,map,rt,worker_id) →
   kv_device_tensors(layer)→(k_dt,v_dt) [1,num_slots,head_dim]`；num_slots=nbytes//itemsize//head_dim，
   itemsize/dtype 可配 int8 或 bf16 —— 下 session 对 8000 定 kv_cache_dtype）**。
3. 用这些 DeviceTensor 替 `_ordered_args` 里的 dummy k_cache/v_cache（per-op attn_setup 已证 rt.run 收 DeviceTensor）。
4. sidecar 程序按 `MAX_SEQ_DEFAULT`=KV-slot 数（166887424/head_dim/itemsize）重编（tile-shape ABI）。
5. socket 协议 length-prefixed 带 block_table/slot_mapping/seq_lens；sidecar `--layers 0..44` 真 W8A8。
6. 对 8000 token-exact A/B（G5b）。**全 device-iteration，须 live 8001，= 专门 session。**

**G5b 精确 wiring 点（读码 pin，2026-07-11，line-level）**：
- **ABI 覆盖**：build 前设 `config.MAX_SEQ_DEFAULT = num_slots`（现默认 4096；`config.py:83`）→ 编译出的
  k_cache 输入 shape = `[1, num_slots, HEAD_DIM]`；`MAX_BLOCKS_PER_SEQ` 随之（`config.py:328`）。
- **喂 KV**：`_ordered_args`（`_stage_whole_decode_run.py`）按 param name 取 `sh[name]` → decode step 前
  `sh["k_cache"]=k_dt; sh["v_cache"]=v_dt`（KvIpcMap.kv_device_tensors(layer) 的 DeviceTensor）替 dummy。
  per-op attn_setup 已证 rt.run 收 DeviceTensor 混 host tensor。
- **dtype**：worker dummy k_cache 现 `bf16`（build），vLLM W8A8 kv_cache dtype 待 live 确认（若 int8 则
  KvIpcMap itemsize=1 → num_slots=1303808 且 attention kernel 需 int8-KV 读取/dequant；若 bf16 →
  itemsize=2 → num_slots=651904 直配）。**这是唯一须对 live 8001 定的量**。
- **attn args**：socket 协议扩 length-prefixed，client（`_pypto_full_forward`）随 hidden 发
  forward_context 的 block_table/slot_mapping/seq_lens；sidecar 每 step `sh[...]=...`。
- **全 45 层**：sidecar `--layers 0,1,...,44 --ckpt <w8a8>`（真权重，sharded ~6GB/卡 fits）。

### G5b 架构 crux 已解 — 每-rank KV feed = `StackedDeviceTensor`（2026-07-11 读码确认）

之前担心「8 rank 各自 vLLM KV 是 8 个独立 IPC buffer，无法喂一个可切片 k_cache」——**runtime 已原生支持**：
- `distributed_runner.py:1073 import_ipc(worker_id=r)` 在 **chip-r 的 fork child ACL context 内** import，
  返回 chip-r 有效指针 → 可 back `DeviceTensor(child_memory=True)`，零拷贝。
- `device_tensor.py:147 StackedDeviceTensor(shards, full_shape, worker_ids)`：leading-dim 堆叠，
  `shards[r]` 是 chip-r 上的 DeviceTensor，`worker_ids==range(tp)`（canonical device=r identity）→
  host_orch `x[r]` 切片路由到 chip-r，`child_memory=True` 跳过 H2D。**这正是每-rank KV feed 机制。**
- **importer 已补全**：`pypto_kv_ipc.py::build_stacked_kv(kv_maps, layer_idx)` → per-layer (k, v)
  `StackedDeviceTensor`（每 rank import_ipc(worker_id=r) 的 KvIpcMap → 8 shards）。AST-verified。

→ **G5b 每个机制现已确认 + 代码就绪**：KvIpcMap + build_stacked_kv（写好）；per-rank import（runtime 支持）；
MAX_SEQ_DEFAULT 覆盖（config.py:83）；`sh["k_cache"]=StackedDeviceTensor` feed（_ordered_args by name）；
socket attn-args 协议扩展。剩：装配 + 对 live 8001 定 kv dtype + 45 层 + token-exact A/B。纯 device-iteration。

### ⚠ G5b 新发现：KV layout 是 multi-head paged，与 worker 单-head 假设不符（须 device 对齐/可能改 kernel）

boot log：`GPU KV cache size: 162,944 tokens`（= 1273 blocks × 128 block_size）。KVPOOL 每 K/V/layer =
166887424 B → **1024 B/slot** = `num_kv_heads_local × head_dim(128) × itemsize`：
- bf16(itemsize=2) → num_kv_heads_local=4；int8(itemsize=1) → num_kv_heads_local=8。
- **worker 的 `k_cache=[1, MAX_SEQ, HEAD_DIM]` 假设 单 KV head**；vLLM 存的是 **多头 paged**
  `[num_blocks, block_size, num_kv_heads_local, head_dim]`。→ **layout 不一致**：sidecar 的 attention
  须按 vLLM 的多头 paged 结构读（num_kv_heads_local + block-based addressing），**可能要改 attention kernel**，
  不是简单换 DeviceTensor。这是 G5b 的真实工程量核心，须 live device 迭代对齐 + 数值验证。
- 精确 num_kv_heads_local / dtype / block layout 须对 running 8001 的 KV tensor spec 定（KVPOOL backend
  的 copy 逻辑 + vLLM kv_cache spec）。**→ G5b 确认是 dedicated device session**（非纯装配）。

### ⭐ G5b 核心 blocker 精确定性（2026-07-11，config + live 数字算死）—— KV-layout 根本不匹配

- **worker attention**：`NUM_KV_HEADS=8, TP=8 → KV_HEADS_LOCAL=1`/rank（config.py:363），
  `k_cache=[1, MAX_SEQ, 128] bf16` → **256 B/slot**（1 head×128×2B），TP-**sharded**、flat。
- **vLLM live KV**：166887424 B / 162944 slots = **1024 B/slot** = num_kv_heads_local×128×itemsize = 8
  → **8 heads/rank int8** 或 **4 heads/rank bf16**（1 head/rank 会得 itemsize=8，无效 → 排除）。即 vLLM KV
  **未 shard 到 1 head/rank**（GQA KV 在 TP 下 replicate）+ 很可能 **int8**，paged
  `[num_blocks, block_size, heads, head_dim]`。
- **→ 根本不匹配**（head 数 1-vs-4/8 + dtype bf16-vs-int8 + flat-vs-paged + sharded-vs-replicated）。
  **G5b 不是接线**，须二选一：(a) 改 worker attention kernel 直接吃 vLLM 多头 paged(int8) KV
  （较大 kernel rework）；或 (b) 每 step KV re-layout（transpose/shard/dequant vLLM KV → worker 布局，
  贵）。两者都须 device 迭代验证。**这是 G5b 的真实工程量核心，phase-24 级 dedicated session。**
- 下 session 先做的事：读 KVPOOL backend copy 逻辑 + vLLM-ascend step3p5 kv_cache spec，确认 exact
  head 数/dtype/paged 结构 → 定 (a) vs (b) → 实现 + token-exact 验证。

**ground-truth 确认（2026-07-11，读 KVPOOL backend copy 代码）**：`patched()` 存
`"shape": list(k.shape)` + `nbytes = k.numel()*k.element_size()`。map 记录 `shape=[166887424]` == `nbytes`
→ **element_size=1**，即 vLLM-ascend 把每层 per-rank KV 存成**扁平 int8 字节 buffer**（166887424 B/layer）。
per-slot = 166887424/162944 = **1024 B** = heads×128×itemsize → 8 int8 heads 或 4 bf16 heads/rank。
→ **mismatch 已从导出代码坐实（非仅算术）**：worker 1-head/rank ≠ vLLM 4/8-head/rank。仍须读
vLLM-ascend kv_cache spec 定 logical dtype（int8 native 还是 bf16-as-int8-bytes）+ paged 维序，才能选 (a)/(b)。

**⭐ vLLM KV 精确 shape 已定（2026-07-11，vLLM-ascend `attention_v1.py:109`）**：
`get_kv_cache_shape() → (2, num_blocks, block_size, num_kv_heads, head_size)`（leading 2 = K/V 合一，
故 KVPOOL backend 用 `kv[0]`/`kv[1]` 拆）。所以每层 per-rank K = `[num_blocks=1273, block_size=128,
num_kv_heads_local, head_dim=128]`，`num_kv_heads_local × itemsize = 8`（8 int8 或 4 bf16 heads/rank）。
model config：`num_attention_heads=64(full)/96(swa)`, `head_dim=128`, `torch_dtype=bfloat16`；worker
`NUM_KV_HEADS=8, KV_HEADS_LOCAL=1`。
- **vLLM KV**：paged `[2, num_blocks, block_size, num_kv_heads, head_dim]`（block-结构、多头、K/V 合一，likely int8）。
- **worker KV**：`[1, MAX_SEQ(=num_blocks×block_size), head_dim]`（单头、flat-over-blocks、K/V 分离、bf16）。
- **G5b 桥接工作**：worker attention 须改成读 vLLM 的 `[2,nb,bs,heads,hd]` 布局（多头 + block/block_pos 维 +
  K/V 合一 + 可能 int8 dequant），或每 step re-layout。exact int8-vs-bf16 + num_kv_heads_local 下 session 对
  running 8001 的 `layer.kv_cache` tensor `.dtype/.shape` 一读即定（attention_v1.py get_kv_cache_shape 传入的
  num_kv_heads = model num_kv_heads // tp）。这是唯一剩余 device 待读量；桥接 kernel 是 G5b 主工程。

**⭐⭐ int8 + 8-heads/rank 定死（2026-07-11，vLLM-ascend attention_v1.py 源码）**：
- **KV cache = int8**：`_quantize_kv_to_int8`（:1353，`clamp().to(torch.int8)`）在 forward 调（:1224/:1269），
  cache path `if key.dtype == torch.int8`（:1538）。→ itemsize=1 → num_kv_heads_local = 8（H×I=8）。
- → **vLLM live KV = int8, 8 KV heads/rank, [2, 1273, 128, 8, 128], KV 未 shard 到 1/rank（8 heads 全留每 rank）**。
- vs **worker = bf16, 1 KV head/rank, flat, KV-sharded**（KV_HEADS_LOCAL=1，假设 8 heads 跨 TP=8 各 1）。
- **⚠ 这是比"布局不同"更深的问题**：worker 的 whole-decode attention **TP 并行模型（KV-sharded 1/rank + TP-reduce）
  与 vLLM 的（KV-replicated 8/rank）根本不同**。worker 每 rank 只算 1 个 KV head 的 attention；vLLM 每 rank 有 8 个。
  → **G5b 不是"接 KV import"，而是 worker attention 的 TP/KV 模型重设计**（每 rank 算全 8 KV heads + int8 dequant +
  `[2,nb,bs,8,128]` combined-KV paged 读）。offline 特性刻画到此完整；G5b = 专门 device session 做 attention 重设计 +
  token-exact 验证。**注**：也须复核 worker 现有 attention 数值（G1/G5a 对的是 worker 自洽 torch-ref，非 vLLM 真 KV 模型）。
