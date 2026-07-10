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
