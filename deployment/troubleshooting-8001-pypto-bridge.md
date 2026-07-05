# Troubleshooting —— 8001 PyPTO bridge（dense/shared/attention live 服务）

> 在 `gpu-a910x-0162` 上把 PyPTO 真实 kernel 接进 live vLLM（patched 8001，
> dense 0-2 + shared 3-44，未来 + attention）时的**运维排障**。撞到下面任一
> 症状先查这里。baseline 8000（cards 0-7，含 MTP `--speculative_config`）与
> patched 8001（cards 8-15）**共享 HOST pid namespace** —— 千万别 blanket-kill。

## 拓扑速查

| 服务 | 容器 | 端口 | 卡 | vllm serve 特征 |
|------|------|------|-----|----------------|
| baseline 8000 | `stepcast-vllm-isolated` | 8000 | 0-7 | 有 `--speculative_config {step3p5_mtp}` |
| patched 8001 | `stepcast-vllm-w8a8` | 8001 | 8-15 | **无** speculative；`PYPTO_DENSE_MLP_BACKEND=1` |

- 8 个 pypto worker（host venv，每 rank 一个，cards 8-15）：`python -m tools.step3p5.pypto_mlp_worker --device 8+r --rank r ...`，socket `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/pypto_mlp_rank{r}.sock`。
- 重启脚本：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/restart_workers.sh`（8 worker，setsid，4s stagger）。
- vLLM 启动脚本：`/logs/start_8001.sh`（容器内 `/logs` = host 上面那个目录）。

---

## ⚠ 症状 1：恢复 8001 时 vLLM `rtBinaryGetFunction 107000` / HCCL error 15，EngineCore 起不来

```
RuntimeError: create_config:...HCCLUtils.cpp:130 HCCL function error:
hcclCommInitRootInfoConfig(...), error code is 15
        rtBinaryGetFunction failed, runtime result = 107000.
```
8 个 TP worker 全报，EngineCore failed to start。

**根因**：pypto worker 占着 cards 8-15 的时候，vLLM 做 TP=8 HCCL init 会失败。
不是硬件（`npu-smi info -t health` 报 OK），`aclrtResetDeviceForce` 也清不掉。

**修复 —— 恢复顺序铁律（2026-06-30 花了 ~1h 才定位）**：

> **先起 8001 让它把 HCCL init 做完（等到 `Application startup complete` / `:8001/health=200`），再起 pypto worker。**

vLLM 在 worker 未就绪时 dense/shared 走 fail-safe fallback（原生 kernel），等 worker
起来后续请求自动切到 pypto。顺序反了（worker 先占卡）必挂。

```bash
# 1. 确认 cards 8-15 干净：无 pypto worker、无残留 8001 vLLM
pgrep -fc '[p]ypto_mlp_worker'                       # 期望 0
pgrep -af 'port 8001' | grep -v grep                 # 期望空
# 2. 先起 8001
sudo docker exec -d stepcast-vllm-w8a8 bash -c "bash /logs/start_8001.sh > /logs/boot_8001.log 2>&1"
#    盯日志直到 Application startup complete（约 1-3 min）
tail -F /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/boot_8001.log | grep -E "Application startup complete|HCCL function error|EngineCore failed"
curl -s -m5 -o /dev/null -w '%{http_code}\n' http://localhost:8001/health   # 等到 200
# 3. 再起 8 个 worker
setsid bash /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/restart_workers.sh </dev/null >/tmp/rw.log 2>&1
```

---

## ⚠ 症状 2：`pkill -f pypto_mlp_worker` 之后 worker 没起来 / 日志没生成 / ssh 命令"无输出"

**根因**：`pkill -9 -f pypto_mlp_worker` 的**命令行本身含 `pypto_mlp_worker` 字串，
`-f` 自匹配把当前 ssh shell 也 `-9` 掉了**，后续 launch 命令根本没执行。

**修复**：括号 trick（regex `[p]` 不匹配它自己 cmdline 里的字面 `[p]`）：
```bash
pkill -9 -f '[p]ypto_mlp_worker'
pgrep -fc '[p]ypto_mlp_worker'      # 计数同理用括号
```
另一个相关坑：**ssh 远程命令结尾带 `&` 后台化 worker 往往 no-op**（ssh 通道处理）。
用 `setsid bash restart_workers.sh </dev/null >LOG 2>&1`（**不带**结尾 `&`，setsid 前台
跑 ~32s，子 worker 自己 detach，ssh 正常返回 LOG）。

---

## ⚠ 症状 3：停 8001 后 / 跑完设备 IPC e2e 后，卡 8-15 进入脏状态

**根因**：(a) `kill -9` 整个 TP=8 vLLM 树时 HCCL 没机会优雅拆掉跨卡 exbus 映射；
(b) IPC exporter 调了 `aclrtIpcMemGetExportKey` 却**没调 `aclrtIpcMemClose`**，泄漏
exbus 句柄（Phase22 P5 记过）。两者都会让后续 vLLM HCCL init 报症状 1。

**预防**：任何 device-IPC exporter（含 e2e/KV-export patch）export 后必须
`aclrtIpcMemClose`（或复用长生命周期 key）。

**已知**：`aclrtResetDeviceForce` 清不掉这个状态；当前唯一可靠解法是
**按症状 1 的顺序在干净的卡上重起**（worker 全停 → 先起 vLLM → 再起 worker）。

---

## ⚠ 症状 4：attention 接线时 `aclrtIpcMemGetExportKey rc=507899`（导出 vLLM KV 失败）

```
[pypto-attn] layer 0 fallback: RuntimeError(
  'aclrtIpcMemGetExportKey rc=507899 dptr=0x... nbytes=1048576')
```
（507899 = `ACL_ERROR_RT_FEATURE_NOT_SUPPORT`）

**根因（2026-06-30 用 `_stage_ipc_diag.py` 在 card 8 复现坐实）**：
`aclrtIpcMemGetExportKey` 的成功充要条件是 **`dptr` 恰好等于某个 allocation block 的基址**。
只要是 `base + offset` 的 sub-pointer，**无论内存来自 torch caching allocator 还是裸
`aclrtMalloc`**，都返回 507899。这不是「torch 内存不能导出」——torch 内存只要落在块基址
照样导出成功。

复现证据（`python3 /logs/_stage_ipc_diag.py 8`）：

| 来源 | 指针 | rc |
|------|------|-----|
| torch tensor（独立块，落基址） | block base | 0 ✅ |
| 同一 caching block 第二个 tensor | base+131584 | 507899 ❌ |
| 裸 `aclrtMalloc` 基址 | base | 0 ✅ |
| 裸 `aclrtMalloc` 基址 + 4096 | base+4096 | 507899 ❌ |

**为什么 live KV 撞上**：`--num-gpu-blocks-override 32` 下每个 KV tensor 仅 1 MiB
（`32*128*1*128*2`），vllm-ascend 的 caching allocator 把多个小 tensor 塞进同一个 block，
KV 落在 `base+offset` → 507899。

**修复 —— 专属 MemPool（保留 torch 管理，不用裸 malloc）**：
给每个 K、每个 V tensor 各分配一个独立的 `torch.npu.MemPool`，使其独占一个 block →
`data_ptr == 块基址` → 可导出。tensor 仍是普通 torch NPU tensor（`is_npu / contiguous /
reshape` 全正常），vLLM 的 paged-attention 显存管理照常工作。**裸 `aclrtMalloc` 替换
`torch.zeros` 的旧方案会让 vLLM 失去对 KV 的管理，已弃用。**

复现验证（section D）：

| 方案 | K export | V export |
|------|----------|----------|
| 共享一个 MemPool | 0 ✅ | 507899 ❌（V 仍 sub-pointer） |
| K/V **各自独立** MemPool | 0 ✅ | 0 ✅ |

落地点：monkey-patch `vllm_ascend.worker.model_runner_v1._allocate_kv_cache_tensors`，
把 `PYPTO_ATTN_LAYERS` 范围内 layer 的 `torch.zeros` 包进
`with torch.npu.use_mem_pool(per_tensor_pool):`（K、V 各一个 pool）。诊断脚本
`pypto/_stage_ipc_diag.py`（容器内 `/logs/_stage_ipc_diag.py`）可随时复跑。

> 仍遵守症状 3 的铁律：export 后必 `aclrtIpcMemClose`（import 侧 teardown），别在
> export 侧裸指针上调 `aclrtIpcMemClose`（会 segfault）。

---

## §head-gate 实现状态（2026-07-01，layer-0 精度的最后一环）

**问题**：decode `attention_full.py` Scope 2.5 bypass 了 step3p5 的 head gate
（`gate = sigmoid(current_hidden @ w_g)` 逐 head 在 o_proj 前乘）。`config.
use_head_wise_attn_gate=True`。线上 layer-0 pypto attn 因此**首 token 对、之后发散**。
离线 e2e 加 gated golden 后 `bad_ratio` 0.0000→**0.7925**（坐实 gate 是主因；旧
golden 也 bypass 所以从没测到）。

**两条实现路径，均未落地（都撞硬约束）**：

1. **host-param R**（原作者推荐：R=块对角常量做 host 参数，kernel 内
   `gate_exp = sigmoid(gate_logits) @ R`，标准 matmul，不碰 codegen 坑）。
   **障碍**：`attention_full` / `attention_swa` 是**共享 inline**，被 decode_layer
   (581/2117)、mtp、single_layer_decode_{full,swa}_draft、collectives 等**大量
   消费者**调用；加一个必需参数会 break 所有调用点（`expects N got N-1`）。
   `decode_layer_full_dense` + `swa_dense` 在 import 时都 build（decode_layer.py
   L2419/2420），所以两个 kernel 都要改，且各自的消费者树递归级联。**scope 巨大**。
   （注：`prefill_fwd` 用的是 `attention_full_prefill`，**不受影响**。）
   补丁（仅 attention_full 那部分）存于 `_gate_impl_attention_full.patch`（dev+0162）。

2. **in-kernel R**（kernel 内自建块对角 R，**不改签名 → 零级联**，只动
   attention_full 一处）。连撞 **5 道 pto-isa codegen 墙**：
   1. 逐 head `pl.slice([BATCH_TILE,1])` + row_expand_mul → `TLOAD ND2ND` layout assert；
   2. 改 R-matmul + `pl.full` 建 R → `pl.full` 不能在 Orchestration body，须在 InCore（spmd）；
   3. 包进 `pl.spmd(1)` 建整块 `[16,1024]` R → `Vec buffer 1MB > 188KB UB`（fuse_create_assemble pass）；
   4. 改按 N-chunk（`[16,256]`）在 matmul spmd 内建 R → `pl.unroll() requires
      compile-time constant bounds`（body 内 `_HEADS_PER_CHUNK` 被当 symbolic）；
   5. （下一道预计：spmd 内 runtime 偏移的单行 `assemble` 是否合法——未验证）。

**结论**：in-kernel 路每修一层撞下一层 pl 约束，需要 pl/pto-isa 深度经验（不是盲试
能收敛的）；host-param 路数值上最稳但跨消费者级联巨大。**建议**：优先请 pl 专家把
in-kernel R 的「按 N-chunk build-time 静态摆放 + InCore pl.full」这对矛盾解开（可能
需要 `pl.full` 的常量-tensor 支持或一个 broadcast/repeat 原语），一旦通即零级联落地
layer-0 + 所有 full 层；否则按 host-param 做有计划的多文件级联重构。

**验证方式（gate 落地后）**：
- 离线：`python _stage_attn_e2e.py exporter 8 K & python _stage_attn_e2e.py worker 8 K`
  （card 8 co-tenant，golden 已 gated）→ 期望 `bad_ratio=0.0000`。
- 线上：同进程 A/B（8001 workers down=vanilla fallback vs workers up=pypto gated）。

### §head-gate 后续（2026-07-01 update）：gate 已 landed + 离线 GREEN，但 live 仍崩

- **gate 落地了**（host-param R 路径）：`attention_full` gate_r 参数 + R-matmul Scope 2.5，
  decode_layer 串接（18 处 gate_r），weight_loader 构造 gate_r。**独立核实**：worker
  staging `compiled Program ... 'gate_r__ssa_v0'` + `verify ok=True bad_ratio=0.0000`
  （`/tmp/_stage_attn_worker_*.log`，跑在 TpAttentionFull=线上程序）。
  （注：`attention_swa` 尚未串 gate_r —— SWA 层的 gate 是后续；layer-0 是 full。）
- **但 (C) live 同进程 A/B 仍崩**：8001 gated worker up → 输出乱码
  （`'First\n-\n\n sentence "\n with'`），worker `SUCCESS` 无 fallback；vs workers down
  的连贯 vanilla（`'First, the user asked...'`）。首 token 对、之后发散。
- **根因（同一类漏洞、更深一层）**：离线 harness 用 `seq_lens=1`（单 position、零 KV），
  **从没测过真实多 token prefill**（ctx_len>1）。所以 `bad_ratio=0` 只证明单 position 的
  kernel 数学对；**多 position 的 KV 读取路径**（block_table/slot_mapping 多块索引 +
  worker **self-KV**（忽略 IPC key、自建 KV）vs vLLM 真实 paged KV `[num_blocks,
  block_size, num_kv_heads, head_dim]` 的布局/索引语义）从没被离线覆盖。即原蓝图
  **open item #1（KV-rows ABI / paged-KV，最大卡点）**。
- **下一步（路径 1，用户选定）**：worker 侧 dual-key `import_ipc` 读 vLLM 真实 paged KV
  （MemPool 导出已验证 rc=0；缺 worker import + 按 vLLM paged 布局索引），**并给离线
  harness 加多 position 用例（seq_lens>1）**，否则又"离线绿/线上崩"。

### §head-gate 再后续（2026-07-01 晚 update）：KV-IPC 打通，多 position bug 离线复现

- **路径 1（真 KV-IPC）已实现 + 跨进程打通**：`_stage_attn_worker.py:attn_setup` 现在检测
  dual-key blob（`len==2*KEY_BUF` 且非零）→ `rt.import_ipc(key_k)` + `rt.import_ipc(key_v)`
  → k_dt/v_dt 指向 vLLM 真实 paged KV 各自 base（K/V 分离分配）。line 证据：worker
  `[attn-stage server] KV-IPC dual-key import layer=0 k_base=0x... v_base=0x...`，backend
  `layer 0 SUCCESS` 无 fallback。**整条链首次跨进程通**（8001 backend MemPool 导出 →
  worker import）。（vLLM K/V 布局 `(num_blocks, block_size, 1, head_dim)` flatten =
  `(kv_rows, head_dim)`，与 kernel block_table 行索引 `b*block_size+off` 对齐。）
- **但 live 仍乱码**（`'H the'`），self-KV 和 真-KV-IPC **都乱码** → **bug 不在 KV 来源**，
  在 **kernel 多 position attention 逻辑 / 每步 metadata**（ctx_len>1）。
- **离线复现成功**：`_stage_attn_e2e.py` 把 `seq_lens` 从全 1 改成 prefill 模式
  `arange(BATCH)+1`（row i attends 0..i，slot i，block 0）→ `bad_ratio=0.9006`。
  **row 0（seq_lens=1）仍匹配**，seq_lens>1 的行 90% 偏差 → 精确坐实多 position 路径。
- **下一步（离线，无需窗口）**：调 kernel flash-attention 多 position 逻辑——嫌疑：
  ①同一 dispatch 内"先写全部 slot 再读 block_table"的相位顺序；②ctx_len>1 的 online-
  softmax 累加 / valid_len 掩码；③block_table 多块索引。建议二分：先单行 seq_lens=2 是否
  已错。修好离线 `bad_ratio→0` 后再开窗口做最终 live A/B。
- **已改文件（dev+0162 已同步）**：`attention_full.py`（gate_r host-param R）、
  `decode_layer.py`（gate_r 串接）、`weight_loader.py`（构造 gate_r）、
  `_stage_attn_worker.py`（dual-key import + gated golden）、`_stage_attn_e2e.py`
  （gated golden + 多 position seq_lens）、`test_decode_layer_full_dense_st.py`（gated golden）。
  `attention_swa.py` **未串 gate_r**（SWA 层 gate 是后续）。

---

## 安全停 8001（开维护窗口腾卡时）

只杀 8001 树，**保 baseline 8000（pid 见 `pgrep -af 'port 8000'`，cards 0-7）不动**：

```bash
SERVE=$(pgrep -f 'port 8001' | head -1)             # 8001 的 vllm serve 根 pid
sudo kill -9 $(pstree -p "$SERVE" | grep -oP '[(]\K[0-9]+')   # 杀整棵 8001 树
kill -9 $(pgrep -f '[p]ypto_mlp_worker')                       # 杀 8 个 worker（infra-owned）
curl -s -m5 -o /dev/null -w '8000=%{http_code}\n' http://localhost:8000/health   # 确认 baseline 仍 200
```
失败的 vLLM boot 会留一堆 `[VLLM::Worker_TP] <defunct>` 僵尸（PPID 是容器 shim）——
无害（已死），不持卡，不用管。

---

## ✅ §head-gate 终局（2026-07-03）：RESOLVED（本地解决，待反馈上游）

**layer-0 full attention 已端到端对齐 vanilla 并 live 验证通过**：in-process A/B
`bad_ratio` 从 **0.97 → 0.0000–0.0002**（max|d| 0.007–0.375，bf16 噪声级）；8001 以
`PYPTO_ATTN_AB=0`（返回 pypto）跑 layer-0 pypto gated-attn，`layer 0 SUCCESS`（非
fallback），中英文 prompt 生成连贯、开头与 vanilla 一致；全程 8000/8001=200。

**根因（唯一真凶 = head-gate 的 gate_logits matmul）**：kernel 内
`gate_logits = normed_all @ w_g`（输出 **N=16**）用 `pl.matmul_acc` 时**丢掉了 K 维
累加**（只约第一块生效），logits ~20× 偏小、逐 head 比例不一 → `sigmoid≈0.35` 不压制
→ "热"卡（拥有该层大幅权重的 TP rank）o_proj partial 爆 ~40× → 整层乱码。输入
（`normed_all`、`w_g`）均已逐位验证正确；q_proj（同累加循环、大 N）正确，唯独小
N=16 出错。排除路径见下表；完整分析 + 复现器 + 上游诉求见
[`docs/upstream-issues/step3p5-head-gate-matmul-acc-n16-codegen.md`](../../docs/upstream-issues/step3p5-head-gate-matmul-acc-n16-codegen.md)。

| 探针 | 结果 | 判定 |
|---|---|---|
| 强制 `gate_sig=0` | 各卡输出塌到 hidden | gate **施加**路径 OK |
| 强制 sigmoid 输入 `=-10`（绕 matmul） | 各卡塌 | **sigmoid** OK |
| dump kernel `gate_logits`（覆写 `resid1[:, :16]` 经 worker 读回） | ~20× 偏小、比例不一 | **matmul** 错 |
| 把 gate matmul 移到 q/k/v 前（fresh normed_all） | 无变化 | 非 staleness |
| `matmul_acc` → `matmul`+`pl.add` | **编不过**（`pto.tmatmul` dst 必须在 acc 空间） | matmul_acc 是唯一累加路径 |

**本地绕过（已落地，标记待上游修）**：gate 改由 **worker 端 python 预算**，绕开坏
matmul。
- kernel `attention_full.py`：删掉 on-device gate_logits matmul + sigmoid + gate_r
  块对角展开；复用 `gate_r` 参数槽承载预算好的 `gate_exp`（`BATCH==NH_PAD==16`，
  `[16,1024]` 形状不变），o_proj 直接 `attn_out * gate_r` 逐元素乘。`w_g` 保留在签名
  但 kernel 不再用。
- worker `_stage_attn_worker.py::_AttnService.attn()`：每次调用算
  `gate_exp = repeat_interleave(sigmoid(RMSNorm(current_hidden, input_rms, eps=1e-5)
  @ w_g_local[:, :NUM_HEADS_FULL_LOCAL]), HEAD_DIM)`，拷进 gate_r buffer。等价 vLLM
  `sigmoid(g_proj(input_layernorm(hidden)))` 逐 head。

**上游修好前**：gate 保持 worker 预算（勿把 gate_logits 放回 on-device 小 N matmul）。

**同链路其它已修项**：worker gate-slice（`g_lo = rank*NUM_HEADS_FULL_LOCAL`，非
`rank*PAD_FULL`）；backend `_AttnWorkerClient` 断线重连（BrokenPipe → 重连，避免
worker 重启后永久 fallback）；轻量 gate fuse（gate_exp 预算 + o_proj 内逐元素乘，
避开 64KB L0 overflow 与 DCE/别名）。

**回退到保守态**：`PYPTO_ATTN_AB=1`（返回 vanilla）重启，或停 workers（自动 fallback）。

**遗留**：(1) 上游报 pypto `matmul_acc` 小 N=16 丢累加 bug；(2) 扩展到 SWA 层 gate +
其余 44 层（当前仅接 layer-0 full attn）。
