# N=1 Whole-Net Canonical Test（唯一准出标准）

> **HARD RULE**：N=1 step3p5 whole-net 的"精度正确 / 无 stall / 可发布"结论，
> 只能由本文定义的真实权重、真实 token 测试给出。**精度准出 = 多步 decode 逐 token
> 对比 vanilla（§2，≥95%）**；单 token `argmax==303` 仅为首 token 冒烟/liveness。
> 禁止用随机输入、`RUN_CLEAN`、P1/P20、中间态、compile-only、只验第一个 token 或自造 harness 替代。

> **Stable environment SSOT（0162）**：
> [`develop/N1/N1-STABLE-ENV-0162-20260717.md`](../develop/N1/N1-STABLE-ENV-0162-20260717.md)。
> 本文定义测试准出标准；stable 文档冻结完整三仓、runtime binary、工具链、
> checkpoint、设备和环境变量。

## 1. 被测对象与固定组合

```text
program = whole_decode_faithful_real_single_chip
layer module = models.step3p5.decode_layer_single_chip
branch = feat/whole-net-n1-fusion
release commit = 3af13f4facbe8db5cd4a6c769e8b9e07e351c7b9
compiler/runtime commit = pypto e49ce111c1503f4fb3e898af4223560cab907a62
machine = gpu-a910x-0162
devices = 8,9,10,11,12,13,14,15
P_FAITHFUL_MOE_LAYERS = 42
dispatch = fixed-slot pull
combine = pull
weights = native W8A8 IPC
KV = IPC
token = 6127
golden argmax = 303
```

当前 whole-net 是 **dispatch pull + combine pull**，且采用 **rank-local
single-submit**：`host_orch.py` 中每个 rank 只调用一次
`_submit_chip(... whole_chip_orch ...)`。TP=8 时仍然是 8 个 rank-local
submission；这不是 8 卡全局只提交一次，也不是合并 CHIP 内部 InCore/Group/SPMD
kernel。PUSH 探针、历史 push 版本、以及旧 46 layer-level submission 版本都不是
当前准出对象。

## 2. 精度标准（2026-07-24 更新为多步 decode）

> **标准更新**：精度准出从"单 token `argmax==303`"升级为**多步 decode 逐 token**对比 vanilla
> vLLM。多步已覆盖第一个 token，**单步/单 token 测试不再单列**；单 token `argmax==303` 降级为
> **首 token 快速冒烟 / liveness** 指标，不再是唯一精度 PASS。

- **精度准出（唯一）**：**多步 decode 逐 token** teacher-forced 对比 live vanilla vLLM W8A8
  oracle，seed=6127 / N=128 → **ALIGNED ≥ 95%**（当前 baseline 124/128=96.9%，miss 均为
  vanilla 自身 near-tie）。驱动：`pypto-lib/tests/step3p5/ci/{LIVE_PRECISION_AB.md,
  run_live_precision_ab.sh}`（`stepfun/develop`）。
- **首 token 冒烟 / liveness（快速回归，非精度准出）**：`--hidden-token 6127`、`ctx=1`、
  rope position-0 identity、`BATCH=16`（仅 row0 有效，row1..15 padding）、检查
  `logits_shard_out[r, 0]` 的 `argmax == 303`。用于快速确认 whole-net 能跑通 + 首 token 对；
  配 `RUN_CLEAN` + 隔离探针 `_probe_barrier_scale.py` 判 stall。
- 以下均不构成精度 PASS：
  - 只有 `RESULT=REAL_WEIGHT_IPC_RUN_CLEAN`；
  - P1/P20 或 `P_FAITHFUL_MOE_LAYERS=0`；
  - 随机 hidden；
  - BF16 dequant 权重回退；
  - 只编译、不执行；
  - 多步 ALIGNED < 95%（或只验第一个 token 就下"精度对"结论）。

## 3. 0162 标准命令

### 3.1 环境

```bash
source /usr/local/Ascend/cann/set_env.sh
source /data/chensiyu/hw_project/pypto/workspace/activate.sh
export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072
export P_FAITHFUL_MOE_LAYERS=42

cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
OUT=/tmp/n1_weight_ipc
```

注意：截至 2026-07-16，0162 上可用 checkpoint 是上述 `/data/chensiyu/...`
路径；旧 `/mnt/hw910test-jfs/...` 默认路径在本机不可用。

### 3.2 fresh exporter pool

```bash
rm -f "$OUT"/ready.rank* "$OUT"/STOP 2>/dev/null

for r in $(seq 0 7); do
  dev=$((8 + r))
  python -m tests.step3p5._stage_whole_faithful_real_ipc \
    --export-rank "$r" \
    --dev "$dev" \
    --kv-ipc \
    --out "$OUT" \
    --ckpt "$CKPT" \
    > "/data/chensiyu/hw_project/pypto/workspace/logs_n1/exp_${r}.log" 2>&1 &
done

while [ "$(find "$OUT" -maxdepth 1 -name 'ready.rank*' | wc -l)" -lt 8 ]; do
  sleep 15
done
```

### 3.3 canonical P42 worker

```bash
python -m tests.step3p5._stage_whole_faithful_real_ipc \
  --device 8,9,10,11,12,13,14,15 \
  --reuse-exporters \
  --kv-ipc \
  --hidden-token 6127 \
  --layer-module models.step3p5.decode_layer_single_chip \
  --layer-name whole_decode_faithful_real_single_chip \
  --out "$OUT" \
  --ckpt "$CKPT"
```

每次必须同时检查：

```text
process rc = 0
存在 [worker] RUN done
argmax=303
```

稳定性发布 gate 为同一份最终源码、fresh exporter pool 下连续 20 次全部满足。
历史候选版本的 2026-07-16 20-run 与最终 release commit 的 source SHA 不同；
该证据缺口已通过对 `0e7a0fdd` 重新执行完整 20-run 补齐。

### 3.3.1 完整复现对象不是只有 pypto-lib

canonical 命令实际从 source/editable 路径加载：

```text
pypto-lib 模型/生成器
pypto editable/source
simpler/runtime editable/source
runtime build artifacts / .so
CANN / PTOAS / pto-isa / Python 环境
checkpoint / devices / ring 环境变量
```

历史 exact-source 20-run 的**模型源码**已绑定到 `pypto-lib 0e7a0fdd`，但
运行时的实际状态是：

```text
pypto HEAD 5e619dc7 + 未提交的 stacked sub-view / import_ipc_all 源码
simpler HEAD 98ce22a6 + 未提交的 child-process ACL IPC import 源码
```

因此该 20-run 证明的是 0162 上模型 release 与当时实际 runtime source 的组合，
不能误写成“三仓 clean pin 上直接跑了 20 次”。

审计后已将这些实际运行依赖 formalize 为可复现的 clean pin：

```text
pypto-lib  0e7a0fddc90c4f2348f1d59e015fb817a0877a02
pypto      e277de9f2a55a686956d66933301204520bd7374
simpler    36957c6b56700ecba3aeb8dbbedd6240594e01de
pto-isa    ecb6c303f797749f811a494742c3c08156aacabb
PTOAS src  72ada0a1
ptoas-bin  self-report 0.45
ptoas-bin  sha256 fe7949daa62cdf787958101df35fe15dfa430bd18bf7e880353935a29b966076
```

当前稳定 binary 实际自报 `ptoas 0.45`；历史 `v0.49` 仅保留在历史记录中，
不能作为当前环境重建 pin。

其中 pypto 提供 `StackedDeviceTensor` 分层连续 sub-view 和
`DistributedWorker.import_ipc_all`；simpler 在 forked chip child 的 ACL
context 内执行 external IPC import。最终 clean pin 另有独立 canonical smoke，
见 §4；不要把它与历史 20-run 的 runtime commit 状态混为一谈。

所以在 0234 **只执行 `git pull pypto-lib` 不构成同一测试对象**。即使
`decode_layer.py`、`moe.py` 和 generator byte-match，也可能缺少 stacked
weight slicing、`import_ipc_all` child-context 路径或使用不同 runtime binary。
在三仓 clean snapshot 和 binary manifest 对齐前，不得把跨机器差异直接归因于
model commit 或 512B signal isolation。

### 3.4 清理

```bash
touch "$OUT/STOP"
```

等待所有 exporter 进程退出后，再执行：

```bash
rm -f "$OUT"/ready.rank* "$OUT"/STOP
```

测试结束后确认设备 8–15 无残留运行进程。不要在 exporter 仍存活时删除
`"$OUT"`。

## 4. 当前 release 证据

当前 single-submit release 证据目录：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/
  single_submit_cleanup_p42_repeat20_20260718_174948
```

结果：

```text
rc=0
REPEAT summary pass=20/20
每次 argmax=303
TOP5=[303, 9592, 1043, 768, 2086]
runtime min/mean/max = 0.5536 / 0.6607 / 2.2766 s
fingerprints_unique=1
```

首轮包含首次 graph/run 开销；第 2～20 轮稳定在约 0.55～0.59s。生成的
`host_orch.py` 只有一个实际 `_submit_chip` 调用点：

```text
build_output/WholeDecodeFaithfulRealSingleChip_20260718_105511/orchestration/host_orch.py
  line 200: _submit_chip(orch, callables["whole_chip_orch"], ...)
```

20 个 worker-run dmesg before/after 窗口新增 relevant=0；未新增
`507018`、`running-stalled`、`stranded CQE`、devmm/page fault、illegal VA
或 timeout。


cleanup 后的瘦身入口 repeat20：

```text
log = /data/chensiyu/hw_project/pypto/workspace/logs_n1/
      single_submit_cleanup_p42_repeat20_20260718_174948
rc=0
repeat pass=20/20
argmax=303 for every run
TOP5=[303, 9592, 1043, 768, 2086]
host_orch still has exactly one _submit_chip call site
dmesg relevant=0
```

最终源码与 binary 指纹：

```text
pypto-lib commit:
  3af13f4facbe8db5cd4a6c769e8b9e07e351c7b9
pypto commit:
  e49ce111c1503f4fb3e898af4223560cab907a62
simpler/runtime:
  36957c6b56700ecba3aeb8dbbedd6240594e01de

39e2ffdbf1b2ecb8ba7969f6b7d9376248ca72fcafa2735028ea843a3f238a45  models/step3p5/decode_layer.py
90894315cc1c6499b3e269a2afc376153009a7a35b7f208d5e686d9256ab6e0b  models/step3p5/decode_layer_single_chip.py
525e755f7aa9f66624f55dfaddbeb48e5738fda1d4948383e51e8640132ed21b  tools/step3p5/_gen_single_chip_real.py
b4c2a0eed6d341fa3aba993f614e28d17e31cd3a0061fd519b9b2731cd30aef8  tests/step3p5/harnesses/_stage_whole_faithful_real_ipc.py
0d3ce0e5c355f87d8e0b9f5593477b6893841c99367fb0c434ba0be5fac825b8  pypto_core.cpython-311-x86_64-linux-gnu.so
```

P1 diagnostic 仍保留一个边界事实：single-submit 与旧 baseline 在 padding /
physical rows 的 routed path 上不逐元素等价；row0 canonical argmax 不受影响。
该诊断不能替代 P42 release，也不能反向否定当前 canonical gate。若后续要做
padding-row 完全等价，应继续从 `local_routed_x_scale` / route metadata 开始排查。

release commit `0e7a0fdd` exact-source 20-run 目录：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/signal512/
  signal512_p42_20_20260717_001135
```

结果：

```text
FINAL pass=20/20 rc=0
每次 argmax=303
TOP5=[303, 9592, 768, 1043, 410]
runtime min/mean/max = 2.50 / 2.5605 / 2.62 s
```

20 次数值指纹一致：

```text
max|next_hidden|=264192.0000
row0|next_hidden|=588.0000
max|h_mid|=294.0000
max|logits|=14.0506
```

日志记录的源码 SHA 与 release commit 一致：

```text
decode_layer.py          9b6c83ca915ca9fcb5b02223e1a733c1c28fabca45dec6019b3b41a5f3fd7d5d
moe.py                   8a3670a047aff5b5af5d352446d8a35c866708f0eccba2b70904ad18896d5a2a
_gen_faithful_real.py    bf65295b2167bd96516e8ef2cebd97b69ebc7d46a86e13d304180ebf6a514010
```

先前整理后 smoke：

```text
log = .../signal512_final_smoke_20260716_230225
runtime = 2.57s
argmax = 303
FINAL_SMOKE = PASS
```

20-run 的 20 个逐 worker-run dmesg 窗口与 smoke 的 worker-run 窗口均未新增：

```text
devmm/page fault
illegal VA / illegal instruction
DMA/UB fault
507018
running-stalled
stranded CQE
```

补充边界：exact-source 20-run 在 20 个 worker 全部结束后关闭 fresh exporter
pool，outer before/after 窗口在 exporter teardown 阶段新增 2 条
`stranded cqe`（dev8/dev11 exporter PID）；20 个逐 run 窗口均为 0。
因此它们记录为 exporter cleanup 现象，不归因于 whole-net worker kernel，
也不得写成“整个进程生命周期 dmesg 绝对无任何相关行”。

### 4.1 最终三仓 clean pin smoke

三仓 clean pin formalize 完成后，在 0162 对最终 manifest 独立执行 canonical
P42 smoke：

```text
log:
  /data/chensiyu/hw_project/pypto/workspace/logs_n1/release_manifest/
  final_stack_smoke_20260717_015635

pypto-lib = 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
pypto     = e277de9f2a55a686956d66933301204520bd7374
simpler   = 36957c6b56700ecba3aeb8dbbedd6240594e01de

P42 / dispatch pull / combine pull
token 6127 / native W8A8 IPC / KV IPC
rc=0
RUN done 2.58s
argmax=303
TOP5=[303, 9592, 768, 1043, 410]
RESULT=REAL_WEIGHT_IPC_RUN_CLEAN
worker-window added relevant dmesg=0
```

outer 窗口在 worker 完成、exporter teardown 后新增 1 条 dev14 exporter
`stranded cqe`；它不在 worker 执行窗口内，按 cleanup 现象归档。

该 clean-pin smoke 绑定的 runtime binary SHA256：

```text
libhost_runtime.so
  7b29004b9d047d550ee6689120be83e650a3bcf39b196fd0ea112a3c6271891a
libaicpu_kernel.so
  62b8c2430abc9cafe257b758148c22fc1ab6da1085b0a103ae7bc465c57ca390
libsimpler_aicpu_dispatcher.so
  1b4b8467f0c899af64ebcd2f0a98e83b89160dca32177d0baecebddd3be4f973
_task_interface.cpython-311-x86_64-linux-gnu.so
  318510dfc2a55b27749609fd56850657b77691bc4078d6a7064f6451076f2c53
```

相关 focused unit tests：

```text
127 passed
```

覆盖 `DeviceTensor`/`StackedDeviceTensor` 分层 view、
`DistributedWorker.import_ipc_all` 和 simpler child IPC import。


### 4.2 stepfun/develop clean regression

N1 single-submit 已合入三仓 `stepfun/develop` 后，在独立 clean source tree 上完成完整回归：

```text
simpler   stepfun/develop = c7fdc574fb0637fda581398e5064994734d913c5
pypto     stepfun/develop = 9ec303f6f984b96a70d467cd46185b7b9157d3ec
pypto-lib stepfun/develop = e1513d22e2ac10ad97f7c9811122edf03bb78a8f

log = /data/chensiyu/hw_project/pypto/workspace/logs_n1/
      stepfun_final_clean_p42_repeat20_20260718_191643
rc=0
repeat pass=20/20
argmax=303 for every run
TOP5=[303, 9592, 1043, 768, 2086]
runtime min/mean/max = 0.5559 / 0.6684 / 2.3542 s
fingerprints_unique=1
dmesg relevant=0 for all 20 worker-run windows
```

## 5. 发布前静态与生成器 gate

```bash
python -m py_compile \
  models/step3p5/decode_layer.py \
  models/step3p5/moe.py \
  tools/step3p5/_gen_faithful_real.py

git diff --check
```

还必须执行真实 generator round-trip：

1. 从当前 `decode_layer.py` 剥离
   `_build_whole_decode_faithful_real_program` 和 binding；
2. 运行 `tools/step3p5/_gen_faithful_real.py`；
3. 与剥离前文件做字节比较。

当前 release 的要求与结果：

```text
PRECOMMIT_ROUNDTRIP=PASS
ROUNDTRIP_CMP_RC=0
```

这项检查用于防止 generator 把 `inverse_map` 从 dispatch 边界移回 combine，
或恢复 self `remote_load`、旧 count snapshot 等未验证结构。

## 6. 结论边界

可以声明：

> release commit `0e7a0fdd` 在 0162 上完成 native-W8A8、KV-IPC、
> dispatch-pull + combine-pull 的 canonical P42 20/20，全部
> `argmax=303`，无 stall、无精度漂移。

根因表述必须保持严谨：

- 32B → 512B control-signal physical isolation 是最终最小布局 A/B 变量；
- 在 0162 上，它与历史随机 stall 消失具有强关联，并由 exact-model-source
  20/20 与最终 clean-pin smoke 支持；
- 现有材料不是 matched 单变量因果证明，也不证明它是跨机器充分条件；
- 不得把某个 PUSH/TPUT、某个 signal bit 或某个历史 kernel 位置写成已经由硬件层
  单独证明的唯一根因。
