# 专项：Wave4 间歇性 TP spread —— source publication / window lifetime 边界

| 字段 | 值 |
|------|----|
| **子系统** | whole-net / `models/step3p5/decode_fwd.py::tp_all_reduce` |
| **error signature** | 固定 oracle 下间歇性 `hidden_tp_spread != 0`；token gate 可能仍 PASS |
| **首次出现** | 2026-08-03 Wave4 immutable |
| **状态** | ✅ 已解（0162 release-qualified） |
| **最终版本** | `pypto-lib@7099476b` / Wave5 manifest `sha256:4acc77cd…` |

## 1. 背景

PERF-C4 已把 all-gather 从 pull 改为 push，Wave3 又在 final local copy 后增加
Wave 3 completion barrier，关闭通信 window 的最终读取生命周期。Wave4 进一步让
two-layer harness 与 canonical 三波协议建立 AST equality contract。

但 Wave4 固定 oracle 两轮中仍有一轮在 step2 出现 TP spread=`2.0`。raw token
两轮都超过 95%，说明 token gate 不能替代跨 rank hidden 一致性 gate。

## 2. 现象

Wave4：

```text
Run1 122/128, hidden finite, step2 TP spread=2.0
Run2 123/128, hidden finite, TP spread=0
```

该现象是间歇性的，且 final-copy lifetime 已有第三波闭合，因此不能继续只从
all-gather 或最终 window reuse 解释。

## 3. 根因证据

协议审计将最早仍未由通信 primitive 明确发布的边界收敛到 Wave 1 前的 source
partial：

```text
ordinary local store to tmp_window
-> notify peers
-> peers remote_load owned shard
```

修复把普通 local store 换成 self-target synchronous TPUT，再发 Wave 1 notify。
独立 stress/A-B、Main N=128 预定义三轮、batch16 与 MTP 均转为 TP spread 全零。

因此本轮可下的强结论是：

> 0162 当前证据支持 source publication/lifetime ordering 是关键边界，
> self-target TPUT 是当前最小整网修复。

没有 bit-level hardware trace，也没有覆盖所有平台，所以不能写成“已唯一证明所有
硬件的根因”。

## 4. 如何解决

保持既有 reduce-scatter + push all-gather 与三波协议，只替换 source publication：

```python
pld.tensor.put(
    dst=tmp_window,
    peer=my_rank,
    src=local,
    chunk_rows=BATCH,
    chunk_cols=TP_ALL_REDUCE_CHUNK,
)
```

完整状态机：

```text
self-target TPUT
-> Wave 1 source publication
-> reduce-scatter
-> push all-gather
-> Wave 2 result publication
-> final local copy
-> Wave 3 lifetime close
```

同步修改 Main、MTP、two-layer harness，并增加 source/AST contracts；不改变数值
顺序、rank ownership、host ABI 或 task graph 结构。

## 5. 走过的弯路

1. Wave 3 只解决 final-read/reuse lifetime，不能自动证明 Wave 1 source payload 已
   在 notify 前可靠发布。
2. 单轮 spread=0 不能关闭 race；必须预定义有限轮次并保持输入、镜像和判据固定。
3. token 正确不代表 rank hidden 一致；必须同时检查 finite 与 `tp_spread_max=0`。
4. standalone probe 只能提供机制证据，不能替代 immutable whole-net Main/MTP gate。

## 6. 如何避免

- 每个跨 rank 协议都显式画出
  `producer -> payload publication -> notify -> wait -> consumer -> reuse`；
- data publication 与 control publication 必须有明确的同方向/同步 primitive；
- source、result 与 final-read lifetime 分别建立 gate，不能只靠一个 completion wave；
- 固定 peer order、单 FP32 accumulator、最终一次 BF16 cast 作为独立数值合同；
- release 必须包含预定义稳定性轮次、batch16、MTP、immutable audit 和 DFX；
- 结论注明 machine scope，单机验证不外推为跨架构定律。
