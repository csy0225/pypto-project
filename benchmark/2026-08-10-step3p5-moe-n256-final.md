# Step3p5 MoE BS1 N256 收口与 `stepfun/develop` 发布（2026-08-10）

## 1. 发布结论

本轮 MoE BS1 优化已收口并发布到：

```text
csy0225/pypto-lib:stepfun/develop
  a31977fbb7ced6d2e599539c223d07813f161140
```

该提交是双父 merge：

```text
491267c45875e9b1e0071eed224e2e73526799e2  # 远端最新 release harness
7d3e02ae4ed447ded543fb716a479350f1f89db6  # 已验证 N256 candidate
```

合并后产品实现与已验证 candidate 逐字一致：

```text
models/step3p5/decode_fwd.py
  sha256:d392311ce1f38a67ddaa007173bb012c87e68cafeb5dca6b47813a2424683eea
```

远端 release harness 只补充文档、CI、route/precision harness 与测试；未改变
`decode_fwd.py` 或 attention 产品实现。因此既有设备性能、精度和 compile-only
证据可继续绑定到该产品源码。

## 2. 落地优化

最终落地包含两层连续改造：

1. 普通 routed expert 的 hidden quant N chunk 从 `64` 扩到 `256`，保留
   L43/L44 specialization；
2. gate/up cube tile 从 `K512xN64` 改为 `K256xN256`，配置
   `pl.split(pl.SplitMode.NONE, slot_num=4)`，每 expert 的 N work 从 `20`
   降到 `5`；
3. empty-rank scatter 不再用 scheduler predicate 跳过整个 grid，而是在 kernel
   内判空，保留 early staging 和固定 credit 协议。

`down24` 不在发布范围内：虽然 e3 down kernel 快约 `5.5--6.0 us`
（`14.7--15.8%`），但 e1/e2、scatter 下游相位和 L4 terminal 回退，最终裁决为
`NO_GO_NO_RERUN`。

## 3. BS1 整网 ITL

工作点：

```text
host                0162
active batch        1
context/max-seq     65536
blocks              512
warmup/measured     10/100
scope               45-layer pre-final-norm hidden-only holder.run
order               parent -> candidate -> parent
```

| 指标 | parent center | candidate | 收益 |
|------|--------------:|----------:|-----:|
| mean | 36.354 ms | 35.055 ms | **1.299 ms / 3.57%** |
| p50 | 35.778 ms | 34.271 ms | **1.507 ms / 4.21%** |

裁决为 `GO_GAIN_CONFIRMED`。三臂 hidden tensor payload byte-exact，tail token
均为 `14371`。

本 harness 的 100 样本 p99 使用排序下标 99，即第 100 个样本，所以 p99 等于
max，仅作诊断，不能作为本轮 release gate：

```text
A1 parent  51.087 ms
B candidate 52.005 ms
A2 parent  41.706 ms
```

## 4. DFX / PMU

最终 BS1 DFX/PMU 报告为 PASS：

```text
swimlane raw
  /mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-n256-final-bs1-dfx-pmu-20260809-214013-1024390-067849361/
  swimlane/dfx_raw

PMU raw
  /mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-n256-final-bs1-dfx-pmu-20260809-214013-1024390-067849361/
  pmu/event2/pmu_raw
```

8 rank 的 `deps.json`、`l2_swimlane_records.json`、`name_map.json` 均齐全。
已知 predicated-skip postprocess 限制不影响 raw capture。

PMU event2 是 busy-cycle 事件，不是 transferred-byte counter。不得用它反推
物理 HBM GB/s，也不得据此直接与 910B 的 `1.6 TB/s` 峰值比较带宽效率。

## 5. 精度与最终验证

正式 v4 臂保留为 formal FAIL：128 步中仅 step77 出现一次 TP spread
`1.65625`。按冻结规则使用相同 128 oracle IDs 做最小 targeted replay 后：

```text
aligned             123/128 >= 122
TP spread           128/128 == 0
step77              token exact, spread == 0
finite/shape/rows   PASS
```

Claude 交叉复审结论为：N256 precision/landing `GO`，v4 归档、v5 为生效证据，
不再扩展验证。

0162 上已验证 candidate：

```text
pytest              30 passed
ruff                PASS
compile-only        PASS
PYPTO_CODEGEN_MAX_WORKERS=1
context/max-seq     65536
blocks              512
```

合并远端 release harness 后，又把完整 merge tree 复制到 0162 做静态复核：

```text
/mnt/persist/chensiyu/workspace/moe-opt/tmp/
pypto-lib-stepfun-develop-a31977fb-static-validation

pytest              30 passed
ruff                PASS
```

本次 merge 后没有重复设备回归；其产品源码 SHA 与已完成 compile/precision/ITL
验证的 candidate 完全一致。

## 6. 权威证据

```text
whole-network ITL
  /mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-n256-bs1-whole-itl-aba-20260809-212340-1012474-705619221

DFX / PMU
  /mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-n256-final-bs1-dfx-pmu-20260809-214013-1024390-067849361

precision targeted replay
  /mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-n256-live-precision-ctx64k-targeted-replay-20260809-v5

0162 final candidate validation
  /mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-n256-landing-final-validation-20260809-2254-v1

down24 read-only audit
  /mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-n256-down24-readonly-final-audit-20260809-v1

final landing seal
  /mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-n256-final-landing-seal-20260809-v1
```

最终 seal manifest：

```text
sha256:4891fbd1cdece0053ac3761467ea392e570e9137cad2fd844d3fa3e2525e8ee6
```
