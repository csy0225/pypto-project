# Workload-sized Attention canonical image：ITL / DFX（2026-08-06）

## 结论

基于当时 GitHub `stepfun/develop` 最新提交构建并发布了无源码挂载的 immutable
canonical 镜像：

```text
tag:
hub.i.basemind.com/stepcast/vllm-pypto:
stepfun-develop-20260806-attn-taskmajor-canonical

manifest:
sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479

config:
sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216
```

镜像 pin：

```text
pypto      8e92b46808f9f7c09b6431ad4691503f09c12ee5
pypto-lib  c9af5790d5fe450e14fd43c88099b87539089d17
simpler    e2efebcbd190302609c0775d2984f409f5f42c76
pto-isa    ecb6c303f797749f811a494742c3c08156aacabb
PTOAS      fc8c6caee561914b4fb991dfc8427bb63194269e
ptoas-bin  v0.50
profile    a2a3
```

credential、五仓 pin、clean worktree、CANN 8.5.1 absence、prepared-swimlane
`RunConfig` 接口及 profile 审计均通过。0162 只按 manifest digest 启动；运行时只挂载
driver、checkpoint/reference 和输出目录，没有源码或 runtime overlay。

构建使用 `BUILD_JOBS=2`，没有固定 `MAX_JOBS=1`。devbox 留存日志：

```text
/data/chensiyu/hw_project/pypto/workspace/attn-opt/build_logs/
stepfun-develop-20260806-attn-taskmajor-canonical.build.log
stepfun-develop-20260806-attn-taskmajor-canonical.audit.log
stepfun-develop-20260806-attn-taskmajor-canonical.push.log
```

## 整网 BS1、每请求 64K ITL

口径：

```text
active_batch = 1
context_len = 65536
num_blocks = 512
warmup = 5
measured iterations = 50
devices = 8,9,10,11,12,13,14,15
```

结果：

| metric | ITL |
|---|---:|
| min | 39.057 ms |
| mean | 39.594 ms |
| p50 | 39.612 ms |
| p99 / max | 40.680 ms |

RC=0；保存的 hidden shape 为 `[8,16,4096]`，全 finite，active row 和全 capacity
的 TP max spread 均为 `0.0`。相对 Wave5 的 `49.796 ms`，p50 下降
`20.45%`；该对比跨越了最新 MoE 基线等整栈改动，**不能把全部收益归因于
attention delta**。

artifact：

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/
image_attn_taskmajor_canonical_20260806_c9af5790_3eb694e0/
bs1_ctx65536_whole_net_itl/
```

## 前两层 Attention DFX

同一 digest，BS1、每请求64K，warmup=5、50 次 timing 后单独执行 dep-gen 与
prepared l2-swimlane capture。结果：

| metric | value |
|---|---:|
| two-layer min | 3.5238 ms |
| two-layer mean | 3.9131 ms |
| two-layer p50 | 3.6323 ms |
| two-layer p99 | 10.1127 ms |
| two-layer max | 13.8748 ms |
| output finite / TP spread | true / 0.0 |
| reference replacement | exact，max diff 0 |

Full 64K 的 workload-derived logical task 数为 QK/softmax/SV-online
`24/32/24`；不是固定 24 个物理核。生成代码中没有 Pass-A/B/C，也没有
Full/SWA standalone out-proj cast。8 个 rank 的 swimlane 均存在。

诊断 LOW-WAIT 为 `rank2/d0`：

```text
makespan                       690.1 us
tp_all_reduce stage span-sum   176.24 us
```

其它 rank 的数百毫秒 span 是 DFX capture 时 collective peer-arrival spin wait；
不能当作两层 wall-clock。权威 wall-clock 是上面的 50 次 timing。

0162 LOW-WAIT swimlane：

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/
image_attn_taskmajor_canonical_20260806_c9af5790_3eb694e0/
bs1_ctx65536_two_layer_dfx/build_output/
TwoLayerAttnPerf_20260806_090839/dfx_outputs/
rank2/d0/l2_swimlane_records.json
```

完整 DFX artifact：

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/
image_attn_taskmajor_canonical_20260806_c9af5790_3eb694e0/
bs1_ctx65536_two_layer_dfx/
```

## 边界

该镜像完成了本轮要求的 latest-source immutable audit、BS1×64K 整网 ITL 和两层
DFX gate。它尚未重跑 Wave5 的 Main N=128×3、Main batch16、MTP batch1/16
等完整 production release matrix，因此不能自动继承 Wave5 的全量
release-qualified 标签。整网 BS16×每请求64K 仍受既有 HBM static-arena OOM
约束。
