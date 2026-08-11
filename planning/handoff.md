# 接力上下文（Handoff）

> **只描述下一位 agent 现在要接的工作。最后更新：2026-08-08。**
> 当前状态以 [`../STATUS.md`](../STATUS.md) 为准；历史过程不要复制回本文件。

## 1. 当前源码与镜像

GitHub `stepfun/develop`（以远端 ref 为准）：

```text
pypto-lib  491267c45875e9b1e0071eed224e2e73526799e2
pypto      8e92b46808f9f7c09b6431ad4691503f09c12ee5
```

pending build spec:
`deployment/docker/builds/stepfun-develop-20260808-moe-opt-latest-source.env`

最近一次已构建的 latest-source partial image（历史 `c9af5790`）：

```text
tag:
hub.i.basemind.com/stepcast/vllm-pypto:
stepfun-develop-20260806-attn-taskmajor-canonical

manifest:
sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479

config:
sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216
```

该历史 image 的五仓 pin、credential、clean tree、CANN、prepared-swimlane 接口
和 A2A3 QK/softmax/online blocks-per-task=`22/16/22` profile 审计均 PASS。
0162 验证按 digest-only 启动，无源码/runtime overlay；它不包含当前 `491267c4`。

## 2. 已完成验证

BS1、每请求64K、整网 warmup=5 + 50 次：

```text
min/mean/p50/p99/max =
39.057/39.594/39.612/40.680/40.680 ms
hidden finite
TP spread = 0
RC = 0
```

同 digest 的前两层 Attention DFX：

```text
two-layer p50 = 3.6323 ms
reference exact = true
TP spread = 0
swimlane ranks = 8/8
LOW-WAIT = rank2/d0
```

LOW-WAIT swimlane：

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/
image_attn_taskmajor_canonical_20260806_c9af5790_3eb694e0/
bs1_ctx65536_two_layer_dfx/build_output/
TwoLayerAttnPerf_20260806_090839/dfx_outputs/
rank2/d0/l2_swimlane_records.json
```

完整记录：
[`../benchmark/2026-08-06-attention-taskmajor-canonical.md`](../benchmark/2026-08-06-attention-taskmajor-canonical.md)。

## 3. 下一步

1. 不要重建或恢复已被 supersede 的 2026-08-05 R2。
2. 先用 pending spec 构建并审计 `491267c4` 镜像，再继续执行 Wave5 同口径的
   Main N=128×3、Main batch16、MTP batch1/16 和 smoke/precision matrix。
3. 整网 BS16×每请求64K 需要先解决约 16 GiB static-arena 的 HBM 容量门禁；
   不能把两层数据写成整网 ITL。
4. live serving 仍需 paged-KV bridge、重复权重/HBM closure 和同代 Main→MTP gate。

## 4. 不得使用的旧信息

- 2026-08-05 R1 已撤销，R2 未发布且其 pypto-lib pin 已先被 `c9af5790`、后被
  当前 `491267c4` 取代。
- 本地 branch 名、dirty tree、source mount 和旧 Wave5 artifact 不能替代当前 digest。
- DFX 其它 rank 的数百毫秒 collective span 含 peer-arrival spin wait，不是
  Attention wall-clock。
- 整网 `39.612 ms` 相对 Wave5 的提升跨越最新整栈代码，不能全部归因于 Attention。
