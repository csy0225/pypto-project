# vllm-pypto 可复现镜像 — 构建 / 部署 / 验证

基于 immutable repository pins 构建自包含的 vLLM + PyPTO 集成镜像。当前权威
build/test/publish 与 registry raw-manifest/fresh-pull 验证主机均为 0162
（containerd/nerdctl + BuildKit）；当前流程不在 devbox 执行。

---

## 0. 当前镜像状态（2026-09-02）

### 0.1 当前 local candidate：r15 K8 reset + a745（尚未 release-admitted）

```text
tag(local): hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260902-a745-k8-r15
manifest:   sha256:19f51d373c5f9d6171ccf3306f260066e873eda48efca23f5d77b4d6f5e64a7f
config:     sha256:7e5dd8683fda03e3e51a0b5217ae71ab82052173f3659db60fd689ea833ed6eb
pins:       pypto 655c7bda / pypto-lib a745ab659
runtime:    PYPTO_H4_RESIDENT=all (OCI injection; not image Env)
```

r15 与已测 r14b manifest/config 完全相同，0162 local build/audit、matched H4 A/B/A
与 5-case extended gate PASS；无 source/runtime/core overlay。registry tag 尚不存在，
push 缺临时 write credential；route exporter `local-routes.v2` 与 release validator
`recv-meta.v1` 未闭合，因此不能写成 published 或 release-admitted。

matched reset A/B/A 为 `21.617/20.516/21.257 ms`，正式收益
`0.921 ms / 4.296%`。`20.516 ms` 未刷新历史 a745 source-overlay `20.172 ms`；
相对 r12 exact-launcher `20.973 ms` 的 `−0.457 ms` 低于 floor 且合同不同，只作
方向性 sanity check。完整对账：
[`../../benchmark/2026-09-02-k8-historical-performance-reconciliation.md`](../../benchmark/2026-09-02-k8-historical-performance-reconciliation.md)。

### 0.2 当前 release-admitted 基线：r12 whole-step host/graph/submit

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260826-r12
manifest: sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d
config:   sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f
spec:     builds/stepfun-upgrade-20260826-r12.env
runtime:  PYPTO_H4_RESIDENT=all
```

`runtime` 行由 canonical launcher 默认注入，不是镜像 Config Env 或 pypto-lib 代码默认值；
设置 `PYPTO_H4_RESIDENT=none` 可回退。

镜像 pins：

```text
pypto      14de90fd74b3c0716f94b9d4eafdd004d4eaed73
pypto-lib  e6c7d8ec34a05c3051ccf0dd169639f40f041a57
pto-isa    cd4a3d3f7a1a27fcfe536f617e9bca3008929664
PTOAS      307d0484a9e7d5e36f01b253d2bebe4d2f45fe81
simpler    85a82c454074c069315ed6485033c3c2b136e562
ptoas-bin  v0.57
vLLM       1b3e538c35999e62b6d24e0651b3a85b7d16c826
```

已完成的 digest-only 门：

- build spec SHA256 `94e018c0…35f93f27`，base 为已发布 r11 manifest
  `sha256:401ead7d…a67b12`；
- r12 exact spec 顶部仍有继承的“published r9”注释；复现时以机器可执行的
  `BASE=…@sha256:401ead7d…a67b12` 为准，不能按旧注释选 base；
- registry push、raw tag/digest manifest/config、isolated fresh pull、
  digest-only audit/smoke/source-bake 与 non-privileged device smoke 全 PASS；
- immutable gate 仅使用 baked runtime，无 source/runtime/core overlay；
  显式设备 `0–7`、保护 `8–15`，major/minor exact，`privileged=false`；
- Main H4 `all/none` 均 `126/128 = 98.4375%`，mismatch `[20,69]`，
  hidden finite、TP spread `0`；
- MTP BS1/BS16 tokens 均 `[6178,410,303]`，三层 hidden pass rate `1.0`、
  max abs diff `0`；
- dep-only DFX 8/8 `deps.json`、hidden/token exact、tail token `43640`；
  此门不声明 whole-swimlane；
- final release contract `1844/1844 PASS`：
  `0162:…/release-admission-r12-20260826-224620/release_contract.json`
  （SHA256 `511a5459…87f3a`）。
- 2026-08-29 deployment launcher 默认 H4=`all`：r12 **source-default-all** matched
  `none/default/none`
  p50 `30.516/22.606/29.440 ms`，收益 `7.372 ms / 24.591%`；父 env unset 的
  exact launcher 64K/1000 p50 `20.973 ms`、RC=0。该值是 privileged 单臂长门，
  不是当前 100-iter non-privileged A/B/A 的回归基线。完整记录见
  [`../../benchmark/2026-08-29-h4-resident-deployment-contract.md`](../../benchmark/2026-08-29-h4-resident-deployment-contract.md)。

r12 的性能收益证据来自 r11 immutable digest 上的 source/runtime-overlay A/B/A，
不是 r12 immutable-image timing：64K warmup10/iters100 下 ITL
`21.6805 → 21.1150 ms`（`−2.608%`）、graph build `−44.429%`、
graph→first runner `−47.936%`、rank submit envelope `−23.887%`。
`bind.args` 候选 p50 仅 `0.054669 ms`、占 ITL `0.259%`，判定
`no_clear_change`，不再优化。rank loop 仍是 8 个独立 chip submit
（`group_size=1`），不是 native group-submit；各 span 重叠，不得相加。完整记录：
[`../../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)。

### 0.3 前一版 release-admitted 基线：r11 local-owner MoE

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260826-r11
manifest: sha256:401ead7da4f957f6532e380fa1a138eda733fe1dc04b40eabc67d79d62a67b12
config:   sha256:35c42510a64ce3e1c8e899e15c36ab8b534d091ea03a085ec663f18df8706876
spec:     builds/stepfun-upgrade-20260826-r11.env
runtime:  PYPTO_H4_RESIDENT=all
```

r11 以 r10 为 base，仅将 pypto-lib 从 `fe641929` 前进到 `e6c7d8ec`。
Build spec SHA256 `9c272afe…5218c62`；H4 `all/none` precision 均
`126/128`，两模式 output/hidden byte-exact；64K/1000 p50 `21.477 ms`。
immutable r10/r11/r10 A/B/A 的 r11 p50 相对 baseline midpoint 仅
`−0.0065 ms / −0.0299%`，结论为性能中性。最终合同 `20/20 PASS`，
SHA256 `570bb04e…740af7`。完整记录：
[`../../benchmark/2026-08-26-local-owner-moe-r11-release.md`](../../benchmark/2026-08-26-local-owner-moe-r11-release.md)。

### 0.4 历史 release-admitted 基线：r10 packed-NZ MoE fusion

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260825-r10
manifest: sha256:8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b
config:   sha256:38ebba41d6aa0c49940c03e2e7c6fa42d85b61d631c143d38944683d0c657b5f
spec:     builds/stepfun-upgrade-20260825-r10.env
runtime:  PYPTO_H4_RESIDENT=all
```

r10 pins 为 pypto `519b588a` / pypto-lib `fe641929`，final contract
`71/71 PASS`；H4-all 64K/1000 p50 `21.742 ms`，immutable r9/r10/r9
A/B/A 为 `−0.731 ms / −3.241%`。六档单次 latency 中 BS8/BS16 回退，
不得宣称多 BS 性能全面提升。完整记录：
[`../../benchmark/2026-08-25-moe-fusion-image-release.md`](../../benchmark/2026-08-25-moe-fusion-image-release.md)。

### 0.5 历史 release-admitted 基线：upgrade r9

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260824-r9
manifest: sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6
config:   sha256:f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae
runtime:  PYPTO_H4_RESIDENT=all
```

r9 的 registry、precision、Main/MTP liveness、L3/L4 exact、
8/8 chip swimlane/DFX 与远端同步均已闭环；64K/1000 p50 为
`22.253 ms`。未显式设置 H4 时默认 `none`，会回到约 `27.8 ms`。
完整证据：
[`../../benchmark/2026-08-24-upgrade-r9-release.md`](../../benchmark/2026-08-24-upgrade-r9-release.md)。

### 0.6 历史 K8 partial-gate image（2026-08-11）

该历史源码 tip 为 pypto `1c048a744d5f63a8bce1ddb45dac8d1b7f458bb0`、pypto-lib
`cb96747eb21f5f4932d6a24eddaa69c85d095ef6`（= `8e92b468` / `491267c4` 之后叠了 K8
选择性清零）。**这个组合已构建成 immutable image 并在 0162 通过精度 + ITL gate**：

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260811-k8-selective
manifest: sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config:   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
spec:     builds/stepfun-develop-20260811-k8-selective.env
```

镜像 pins：

```text
pypto      1c048a744d5f63a8bce1ddb45dac8d1b7f458bb0
pypto-lib  cb96747eb21f5f4932d6a24eddaa69c85d095ef6
pto-isa    ecb6c303f797749f811a494742c3c08156aacabb
PTOAS      fc8c6caee561914b4fb991dfc8427bb63194269e
simpler    e2efebcbd190302609c0775d2984f409f5f42c76
ptoas-bin  v0.50
vLLM       1b3e538c35999e62b6d24e0651b3a85b7d16c826
```

0162 digest-only、无源码/runtime overlay 的验证：

- audit + smoke 四门全 PASS（`IMAGE_IMMUTABLE_AUDIT` /
  `CANONICAL_ONLY_SYMBOL_AUDIT` / `K8_LANDING_PRESENT` / `[smoke]`）；
- byte-exact `hidden_sha256` `567b206b…f03e` == 生产 baseline、token `14371`；
- N=128 预定义冻结 oracle **三轮** `123/128 = 96.09375%`、`tp_spread_max=0.0`；
- clean ITL bs=1 ctx=65536 blocks=512 iters=100 p50 **`32.14 ms`**
  （pre-K8 `33.84` → **−5.02%**）。

⚠ **本镜像未重跑** Main batch16、MTP batch1/16、六档（BS 1/2/4/7/8/16）64K
golden/A/B、formal matched-source DFX ⇒ **不是完整 production release-qualified**；
完整矩阵的回退基线仍是 Wave5（`sha256:4acc77cd…`）。数据见
[`../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md`](../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md)。

### 更早的 latest-source partial-gate image（历史 pre-fix evidence）

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260806-attn-taskmajor-canonical
manifest: sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479
config:   sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216
```

镜像 pins：

```text
pypto      8e92b46808f9f7c09b6431ad4691503f09c12ee5
pypto-lib  c9af5790d5fe450e14fd43c88099b87539089d17
pto-isa    ecb6c303f797749f811a494742c3c08156aacabb
PTOAS      fc8c6caee561914b4fb991dfc8427bb63194269e
simpler    e2efebcbd190302609c0775d2984f409f5f42c76
ptoas-bin  v0.50
vLLM       1b3e538c35999e62b6d24e0651b3a85b7d16c826
```

credential、pin、clean-tree、CANN absence、prepared-swimlane `RunConfig` 和
A2A3 QK/softmax/online blocks-per-task=`22/16/22` profile 审计均 PASS。
0162 digest-only、无源码/runtime
overlay 的 BS1×64K 整网 50 次 ITL 为
`39.057/39.594/39.612/40.680/40.680 ms`
（min/mean/p50/p99/max），hidden finite、TP spread=0；同 digest 两层
Attention p50 `3.6323 ms`、reference exact、DFX `8/8` rank 完整。
详见
[`../../benchmark/2026-08-06-attention-taskmajor-canonical.md`](../../benchmark/2026-08-06-attention-taskmajor-canonical.md)。

该镜像完成其 `c9af5790` 源码层级的 Attention/ITL/DFX gate，但尚未重跑 Wave5 的 Main
N=128×3、Main batch16 和 MTP 全矩阵，因此不能自动继承完整 production
release-qualified 标签，也不能代表当时 pending 的 `491267c4`。该历史阶段最后一个
**完整 production-matrix release-qualified 基线**是 Wave5；当前直接回退为 r11：

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260803-attn-final-wave5
manifest: sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32
config:   sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394
```

Wave5 镜像内 pin 是 pypto `defa97c5`、pypto-lib `7099476b`，不是当前 r12 源码。
其完整 audit/smoke/Main+MTP matrix PASS，Main N=128 三轮均 `123/128` 且
TP spread=0；64K p50 `49.796 ms`。部署/回归必须按 manifest 核对，不能只看 tag。
2026-08-05 R1 已撤销、R2 从未发布且已被后续镜像 supersede；不得恢复 R2。

本轮 immutable 验证只挂载 driver(ro)、checkpoint/reference(ro)、output(rw)，
无宿主源码；整网 ITL 与两层 DFX 使用 cards `8–15`。作业结束后 container 已退出，
0162 无本轮 NPU 进程残留。

```bash
IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260806-attn-taskmajor-canonical
sudo "$NC" pull "$IMG"
# 核对 manifest sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479
```

历史 `attn-final-canonical`（`76d96bdb`、p50 `50.563 ms`、三轮 `121/128`）和
Wave3（`d58b6be7`、`124/128`、spread=0）只用于演进对账。该历史阶段判定以本节、
[`../version-matrix.md`](../version-matrix.md) 与 2026-08-06 benchmark 为准；
需要完整 production 回退证据时再读取
[`../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md`](../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md)。

## 1. 历史 2026-07-29 镜像内容与 pin

> 本节只描述 `stepfun-develop-20260729-allreduce-push` 这一历史组合，保留作
> 复现/回退 runbook；它不是 2026-08-02 clean canonical candidate 的当前 pin。
> 当前 release 的 pin、digest 和发布判定以 §0、[`../version-matrix.md`](../version-matrix.md)
> 为准。

- **base**: `hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938`
  (自带 CANN 8.5.1 + CANN 9.0.0-beta.1 + vLLM 0.19.0 + vllm-ascend + python3.11.14)
- 本镜像在其上:
  1. **删 CANN 8.5.1**,只留 `cann-9.0.0-beta.1`;并修好 base 把 8.5.1 设成默认后留下的悬空引用
     (ENTRYPOINT / `/etc/profile` / ENV 里 hardcode 的 `cann-8.5.1` → `beta.1`,见 §7)
  2. clone pypto 栈到 `/workspace`,切到 immutable release pin(下表)
  3. `ptoas-bin v0.50`(含 `$PTOAS_ROOT/ptoas` 顶层符号链接 → `bin/ptoas`,codegen 需要)
  4. 编译 `pypto` + `runtime`(`build_runtimes --platforms a2a3`)
  5. **vLLM Track-B 补丁**:`step3p5.py`(tail-only 主网 + `PyPtoMetadataOnlyStep3p5DecoderLayer`)+
     `step3p5_mtp.py`(MTP-proposer 挂点 + MTP3 `hf_overrides` boot fix),来自 gitlab
     `sys/stepcast/vllm:csy/pypto-tail-mtp-integration`
  6. env(CANN beta.1 / PTO_ISA_ROOT / PTOAS / PYTHONPATH / PTO2_RING_*)写进 `/etc/profile.d/pypto-env.sh`
  7. 冒烟脚本 bake 在 `/workspace/pypto-smoke.sh`

| 仓库 | pin | 说明 |
|------|-----|------|
| pypto | `6933b1aa` | 0724 pin(`ca21ab5f`) + 一个 **submodule gitlink bump**：`runtime` → simpler `8459d60f`（见 §2(b)，simpler 换版只能走 gitlink） |
| pypto-lib | `cfbdcce8` | `stepfun/develop`：C/D/G 收口(`563fe62a`) + PERF-C4 TP all-reduce reduce-scatter + push all-gather |
| pto-isa | `ecb6c303` | 0724 镜像 pin |
| PTOAS(src) | `fc8c6cae` | 0724 镜像 pin |
| simpler(pypto/runtime submodule) | `8459d60f` | 0724 pin(`216e7632`) + span-aware child provenance（`worker.py`/`orchestrator.py`，把 0728 candidate 里未提交的 152 行补丁入库；缺它整网 CI 在 `copy_to` interior pointer 处失败） |
| ptoas-bin | `v0.50` | 0724 验证二进制 |

> **镜像 tag**: `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-allreduce-push`
>
> **digest**: `sha256:7924925f4b2816c5645910b90fd2a9fa9469baace2f48f7e0ee41a587bd5d6ba`
>
> **image config**: `sha256:5402e07ba0d19b315935bfda1e9f6b445d1a3fdc9067c634a2ce302fd7f2a3dd`
>
> **安全说明**：最终镜像已在 0162 验证 immutable pins、Git credential
> audit、canonical-only symbol audit 和 smoke。旧 digest
> `sha256:285514c1…` 的源码 checkout
> `.git/config` 中保留了 credential-bearing clone URL，已废弃。当前 digest
> 已将 GitHub remote/submodule URL scrub 为无凭据公开 URL，并在 0162
> 验证 `IMAGE_GIT_CREDENTIAL_AUDIT=PASS`。
>
> **⚠ 该 tag 被重写过一次，早于 2026-07-29 07:00 拉过它的机器必须重新 `pull`。**
> 本机/节点如果缓存了旧 digest，`nerdctl images --digests` 会显示成
> `sha256:8bf62b29…`（旧内容，整网 CI 跑不通）；重新 `pull` 后应变成
> `sha256:7924925f…`。**部署前请先核对 digest，不要只看 tag。**
>
> **0724 base digest**: `sha256:2b0dc4612796a34bea6720ccb4bf8fa3af4ea406cdd0f12add34586ca860d7e0`

> **代码/镜像边界（2026-07-29）**：镜像内 pin = `pypto 6933b1aa` / `pypto-lib cfbdcce8` /
> `pto-isa ecb6c303` / `PTOAS fc8c6cae` / `simpler 8459d60f`，五个仓工作树全 clean
> （`IMAGE_WORKTREE_CLEAN_AUDIT=PASS`）。**产品代码与镜像完全一致**；`stepfun/develop`
> 在镜像之后各多一个**纯测试**提交（`pypto ce7fcb64` = all-reduce 微基准、
> `pypto-lib cc850ee5` = ITL `--active-batch`），不影响镜像内 kernel/runtime 行为，
> 也不需要重建镜像。
> `stepfun-develop-20260726-step3p5-only`(`53eb7212`) 为上一个已发布 tag，保留作回退。
> 本 tag 的验证记录见 [`benchmark/2026-07-28-tp-allreduce-push.md`](../../benchmark/2026-07-28-tp-allreduce-push.md)，
> all-reduce 算法与 race 根因见 [`postmortems/13`](../../postmortems/13-tp-allreduce-pull-notify-race.md)。

---

## 2. 构建(devbox；通用流程，示例以当前 release 为准)

> 下面的构建流程适用于所有 immutable spec。旧的 2026-07-29 baseline 仍保留在
> 历史 spec 中；若目标是复现当前 latest-source canonical image，应使用
> `builds/stepfun-develop-20260806-attn-taskmajor-canonical.env`。

> ### ⚠ 两条硬性流程要求（均为踩过的坑，见 [`postmortems/14`](../../postmortems/14-image-dirty-worktree-unreproducible-pins.md)）
>
> **(a) 推 registry 前必须先在本地验证镜像内容。** 顺序是 build → 本地 `docker run` 核对
> → 才 `docker push`。最小核对集：
>
> ```bash
> IMG=hub.i.basemind.com/stepcast/vllm-pypto:<tag>
> docker run --rm --entrypoint bash "$IMG" -lc '
>   for r in pypto pypto-lib pto-isa PTOAS; do printf "PIN %-10s %s\n" $r $(git -C /workspace/$r rev-parse --short=10 HEAD); done
>   printf "PIN %-10s %s\n" simpler $(git -C /workspace/pypto/runtime rev-parse --short=10 HEAD)
>   d=0; for g in /workspace/pypto /workspace/pypto-lib /workspace/pto-isa /workspace/PTOAS /workspace/pypto/runtime; do
>     n=$(git -C $g status --porcelain | wc -l); [ "$n" != 0 ] && { echo "  DIRTY $g ($n)"; d=1; }; done
>   [ $d = 0 ] && echo WORKTREE_CLEAN=PASS || echo WORKTREE_CLEAN=FAIL'
> ```
>
> 五个 pin 必须与 spec 逐字一致，**且每个仓的工作树必须 clean**——工作树带未提交改动会造成
> "pin 相同但内容不同"，验证结论无法按 pin 复现。`img_regress.sh` 已加
> `IMAGE_WORKTREE_CLEAN_AUDIT` 把这两项固化成 gate。
>
> **(b) simpler(runtime) 的 pin 只能靠 pypto 的 submodule gitlink，不能靠 Dockerfile 里
> 显式 checkout。** Dockerfile 里的 `git fetch + checkout ${SIMPLER_COMMIT}` 在更早的 layer；
> 后面 `pip install -e ${WS}/pypto` / `build_runtimes` 会跑 `git submodule update`，把 runtime
> **切回 gitlink**（镜像内 reflog 可见 `216e7632 → 8459d60f → 216e7632`）。所以要换 simpler
> 版本，**必须在 `csy0225/pypto` 提一个 gitlink bump**（`git update-index --cacheinfo
> 160000,<sha>,runtime`）并把 `PYPTO_COMMIT` 一起前进；spec 里的 `SIMPLER_COMMIT` 只作断言与文档用途。

```bash
cd deployment/docker
# 每次构建的 pins+tag 在一个 spec 文件里(builds/<tag>.env),配方共用单一 Dockerfile。
GH=/data/chensiyu/secrets/github.env GL=/data/chensiyu/secrets/gitlab.env \
bash build.sh builds/stepfun-develop-20260806-attn-taskmajor-canonical.env
# ⚠ 先做 §2(a) 的本地核对，PASS 之后才 push
docker push hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260806-attn-taskmajor-canonical
```

- `build.sh <spec>` 读 spec 里的 pins,以 `--build-arg` 传进 Dockerfile,`-t` 用 `IMAGE_TAG`。
  不带参数默认用最新 spec。**加新 build 见 [§9 组织方式](#9-组织方式--加新-build)**。

- `GH`/`GL` 是含 PAT 的文件,以 BuildKit `--secret` 传入,**不落镜像层**。
  构建前后还会执行 `audit_image_credentials.py`，同时检查当前 credential
  精确值与常见 literal-token 模式；任何命中都会阻止发布，且日志只记录长度与
  SHA256 前缀，不打印 credential。
- 原始 StepCast base
  `stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938`
  的 OCI history 含历史 GitLab token，禁止再直接作为 release base。当前使用
  history-free、rootfs/config 等价的 digest pin：
  `hub.i.basemind.com/stepcast/vllm-pypto@sha256:3d6392588fe9fb6ce4f5852100667d24f09d70f262dbd0ebe6c45b380f49573a`
  （config `sha256:f4c87b469ec42f308340f26ede123a6cc42f726b16b64f82c8b34f12bd990387`）。
- build.sh 做了三件网络相关的事(devbox 内网特性):
  1. `DOCKER_BUILDKIT=1 docker build --network=host`(走宿主路由到代理);
  2. 从官方入口 `deploy.i.shaipower.com/httpproxy` 取代理并以 `--build-arg` 传入
     (github clone/release 经 `proxy.i.shaipower.com:3128`;内网 pip 镜像/gitlab/hub 直连不走代理);
  3. `ptoas-bin` 从 0162 验证过的二进制打进 build context(fork 无 release asset)。
- 编译并发由 spec 的 `BUILD_JOBS` 控制，并同步到
  `CMAKE_BUILD_PARALLEL_LEVEL / MAX_JOBS`；`simpler` runtime 的 variant
  ThreadPool、host/aicpu/aicore ThreadPool 和显式 `cmake --parallel` 三层并发
  也由构建期桥接遵守该值，之后恢复固定 commit 的源码。
  当前 17GB/5 核且无 swap 的 devbox 已用 `BUILD_JOBS=2` 完成构建；若监控到
  内存压力再降低并发，不再把 `MAX_JOBS=1` 当作固定要求。

---

## 3. 部署到 0162(containerd / nerdctl；历史 baseline 示例)

> 本节的设备占用和命令是 2026-07-29 baseline 的历史示例（当时使用 cards
> `8–15`）。2026-08-02 candidate 的 immutable 验证使用 cards `0–7`，且只允许
> 继续定位/复核，不得据此进行 production rollout；实际部署必须按 §0 的 manifest
> 重新核对并由设备 owner 分配 cards。

0162 **没有 docker**,用 containerd 自带的 `nerdctl`(路径 `/mnt/persist/k8s-install/containerd/bin/`)。
0-7 卡通常被 vanilla vLLM oracle(8000)占用,pypto 用 **8-15 卡**。

```bash
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
# Historical 2026-07-29 baseline; do not use as the current candidate.
IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-allreduce-push
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp   # W8A8 ckpt

# 拉取(base blob 已在 containerd content store, 只下增量)
sudo $NC pull "$IMG"

# 起容器(以冒烟为例)。8 卡设备 + manager/hdc/svm + driver 挂载 + ckpt 挂载。
DEVS=""; for i in 8 9 10 11 12 13 14 15; do DEVS="$DEVS --device /dev/davinci$i"; done
sudo $NC run --rm --net host --ipc host --privileged \
  --security-opt apparmor=unconfined \
  $DEVS --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro \
  --shm-size 32g \
  "$IMG" bash -lc 'bash /workspace/pypto-smoke.sh'
```

**nerdctl 必备 flag(否则起不来)**:

| flag | 原因 |
|------|------|
| `--security-opt apparmor=unconfined` | 规避 nerdctl `apparmor_parser resolves to executable in current directory` 报错 |
| `--net host` | 0162 未装 CNI bridge 插件(`/opt/cni/bin/bridge` 缺失) |
| `--privileged --ipc host` | 整网多卡:forked chip 子进程 + 跨卡 IPC(shmem/peer-access)需要 |
| `--shm-size 32g` | 多进程共享内存 |
| `bash -lc '...'` | **登录 shell** 才会 source `/etc/profile.d/pypto-env.sh`(PATH/PYTHONPATH/PTO2_RING_*) |

---

## 4. 冒烟验证(镜像 + 硬件基本可用)

镜像内 bake 了 `/workspace/pypto-smoke.sh`,单卡即可:

```bash
sudo $NC run --rm --net host --security-opt apparmor=unconfined \
  --device /dev/davinci8 --device /dev/davinci_manager \
  --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  "$IMG" bash -lc 'bash /workspace/pypto-smoke.sh'
```

**期望输出**(0724-derived v0.50 environment):

```
[smoke] ptoas   : ptoas 0.50
[smoke] pypto   : 0.1.0
[smoke] simpler : OK
[smoke] runtime : /workspace/pypto/runtime/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
[smoke] vllm-pypto CI dir: OK
[smoke] PASS
```

---

## 5. 历史 2026-07-29 整网精度与 ITL 记录

> 本节数据属于历史 `stepfun-develop-20260729-*` 镜像，不是当前
> `attn-final-wave5` 的 release gate 结果。当前 0162 release-qualified 的稳定性、
> 64K/batch16 ITL 和 DFX 证据见 §0 及
> [`../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md`](../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md)。

权威 runner:`tests/step3p5/ci/run_whole_network_ci.py`(preflight → Main 45 层 hidden-only
8-step → MTP45/46/47 → 清理)。用 §3 的 8 卡 run 命令,把最后一行换成:

```bash
"$IMG" bash -lc "cd /workspace/vllm-pypto && \
  python -m tests.step3p5.ci.run_whole_network_ci \
    --ckpt $CKPT --devices 8,9,10,11,12,13,14,15 --out /tmp/n1_ci"
```

> 要留存 artifact/日志时,加 `-v <宿主目录>:/tmp/n1_ci -v <宿主目录2>:/tmp/n1_ci_artifacts`。

**canonical 金标准**:token `6127` → argmax `303`。

**实测（本发布镜像，2026-07-29，45 层 hidden-only decode，W8A8，TP=8，active_batch=1）**：

| context | ITL p50 | mean | p99 | min |
|--------:|:-:|:-:|:-:|:-:|
| 1024 | 70.177 | 70.196 | 70.597 | 69.898 |
| 8192 | 71.450 | 71.549 | 72.699 | 71.167 |
| 32768 | 77.522 | 77.459 | 78.116 | 77.043 |
| **65536** | **83.349** | 83.529 | 84.658 | 82.902 |

**64k decode ITL ≈ 83 ms/step；1k→64k 只涨 13.3 ms（+18.8%）。**
active_batch 扫描（同工作点）：bs2 `87.8` / bs4 `104.0` / bs8 `145.1` ms，**bs16 撞 device HBM**。

> **⚠ `--shm-size` 要跟着 batch 调**：dummy KV 池按 host 共享内存分配，`--num-blocks`
> 与 active_batch 同步放大（bs=8/nb=4096 时 11.29 GiB/rank × 8 卡 ≈ **90 GiB**）。
> 上面示例的 `--shm-size 32g` 只够 bs=1（≈11.6 GiB）；跑 bs≥2 的 64k 要显式加大
> （实测用 `--shm-size 400g`）。bs=16 即使 shm 给够也会在 device HBM 上 OOM。

> **完整数据 + DFX 逐 kernel 拆解见
> [`../../benchmark/2026-07-29-release-image-64k-dfx-itl.md`](../../benchmark/2026-07-29-release-image-64k-dfx-itl.md)。**
> 那里也给出了占比的正确读法（DFX 插桩 span 是真实单步的 5.21×，份额不能直接当延迟占比）。

> **⚠ 历史数据已作废**：本节此前记录的 2026-07-23 实测（64k ≈ **654 ms/step**）以及由它得出的
> 「1k→64k 仅 +19 ms → 整网 decode **计算受限**，非 attention/KV 受限」两条都不再适用：
> - 绝对值：同一 harness、同一批 flag、同一台机器，654 → 83.5 ms（**7.8×**）。
>   塌掉的是与 context 无关的固定 floor（≈635 → ≈70 ms），随 context 增长的那部分两次相当
>   （+18.7 vs +13.3 ms）。归因需要拿旧镜像跑同一条命令做受控 A/B，尚未做。
> - 结论：「计算受限、attention 不重要」是把"近平坦"误读的结果 —— 真实原因是当时有个巨大的
>   context 无关开销把 attention 的增长淹没了。
> 另：该节原先引用的 `benchmark/2026-07-23-step3p5-decode-64k-itl.md`（`≈590 ms` raw `rt.run`，
> 含 lm_head）**在本仓中并不存在**，溯源是待补项；且它与本表口径不同（含 lm_head + 只计
> raw `rt.run()`），不可直接比较。

## 7. 已知坑(都已修进本镜像 / Dockerfile)

- **删 CANN 8.5.1 的悬空引用**:base 镜像把 8.5.1 设成默认——**ENTRYPOINT** 的 `&&` 链、
  `/etc/profile:29`、一堆 `ASCEND_*` ENV 都 hardcode `cann-8.5.1`。删了 8.5.1 后每个 bash
  登录 shell / 容器启动 source 缺失文件 → **rc=1 起不来**。修:ENTRYPOINT/profile/ENV 里
  `cann-8.5.1` → `cann-9.0.0-beta.1`(beta.1 有同名 `share/info/ascendnpu-ir/bin/set_env.sh`)。
- **`$PTOAS_ROOT/ptoas` 顶层符号链接**:pypto codegen(`pto_backend.py`)按 `$PTOAS_ROOT/ptoas`
  找(不是 PATH)。ptoas-bin release 解出来是 `bin/ptoas`,需补 `ptoas-bin/ptoas → bin/ptoas`。
- **simpler `c7fdc574` 编不过**:develop tip = `36957c6b` + 9 个 WIP commit,其中 Phase-24
  `import_ipc` 半成品(`orchestrator.cpp:41` `get_worker` 笔误 + `control_import_ipc` 缺头声明)。
  已把 simpler develop 回退到可编译的 `36957c6b`(0162 验证过的 .so 就是它),pypto develop
  gitlink 同步(`8af501fc`);原 `c7fdc574` 存 simpler tag `backup/stepfun-develop-c7fdc574-20260723`。
- **`build_runtimes` 内部 clone pto-isa**:为 a2a3 平台会 `git clone hw-native-sys/pto-isa`
  (pin `pto_isa.pin`)到 `runtime/build/pto-isa`;compile 步用 `git config --global http.version HTTP/1.1`
  + 代理让它过。
- **stale oracle**:见 §5,`run_whole_network_ci` step2 的 FAIL 是 harness 常量过时,非精度问题。

---

## 8. `/workspace` 挂载(统一管理, 可选)

镜像已把仓库 bake 在 `/workspace`(可独立运行)。做统一管理时可把宿主仓库挂载覆盖
`-v <宿主>/workspace:/workspace`,但宿主仓库需已用**同 python3.11.14 + CANN beta.1** 编译
(`pypto` 扩展 + runtime `.so`),ABI 一致;否则用镜像内 bake 的即可。

## 9. 组织方式 / 加新 build

配方(单一 `Dockerfile`)与「一次构建的版本规格」分离,后续新 commit 构建只加规格 + 一行登记:

```text
deployment/docker/
├── Dockerfile          # 稳定构建配方(全 pins 走 ARG, 不写死具体值)
├── build.sh            # bash build.sh builds/<spec>.env → 读 spec 传 --build-arg + tag
├── pypto-smoke.sh      # bake 进镜像 /workspace/pypto-smoke.sh
├── builds/             # 每次镜像构建一个 spec(pins + IMAGE_TAG)
│   └── stepfun-develop-20260726-step3p5-only.env
├── README.md           # 本文档 + 下方「构建登记表」
├── .dockerignore / .gitignore
└── (ptoas-bin.tgz / build_*.log 由 build.sh 生成, gitignored)
```

**加一个新 build**(例:新 pypto commit):

1. `cp builds/stepfun-develop-20260726-step3p5-only.env builds/<新tag>.env`,改 `IMAGE_TAG` + 变动的 `*_COMMIT`。
2. `bash build.sh builds/<新tag>.env && docker push hub.i.basemind.com/stepcast/vllm-pypto:<新tag>`。
3. 在下方**构建登记表**加一行(tag / 日期 / pins 摘要 / 验证状态)。
4. 拿到 manifest/config digest 后，同步 [`../../STATUS.md`](../../STATUS.md) §2/§4、
   [`../../progress/landed.md`](../../progress/landed.md) 表 A、
   [`../version-matrix.md`](../version-matrix.md) 与对应 [`../../benchmark/`](../../benchmark/)
   报告；镜像新增优化、性能证据层级和限制条件必须写进 STATUS §4。
5. Dockerfile **不动**(除非配方本身要改,如新踩坑修复——那属于所有 build 共享的配方演进)。

> 旧 spec 文件保留(可复现历史某次镜像);Dockerfile 单一、不随 build 复制,避免配方漂移。

## 构建登记表

| IMAGE_TAG | 日期 | pypto / pypto-lib / pto-isa / PTOAS / simpler / ptoas-bin | 验证(0162) |
|-----------|------|----------------------------------------------------------|-------------|
| `stepfun-upgrade-20260902-a745-k8-r15` | 2026-09-02 | `655c7bda` / `a745ab659` / `cd4a3d3f` / `307d0484` / `85a82c45` / `v0.57` | **local-only candidate，未发布**。manifest `sha256:19f51d37…64a7f`、config `sha256:7e5dd868…d6eb`；0162 immutable audit、matched reset A/B/A、5-case extended gate PASS，无 overlay。registry push/fresh pull、route v2/v1 publication、64K/1000 historical guard 与 final contract 未完成；`20.516 ms` 不是历史新低 |
| `stepfun-upgrade-20260826-r12` | 2026-08-27 | `14de90fd` / `e6c7d8ec` / `cd4a3d3f` / `307d0484` / `85a82c45` / `v0.57` | **当前 release-admitted**。manifest `sha256:ba42fd19…e6eb805d`，config `sha256:b36f0cec…3d73a5233f08f`；prepared TaskArgs signature/cache + rank submit envelope 优化已 bake。Registry/fresh pull、baked-runtime identity、Main H4 all/none `126/128`、MTP BS1/BS16、dep-only DFX、五仓远端 exact 均 PASS；final contract `1844/1844`，SHA `511a5459…87f3a`。性能收益来自 r11 digest source-overlay A/B/A，不是 r12 immutable timing；正式仍是 serial 8-rank independent submit；2026-08-29 canonical launcher 默认 H4=`all`，`none` 可回退 |
| `stepfun-upgrade-20260826-r11` | 2026-08-26 | `519b588a` / `e6c7d8ec` / `cd4a3d3f` / `307d0484` / `85a82c45` / `v0.57` | **前一版 release-admitted / r12 直接回退**。manifest `sha256:401ead7d…a67b12`，config `sha256:35c42510…06876`；replicated-input local-owner MoE。H4 all/none `126/128` 且 parity PASS，64K/1000 p50 `21.477 ms`；r10/r11/r10 A/B/A 性能中性；final contract `20/20`，SHA `570bb04e…740af7` |
| `stepfun-develop-20260723` | 2026-07-23 | `8af501fc` / `4c48215b` / `ecb6c303` / `72ada0a1` / `36957c6b` / `v0.45` | 冒烟 PASS + 整网 decode `6127→303` / step2→`6127`(与 vanilla 逐 token 一致)✅ |
| `stepfun-develop-20260724` | 2026-07-24 | `ca21ab5f` / `fd26b1be` / `ecb6c303` / `fc8c6cae` / `216e7632` / `v0.50` | 合并 origin/main + IPC 权重 interior 指针 provenance 修复（解 `submit_next_level child_memory` 卡点）。冒烟 PASS(ptoas 0.50) + 整网 8 步 decode `6127→303→1207→6127`(与 live vanilla 逐 token 一致)✅ |
| `stepfun-develop-20260729-allreduce-push` | 2026-07-29 | `6933b1aa` / `cfbdcce8` / `ecb6c303` / `fc8c6cae` / `8459d60f` / `v0.50` | **2026-07-29 正式发布基线**。registry digest `sha256:7924925f4b281…`（config `sha256:5402e07ba0d19…`）；PERF-C4 TP all-reduce → reduce-scatter + push all-gather，simpler span-aware provenance 入库。0162 immutable-image 回归：audit 5 pin 一致 + `IMAGE_GIT_CREDENTIAL_AUDIT` / `IMAGE_WORKTREE_CLEAN_AUDIT` / `CANONICAL_ONLY_AUDIT` / `ALLREDUCE_PUSH_PRESENT` 全 PASS；smoke PASS；整网 CI `rc=0` 198.3 s，6 项 check 全 true（`tokens_exact` / `eight_steps` / `result_clean` / `pypto_hidden_only` / `step0_hidden_saved` / `process_rc_zero`），token `303,1207,19384,872,428,6127,4231,2636`；`hidden_tp_spread` 在 ci/main + rep1/rep2/rep3 共 **4×8 = 32 步全 `0.0`**；ITL p50 `65.942 ms`(ctx=1024) / `66.455 ms`(ctx=4096)（`--num-blocks 32`），权威 64k 工作点 `--num-blocks 512` 为 p50 `83.349 ms`，DFX 拆解见 benchmark/2026-07-29 |
| `stepfun-develop-20260726-step3p5-only` | 2026-07-26 | `ca21ab5f` / `53eb7212` / `ecb6c303` / `fc8c6cae` / `216e7632` / `v0.50` | registry digest `sha256:99b2b971…`（config `sha256:d2964610…`）；0162 credential/symbol/ldd audit、smoke、unit `136 passed/4 skipped`、contract `15 passed`；唯一 program=`whole_decode_step3p5`；N=256 raw `240/256=93.75%`，与清理前 canonical token/hidden `256/256` exact、`max_abs_diff=0`、TP spread `0`，step127/128/255 PASS |
| `stepfun-develop-20260729-perf-h1` | 2026-07-29 | `1f704616` / `4513007d` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **PERF-H1 device-memset 优化镜像**（perf 上取代 C4 发布，正确性等价）。registry digest `sha256:b4e8c8a457a5…`。在 C4 上前进 pypto→`1f704616`(gitlink→simpler `e2efebcb`)、pypto-lib→`4513007d`(=`cfbdcce8` + ITL `--active-batch` + MTP CI oracle-dir 可配置化)。0162 回归：smoke PASS；整网 CI `ok=true`（Main token `303,1207,19384,872,428,6127,4231,2636` exact + MTP single/batch16 `6178,410,303` exact，`hidden_tp_spread=0`）；N=256 H1 vs C4 **token 256/256 exact**（step127/128/255 含），全步 finite（raw-hidden run-to-run 抖动=C4 push all-reduce 归约顺序，非 H1 回归，H1a-vs-H1b 复跑证实）。**ITL p50(`--num-blocks 512`)：1024 `50.9` / 8192 `52.0` / 32768 `58.0` / 65536 `64.1` ms，较 C4 同工作点降 23–27%**。PMU/scope 与 C4 逐项一致（cube_int8 46.35%，ring heap 79.9%，dropped=0）。benchmark: [`../../benchmark/2026-07-29-perf-h1-image-itl-dfx.md`](../../benchmark/2026-07-29-perf-h1-image-itl-dfx.md)。⚠ MTP CI oracle-wiring 修复 `0f3650c7`(test-only) 为 mount 验证、未 rebuild 烤进本镜像 |
| `stepfun-develop-20260802-attn-final` | 2026-08-02 | `1f704616` / `76d96bdb` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **v1 失败证据**：缺 `defa97c5` 动态 SPMD codegen 修复，immutable-image compile 报 launch-bound 变量未声明；非 candidate |
| `stepfun-develop-20260802-attn-final-v2` | 2026-08-02 | `defa97c5` / `76d96bdb` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **v2 非 canonical**：代码可运行，但 image config 仍含旧 CANN 8.5.1 字符串；历史 `123/128` 不得借给 clean 镜像 |
| `stepfun-develop-20260802-attn-final-canonical` | 2026-08-02 | `defa97c5` / `76d96bdb` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **clean canonical candidate / release blocked**。manifest `sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d`，config `sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea`；audit/smoke/64K ITL/DFX PASS，p50 `50.563 ms`；fresh-oracle 三轮均 `121/128=94.53125%`，低于 raw gate |
| `stepfun-develop-20260802-attn-final-wave3` | 2026-08-03 | `defa97c5` / `d58b6be7` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **历史中间版本**。canonical TP communication window 增第三 completion wave；manifest `sha256:5c38b669269f686a105e810a39a97242ee223ceb0a3d7437fc306df78a920b22`，config `sha256:c2de331156de9f182f54a0ee840564850a522d1214bb3b83b0f6d0c3e4160cb0`；audit/smoke PASS；N=128 `124/128`、spread=0；two-layer harness 尚未 AST 对齐 |
| `stepfun-develop-20260802-attn-final-wave4` | 2026-08-03 | `defa97c5` / `d7e1381b` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **历史 immutable candidate，非正式 release；已由 Wave5 取代**。manifest `sha256:8125c678779c332d196b3d770242659d9a86185e0a8d96d89681647b00c864ab`，config `sha256:c340001f791bd4666310b2f1755daba5492fec8c65f126888d46ed4366131c92`；audit/smoke/compile/64K ITL/DFX PASS，p50 `50.204 ms`；N=128 Run1 `122/128` 但 step2 spread=`2.0`，Run2 `123/128` spread=0；raw token gate PASS，但未通过当时 TP-spread stability gate |
| `stepfun-develop-20260803-attn-final-wave5` | 2026-08-03 | `defa97c5` / `7099476b` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **历史完整 production-matrix release-qualified 基线；当前直接回退为 r11**。manifest `sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32`，config `sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394`；self-target TPUT source publication + 既有三波 lifetime；audit/smoke/Main+MTP compile、Main N=128×3、Main batch16、MTP batch1/batch16×2、64K/batch16 ITL/DFX PASS；N=128 三轮均 `123/128` 且 spread=0；64K p50 `49.796 ms` |
| `stepfun-develop-20260805-attn-final-canonical` | 2026-08-05 | `defa97c5` / `91c7f46e` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **R1 REVOKED，禁止交付**。manifest `sha256:fb613c2d5a74592f248c6d923e3ada6582edbe40349ada530017e622ca735b23`，config `sha256:95bf9657adc09650fc85c23544756169519f85c145b42b14641bfc41e6c173e2`；bs1/64K 两层 50 次 timing 与数值检查完成，但 immutable DFX 因镜像缺 `l2_swimlane_reuse_dep_gen` 接口失败，没有最终 swimlane；不得用源码挂载绕过 |
| `stepfun-develop-20260805-attn-final-canonical-r2` | 2026-08-05 | `8e92b468` / `91c7f46e` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **HISTORICAL / NEVER PUBLISHED / SUPERSEDED**。当时包含 prepared-swimlane 修复但构建被停止；无 manifest/config digest，不得恢复或借用其它镜像数据。历史记录见 [`../../benchmark/2026-08-05-attention-canonical-r1-r2.md`](../../benchmark/2026-08-05-attention-canonical-r1-r2.md) |
| `stepfun-develop-20260806-attn-taskmajor-canonical` | 2026-08-06 | `8e92b468` / `c9af5790` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **历史 latest-source partial gate；pre-fix evidence**。manifest `sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479`，config `sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216`；BS1×64K 整网 p50 `39.612 ms`，hidden finite、TP spread=0；同 digest 两层 p50 `3.6323 ms`、exact、DFX 8/8 rank。该镜像不包含当时 pending 的 `491267c4`，且未重跑完整 Main/MTP production matrix |
| `stepfun-develop-20260808-moe-opt-latest-source` | 2026-08-08 | `8e92b468` / `491267c4` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **PENDING / 未构建**。spec 要求 `ATTN_TASK_PROFILE=a2a3` 和 `l2_swimlane_reuse_dep_gen`；`BUILD_JOBS` 使用 build.sh 默认值。构建及 0162 N=128、六档 BS×64K、DFX/swimlane 完成前无 manifest/config digest |

## Pin 依据

见 [`../version-matrix.md`](../version-matrix.md) 与 [`../../STATUS.md`](../../STATUS.md)
的 Pin Snapshot(2026-07-23 行)+「两条线(项目结构)」。
