# 版本矩阵

5 个代码仓 + 3 个工具链支柱的兼容矩阵。下面"已验证组合"表的一行是一个
已知端到端能跑的状态集。跨行混搭**不**支持，混搭后必须重新验证。

## 已验证组合

### 当前 canonical SRC / matched candidate（2026-09-01）

> **SRC 已落地，IMG 尚未 release-admitted。** `pypto@655c7bda` 修复
> local-owner persistent reset ABI 回退，`pypto-lib@a745ab659` 为最新
> routed GMM latch participant 优化；两者分别从 `14de90fd`、`e6c7d8ec`
> exact-lease 前进到 `stepfun/develop`。candidate digest 仅用于验证，
> 不替代下方 r12 release IMG。

```text
candidate image: hub.i.basemind.com/stepcast/vllm-pypto@sha256:19f51d373c5f9d6171ccf3306f260066e873eda48efca23f5d77b4d6f5e64a7f
config:          sha256:7e5dd8683fda03e3e51a0b5217ae71ab82052173f3659db60fd689ea833ed6eb
runtime:         PYPTO_H4_RESIDENT=all
```

| 槽位 | Pin | 备注 |
|------|-----|------|
| pypto | `655c7bda7b0a0b495a3387b2570ea68c4a857a40` | local-owner reset ABI exact profile；parent `14de90fd` |
| pypto-lib / vllm-pypto | `a745ab659c68afca01de37870e29ccb9648d7c87` | routed GMM latch participant reduction；parent `e6c7d8ec` |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` | unchanged |
| PTOAS | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` | unchanged |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` | unchanged |
| ptoas-bin | `v0.57` | unchanged |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` | unchanged |

验证：matched immutable H4=`all` A/B/A p50 `21.617/20.516/21.257 ms`，
gain `0.921 ms` > required `0.616 ms`；5-case extended correctness
admission v2 checks 16/16 PASS，设备/任务清理 PASS。尚未采集 a745 provenance
匹配的 `recv_meta`/route-identity publication sidecar，因此不得写成 IMG release
admission；旧 r12 count-only sidecar 不可复用。

### 当前 r12 whole-step host/graph/submit release admission（2026-08-27）

> **0162 immutable-image release admission：PASS。**
> r12 bake 入 prepared TaskArgs signature/cache 与 rank submit envelope 优化；
> registry publication、fresh digest pull、baked-runtime identity、Main precision、
> MTP BS1/BS16、dep-only DFX、五仓远端 exact ref 与最终 release contract 已闭环。
> H6 性能收益来自 r11 immutable digest 上的 source/runtime-overlay A/B/A；不得写成
> “r12 immutable A/B/A 实测”。2026-08-29 canonical deployment launcher 已默认
> 注入 `PYPTO_H4_RESIDENT=all`，`none` 可回退；镜像 Config Env 与代码默认仍未 bake。

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260826-r12
manifest: sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d
config:   sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f
spec:     docker/builds/stepfun-upgrade-20260826-r12.env
runtime:  PYPTO_H4_RESIDENT=all
```

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `14de90fd74b3c0716f94b9d4eafdd004d4eaed73` | TaskArgs signature/cache + rank submit envelope 优化 |
| pypto-lib / vllm-pypto | `e6c7d8ec34a05c3051ccf0dd169639f40f041a57` | replicated-input local-owner MoE |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` | 与 r11 相同 |
| PTOAS | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` | source pin |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` | pypto runtime gitlink |
| ptoas-bin | `v0.57` | binary SHA `2183e4cf…` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` | immutable checkout |

当前门结论：

- build spec SHA256 `94e018c0…35f93f27`；base 为已发布 r11 manifest
  `sha256:401ead7d…a67b12`，构建时重新 materialize 全部 pinned source tree；
- registry tag/digest raw manifest/config、fresh pull、digest-only audit/smoke/
  source-bake 与 non-privileged device smoke 全 PASS；
- immutable baked runtime 无 source/runtime/core overlay；显式设备 `0–7`，
  保护 `8–15`，major/minor exact，`privileged=false`；
- Main H4 `all/none` 均 `126/128 = 98.4375%`，mismatch `[20,69]`，
  hidden finite、TP spread `0`；
- MTP BS1/BS16 tokens 均为 `[6178,410,303]`，三层 hidden pass rate `1.0`、
  max abs diff `0`；
- dep-only DFX 为 8/8 `deps.json`，hidden/token exact、tail token `43640`；
  本门不声明 whole-swimlane；
- r11 source-overlay A/B/A（H4 all、64K、warmup10/iters100）：
  ITL `−0.5655 ms / −2.608%`、graph build `−1.8183 ms / −44.429%`、
  graph→first runner `−1.4851 ms / −47.936%`、rank submit envelope
  `−0.2875 ms / −23.887%`、graph→chip done `−1.7624 ms / −8.443%`；
  span 有重叠，不得相加；
- `bind.args` 为 `0.054449 → 0.054669 ms`，候选 ITL 占比 `0.259%`，
  判定 `no_clear_change`，不再继续优化；
- H4 source-default-all matched `none/default/none` p50 `30.516/22.606/29.440 ms`，
  default=`all` 相对 midpoint 收益 `7.372 ms / 24.591%`，三臂 hidden/token exact；
- 父 env unset 的 exact launcher 64K/1000 p50 `20.973 ms`、RC=0，context curve
  `20.139/20.698/20.827/20.821 ms`，pre/postflight clean；
- 产品 host loop 仍是 serial 8-rank，发出 8 个独立 chip submit
  （`group_size=1`），不是 native group-submit；commit 标题中的
  `parallelize rank submit` 不等于正式设备门已证明并行 fanout；
- final contract `1844/1844 PASS`：
  `0162:…/release-admission-r12-20260826-224620/release_contract.json`，
  SHA256 `511a5459…87f3a`。

完整记录：
[`../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)。

### 前一版 r11 replicated-input local-owner MoE（2026-08-26）

> **0162 immutable-image release admission：PASS。**
> r11 以 r10 为 base，仅将 pypto-lib 前进到 `e6c7d8ec`；registry/fresh pull、
> source identity、H4 all/none precision/parity、64K ITL 与 immutable
> r10/r11/r10 A/B/A 已闭环。

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260826-r11
manifest: sha256:401ead7da4f957f6532e380fa1a138eda733fe1dc04b40eabc67d79d62a67b12
config:   sha256:35c42510a64ce3e1c8e899e15c36ab8b534d091ea03a085ec663f18df8706876
spec:     docker/builds/stepfun-upgrade-20260826-r11.env
runtime:  PYPTO_H4_RESIDENT=all
```

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` | 与 r10 相同 |
| pypto-lib / vllm-pypto | `e6c7d8ec34a05c3051ccf0dd169639f40f041a57` | replicated-input local-owner MoE |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` | 与 r10 相同 |
| PTOAS | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` | source pin |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` | 与 r10 相同 |
| ptoas-bin | `v0.57` | binary SHA `2183e4cf…` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` | immutable checkout |

验证结论：

- build spec SHA256 `9c272afe…5218c62`，base 为 r10 manifest
  `sha256:8510f30e…e9907b`；
- H4 `all/none` 均 `126/128 = 98.4375%`、mismatch `[20,69]`；
  output token、active hidden、row0 hidden 两模式逐步 byte-exact；
- H4-all 64K/1000 p50 `21.477 ms`；
- immutable r10/r11/r10 A/B/A p50 `21.751/21.745/21.752 ms`，
  r11 相对 midpoint `−0.0065 ms / −0.0299%`，结论为**性能中性**；
- final contract `20/20 PASS`：
  `0162:…/release-admission-r11-20260826-113923/release_contract.json`，
  SHA256 `570bb04e…740af7`。

完整记录：
[`../benchmark/2026-08-26-local-owner-moe-r11-release.md`](../benchmark/2026-08-26-local-owner-moe-r11-release.md)。

### 历史 r10 packed-NZ MoE fusion release admission（2026-08-25）

> **0162 immutable-image release admission：PASS。**
> Build/audit/fresh-pull/compile、Main+MTP liveness、N=128/H4 parity、ITL、
> matched A/B/A、六档 correctness、前五层 hidden/swimlane/DFX、
> `stepfun/develop` exact-lease sync 与最终 release contract 均已闭环。

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260825-r10
manifest: sha256:8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b
config:   sha256:38ebba41d6aa0c49940c03e2e7c6fa42d85b61d631c143d38944683d0c657b5f
runtime:  PYPTO_H4_RESIDENT=all
```

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` | 与 r9 相同 |
| pypto-lib / vllm-pypto | `fe641929dbf959d887ad111f3bd7cac0b73fa34b` | `stepfun/develop` 远端 exact-lease fast-forward 后已复核 |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` | 上游声明 pin |
| PTOAS | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` | source pin |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` | 与 r9 相同 |
| ptoas-bin | `v0.57` | binary SHA `2183e4cf…` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` | immutable checkout |

历史门结论：

- source unit `162 passed`；镜像 audit/smoke、whole compile、registry/fresh pull PASS；
- Main 8-step、MTP single/batch16 PASS；
- accepted-oracle N=128：H4 `all/none` 均 `127/128`、mismatch `[94]`、
  finite、TP spread 0；256/256 hidden tensor pair byte-exact；
- H4-all 64K/1000 p50 `21.742 ms`，curve p50
  `21.503/22.106/22.135/22.285 ms`，ITL admission PASS；
- immutable r9/r10/r9 A/B/A p50 `22.524/21.821/22.580 ms`，r10 相对
  baseline midpoint `22.552 ms` 为 `-0.731 ms / -3.241%`，三臂 hidden/token exact；
- 六档 BS correctness `6/6` exact、`12/12` tensor health PASS；单次诊断中
  BS8/BS16 分别回退 `+5.677/+0.551 ms`，不宣称多 BS 性能全面提升；
- L3/L4 hidden exact，8/8 rank DFX 完整；fused E3→E4 median
  `44.97/41.62 us`，routed down `16.18/16.44 us`；
- pypto-lib `stepfun/develop` 已从 `bf3ff440` fast-forward 到 `fe641929`；
- final contract：
  `0162:…/r10-release-admission-20260825-150350/release_contract.json`，
  `pass=true`，SHA256 `bcdd0b11…8dafca`。

完整记录：
[`../benchmark/2026-08-25-moe-fusion-image-release.md`](../benchmark/2026-08-25-moe-fusion-image-release.md)。

### 历史 upgrade r9 release admission（2026-08-24）

> **0162 immutable-image admission：PASS。** Registry push、raw manifest/config、
> fresh pull、precision、Main/MTP liveness、前五层 hidden/swimlane/DFX 与五仓同步均闭环。
> ⚠ 性能准出依赖运行合同 `PYPTO_H4_RESIDENT=all`；镜像 Config Env 没有内置该值，
> 未设置时代码默认 `none`。

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260824-r9
manifest: sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6
config:   sha256:f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae
runtime:  PYPTO_H4_RESIDENT=all
```

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` | `stepfun/develop` 远端已复核 |
| pypto-lib / vllm-pypto | `bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373` | H4 resident constants + r9 holders/DFX |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` | 上游声明 pin |
| PTOAS | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` | source pin |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` | `stepfun/develop`；旧 tip 有 backup ref |
| ptoas-bin | `v0.57` | 与该历史组合的 pypto 声明配对 |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` | immutable checkout |

验证结论：

- 64K/1000：默认 `none` p50 `27.812 ms`；`all` p50 `22.253 ms`；
- precision `127/128 = 99.21875%`，`all/none` 输出 parity；
- Main 8-step、MTP single、MTP batch16 PASS；
- L3/L4 `torch.equal=true`，8/8 rank `chip_swimlane_records.json`；
- final contract：
  `0162:…/r9-release-admission-20260824-151848/release_contract.json`，`pass=true`。

完整记录：
[`../benchmark/2026-08-24-upgrade-r9-release.md`](../benchmark/2026-08-24-upgrade-r9-release.md)。
该组合不自动声明 Phase 28 live serving 已完成；正式 launcher 仍需接入 H4 env。

### K8 immutable image（精度 + ITL gate PASS，2026-08-11）

> **audit/smoke + 双精度门 + clean ITL：PASS ON 0162（digest-only，无 overlay）。**
> ⚠ 未重跑 Main batch16 / MTP / 六档 64K / formal DFX，因此**不**继承 Wave5 的
> 全量 release-qualified 标签；完整矩阵回退基线仍是 Wave5。

```text
tag:      hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260811-k8-selective
manifest: sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config:   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
```

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | image config/runtime audit PASS |
| pypto | `1c048a744d5f63a8bce1ddb45dac8d1b7f458bb0` | = `8e92b468` + reset 仪表 + K8 选择性清零 |
| pypto-lib / vllm-pypto | `cb96747eb21f5f4932d6a24eddaa69c85d095ef6` | = `27a43f6a` + K8 control-prefix 重排 |
| pto-isa | `ecb6c303f797749f811a494742c3c08156aacabb` | immutable pin |
| PTOAS | `fc8c6caee561914b4fb991dfc8427bb63194269e` | immutable pin |
| simpler | `e2efebcbd190302609c0775d2984f409f5f42c76` | pypto runtime gitlink |
| ptoas-bin | `v0.50` | binary release |
| vLLM overlay | `csy/pypto-tail-mtp-integration@1b3e538c35999e62b6d24e0651b3a85b7d16c826` | immutable checkout |

数据：[`../benchmark/2026-08-11-k8-selective-window-zeroing-image.md`](../benchmark/2026-08-11-k8-selective-window-zeroing-image.md)。

### 历史 latest-source canonical image（partial gate，2026-08-06）

> **Attention/ITL/DFX gate：PASS ON 0162。**
> 尚未重跑完整 Main/MTP production matrix，因此不自动继承 Wave5 的全量
> release-qualified 标签。

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | image config/runtime audit PASS |
| pypto | `8e92b46808f9f7c09b6431ad4691503f09c12ee5` | prepared-worker immutable swimlane dep-gen reuse |
| pypto-lib / vllm-pypto | `c9af5790d5fe450e14fd43c88099b87539089d17` | 历史 image pin；不包含当时 pending 的 `491267c4` |
| pto-isa | `ecb6c303f797749f811a494742c3c08156aacabb` | immutable pin |
| PTOAS | `fc8c6caee561914b4fb991dfc8427bb63194269e` | immutable pin |
| simpler | `e2efebcbd190302609c0775d2984f409f5f42c76` | pypto runtime gitlink |
| ptoas-bin | `v0.50` | binary release |
| vLLM overlay | `csy/pypto-tail-mtp-integration@1b3e538c35999e62b6d24e0651b3a85b7d16c826` | immutable checkout |
| **historical latest-source partial image** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260806-attn-taskmajor-canonical@sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479` | config `sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216` |

验证：credential、pin、clean tree、CANN absence、prepared-swimlane 接口和 A2A3
QK/softmax/online blocks-per-task=`22/16/22` profile audit PASS；digest-only、
无源码/runtime overlay。
BS1×每请求64K 整网 50 次 min/mean/p50/p99/max =
`39.057/39.594/39.612/40.680/40.680 ms`，hidden finite、TP spread=0。
同 digest 两层 Attention p50 `3.6323 ms`、reference exact、DFX 8/8 rank。

构建 spec：
[`docker/builds/stepfun-develop-20260806-attn-taskmajor-canonical.env`](docker/builds/stepfun-develop-20260806-attn-taskmajor-canonical.env)。
完整记录：
[`../benchmark/2026-08-06-attention-taskmajor-canonical.md`](../benchmark/2026-08-06-attention-taskmajor-canonical.md)。

### 历史 pending latest-source MoE spec（未构建，已由 r9 supersede）

| 槽位 | Pin | 备注 |
|------|-----|------|
| pypto | `8e92b46808f9f7c09b6431ad4691503f09c12ee5` | prepared-worker immutable swimlane dep-gen reuse |
| pypto-lib | `491267c45875e9b1e0071eed224e2e73526799e2` | active-route scheduling + MoE route/precision harness |
| Attention profile | `a2a3` | required |
| L2 swimlane capability | required | required for DFX |
| Build jobs | default | `BUILD_JOBS` intentionally omitted from spec |
| Image | — | no manifest/config digest until 0162 build and gates pass |

构建 spec：
[`docker/builds/stepfun-develop-20260808-moe-opt-latest-source.env`](docker/builds/stepfun-develop-20260808-moe-opt-latest-source.env)。

### 历史完整 production-matrix 回退基线：Wave5（2026-08-03）

> **发布状态：RELEASE-QUALIFIED ON 0162。** 其它机器/架构未由本轮独立证明。

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | image config/runtime audit PASS |
| pypto | `defa97c526fec7e8f032dbbfcc39c820add02bf7` | dynamic SPMD launch-bound codegen fix |
| pypto-lib / vllm-pypto | `7099476b7c4f13112b159e237e7a64344803caf0` | self-target TPUT source publication + 既有三波 lifetime；Main/MTP/harness 对齐 |
| pto-isa | `ecb6c303f797749f811a494742c3c08156aacabb` | immutable pin |
| PTOAS | `fc8c6caee561914b4fb991dfc8427bb63194269e` | immutable pin |
| simpler | `e2efebcbd190302609c0775d2984f409f5f42c76` | pypto runtime gitlink |
| ptoas-bin | `v0.50` | binary release |
| vLLM overlay | `csy/pypto-tail-mtp-integration@1b3e538c35999e62b6d24e0651b3a85b7d16c826` | immutable checkout |
| **Wave5 release** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260803-attn-final-wave5@sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32` | config `sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394` |

验证：audit/smoke/Main+MTP compile/codegen PASS；Main N=128 预定义三轮均
`123/128`、finite、TP spread=0；Main batch16 `8/8 exact`、spread=0；MTP batch1
与 batch16×2 token/hidden/TP spread 全通过。64K p50 `49.796 ms`，rank2 LOW-WAIT
makespan `38.367 ms`、TP AR compute `2.437 ms`；batch16/context1 p50
`112.827 ms`，rank2 makespan `107.076 ms`、TP AR compute `2.429 ms`。

构建 spec：
[`docker/builds/stepfun-develop-20260803-attn-final-wave5.env`](docker/builds/stepfun-develop-20260803-attn-final-wave5.env)。
Wave3/Wave4 为历史中间版本；Wave4 的 TP-spread blocker 已由 Wave5 关闭。

### 历史 2026-08-05 R1/R2（已 supersede）

| 项目 | R1 | R2 |
|---|---|---|
| pypto | `defa97c526fec7e8f032dbbfcc39c820add02bf7` | `8e92b46808f9f7c09b6431ad4691503f09c12ee5` |
| pypto-lib | `91c7f46ee949045e2fce807276412b48d8121763` | 同 R1 |
| image | manifest `sha256:fb613c2d5a74592f248c6d923e3ada6582edbe40349ada530017e622ca735b23` | 从未发布 |
| 最终状态 | REVOKED：prepared-swimlane 接口缺失 | NEVER PUBLISHED / SUPERSEDED |

R1 不得用于交付；R2 不得恢复。二者只作为失败过程记录；在该历史阶段，源码和 pending
构建对象以本页 2026-08-08 spec 为准，当时最后一个完整 production-matrix release
仍是 Wave5。本段的源码/pending 目标已由后续 r9、r10、r11、r12 组合取代；
r9 是升级任务 release admission，不取代 Wave5 的历史完整 production-matrix 回退口径。详见
[`../benchmark/2026-08-05-attention-canonical-r1-r2.md`](../benchmark/2026-08-05-attention-canonical-r1-r2.md)。

### 历史 clean canonical candidate（2026-08-02）

> **历史发布状态：BLOCKED。** 下表是该历史源码与镜像内容的权威 pin；镜像 audit、smoke、
> 64K ITL 和 DFX 已通过，但 fresh-oracle N=128 三轮均为
> `121/128=94.53125% < 95%`，所以不能标记为正式 release。

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | clean canonical image config/runtime 仅保留该版本 |
| pypto | `defa97c526fec7e8f032dbbfcc39c820add02bf7` | 动态 SPMD launch bound 的 orchestration codegen 变量重命名/声明修复；已合入 `stepfun/develop` |
| pypto-lib / vllm-pypto | `76d96bdbeac280f12ecf626b1bbd722b9278719e` | workload-derived attention、Full SV+segment recurrence、Full/SWA out-proj cast fusion、dense RMS/down-cast、当时的 two-wave TP AR；后续由 Wave3/Wave4 取代 |
| pto-isa | `ecb6c303f797749f811a494742c3c08156aacabb` | 镜像显式源码 pin |
| PTOAS | `fc8c6caee561914b4fb991dfc8427bb63194269e` | 镜像显式源码 pin |
| simpler | `e2efebcbd190302609c0775d2984f409f5f42c76` | pypto `runtime` submodule |
| ptoas-bin | `v0.50` | binary release |
| vLLM overlay | `csy/pypto-tail-mtp-integration@1b3e538c35999e62b6d24e0651b3a85b7d16c826` | immutable checkout |
| Python | `3.11.14` | 镜像内 `/usr/local/python3.11.14/bin/python3` |
| **clean canonical candidate** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260802-attn-final-canonical@sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d` | config/image ID `sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea`；audit/smoke/ITL/DFX PASS，raw precision gate FAIL |

验证结论：

- `IMAGE_CONFIG_CANN_851_AUDIT`、`IMAGE_WORKTREE_CLEAN_AUDIT`、
  `IMAGE_GIT_CREDENTIAL_AUDIT`、`CANONICAL_ONLY_AUDIT`、
  `CANN_851_RUNTIME_AUDIT`、`EXPECTED_OPTIMIZATION_SYMBOL_AUDIT`、
  `PTOAS_LDD_AUDIT` 与 smoke 全 PASS；
- immutable 验证只挂载 driver(ro)、checkpoint(ro)、output(rw)，无宿主源码挂载；
- 64K hidden-only ITL：min `49.213`、mean `50.568`、p50 `50.563`、
  p99/max `52.537 ms`；
- DFX LOW-WAIT reference 是 rank2，不是 rank5/rank7。rank2 makespan
  `38.924 ms`，`tp_all_reduce` critical-path compute `2.049 ms`；rank5 的
  `344.553 ms` TP AR compute 主要吸收 collective 自旋等待；
- fresh oracle 三轮均 `121/128`，所有 hidden finite；run2/run3 分别出现瞬态
  TP spread，故发布阻塞；
- `/workspace/pto-isa` 是显式 pin 的外部源码；`/workspace/pypto/runtime/build/pto-isa`
  是 `build_runtimes` 生成/克隆的构建树，两者职责不同，不能把后者当作 release pin。

构建 spec：

- [`docker/builds/stepfun-develop-20260802-attn-final.env`](docker/builds/stepfun-develop-20260802-attn-final.env)：v1，动态 SPMD codegen 缺失，失败；
- [`docker/builds/stepfun-develop-20260802-attn-final-v2.env`](docker/builds/stepfun-develop-20260802-attn-final-v2.env)：v2，image config 含旧 CANN 路径，非 canonical；
- [`docker/builds/stepfun-develop-20260802-attn-final-canonical.env`](docker/builds/stepfun-develop-20260802-attn-final-canonical.env)：历史 clean candidate，raw precision gate 阻塞；
- [`docker/builds/stepfun-develop-20260802-attn-final-wave3.env`](docker/builds/stepfun-develop-20260802-attn-final-wave3.env)：历史三波 lifetime 中间版本；
- [`docker/builds/stepfun-develop-20260802-attn-final-wave4.env`](docker/builds/stepfun-develop-20260802-attn-final-wave4.env)：历史 Wave4 immutable candidate；
- [`docker/builds/stepfun-develop-20260803-attn-final-wave5.env`](docker/builds/stepfun-develop-20260803-attn-final-wave5.env)：完整 production matrix 的 Wave5 回退基线。

### 历史已发布组合（2026-07-29）

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | 0162 device verified |
| Firmware | `7.8.0.7.220` | 与 driver 成对 |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `6933b1aa838ebc81643166eb2cf686af894d543c` | 0724 pin(`ca21ab5f`) + `runtime` submodule gitlink bump → simpler `8459d60f` |
| pypto-lib / vllm-pypto | `cfbdcce858e63b9fb3775111dff1b20e97b24808` | GitHub `stepfun/develop`；C/D/G 收口(`563fe62a`) + PERF-C4 TP all-reduce reduce-scatter + push all-gather；唯一 Main=`models.step3p5.decode_fwd:whole_decode_step3p5` |
| pto-isa | `ecb6c303f797749f811a494742c3c08156aacabb` | 与 0724 镜像一致 |
| PTOAS | `fc8c6caee561914b4fb991dfc8427bb63194269e` | 与 0724 镜像一致 |
| simpler | `8459d60f04b64b74322e965e0dd038ab26165124` | pypto `runtime` gitlink；0724 pin(`216e7632`) + span-aware child provenance（入库 0728 candidate 里未提交的 152 行补丁） |
| ptoas-bin | `v0.50` | binary sha256 `ba93fabeff6dc7fdcd2278a72fd1d4fd92cb2949faedbc83fa58e801bd5ff23b` |
| vLLM overlay | `csy/pypto-tail-mtp-integration@1b3e538c35999e62b6d24e0651b3a85b7d16c826` | build 时按 commit checkout，不能只依赖可变 branch |
| Python | `3.11.14` | 镜像内 `/usr/local/python3.11.14/bin/python3` |
| **2026-07-29 正式发布基线** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-allreduce-push@sha256:7924925f4b2816c5645910b90fd2a9fa9469baace2f48f7e0ee41a587bd5d6ba` | config `sha256:5402e07ba0d19b315935bfda1e9f6b445d1a3fdc9067c634a2ce302fd7f2a3dd`；含 PERF-C4 TP all-reduce reduce-scatter + push all-gather 与 simpler span-aware provenance；0162 immutable-image 回归见 benchmark/2026-07-28-tp-allreduce-push.md（4×8 步 `hidden_tp_spread` 全 `0.0`，`IMAGE_WORKTREE_CLEAN_AUDIT=PASS`）。⚠ 该 tag 曾指向一份不可用镜像，**早期拉过的机器须重新 pull 并核对 digest** |
| 上一个已发布镜像（代码 pin 53eb7212，保留回退） | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260726-step3p5-only@sha256:99b2b9718cfa6bf0bb87b221f7d565bf23afd2b89a30ba150e523c44a536ed81` | config `sha256:d296461051559e6ea0e22d04a4cc44f749c82f19a50418fe6db75387f1f067e9` |

验证结论：

- **该历史发布镜像的 0162 immutable-image 回归**：5 个 pin 与 spec 逐字一致；
  `IMAGE_GIT_CREDENTIAL_AUDIT` / `IMAGE_WORKTREE_CLEAN_AUDIT` / `CANONICAL_ONLY_AUDIT` /
  `ALLREDUCE_PUSH_PRESENT` 全 PASS；smoke PASS；整网 CI `rc=0`（198.3 s）6 项 check
  全 true，token `303,1207,19384,872,428,6127,4231,2636`；`hidden_tp_spread` 在
  ci/main + rep1/rep2/rep3 共 4×8 = 32 步全 `0.0`（PERF-C4 的准出指标）；
  ITL p50 `65.942 ms`(ctx=1024) / `66.455 ms`(ctx=4096)。
- 当时对应的 GitHub 代码（`pypto-lib cfbdcce8`）默认入口为 `whole_decode_step3p5`，与镜像内 pin 一致。
  `stepfun/develop` 在镜像之后各多一个**纯测试**提交（`pypto ce7fcb64` all-reduce 微基准、
  `pypto-lib cc850ee5` ITL `--active-batch`），产品代码无差异，无需重建镜像；
  下面 `563fe62a` / `53eb7212` 相关结论属于 0726/0728 阶段的历史记录；
- N=256 canonical-only 与清理前镜像 token/hidden `256/256` exact，`max_abs_diff=0`，
  TP spread `0.0`，compatibility removal regression PASS；
- 与清理前 canonical 镜像产物 token/hidden `256/256` bit-exact，
  step127/128/255 PASS；
- 对同一 vanilla oracle，canonical-only 为 `240/256=93.75%`，低于历史
  `>=95%` raw gate；不能写成 vanilla raw precision PASS；
- 2026-07-27 后 retired unroll source、rollback selector 和自定义 Main
  module/name 参数均已删除；旧 rollback smoke 只作为历史证据，不属于当前镜像 gate；
- 最终镜像内默认 holder 实际打印
  `program=whole_decode_step3p5`；8-step device smoke hidden 全 finite、
  TP spread `0.0`，除已知 stale oracle step2 外其余 `7/8` token exact；
- 已发布 0726 镜像的 N=256 raw `240/256=93.75%`；与既有 canonical N=256 artifact
  token/hidden `256/256` exact、`max_abs_diff=0`、TP spread `0.0`，
  step127/128/255 全通过；
- 该 0726/0729 历史证据只证明 canonical-only Main replacement，不等价于当时已证明
  完整 production Main+MTP serving 可无条件平替；
- 0728 阶段的 C/D/G 结论（BS1/2/16、N=256、Main 8-step）产自工作树 dirty 的本地 candidate，
  需在 `8459d60f` 基线上复核，跟踪见 [`../blockers.md`](../blockers.md) 与
  [`../postmortems/14-image-dirty-worktree-unreproducible-pins.md`](../postmortems/14-image-dirty-worktree-unreproducible-pins.md)。

C/D/G candidate 设备证据：BS1/2/16 单步与 BS1 persistent 4-step 通过；固定 expert lane physical bases 修复 BS1 batch-extension invariance。

构建 spec：
[`docker/builds/stepfun-develop-20260729-allreduce-push.env`](docker/builds/stepfun-develop-20260729-allreduce-push.env)。

### 历史生产目标（2026-06-22，禁止作为当前 pin）

| 槽位 | Pin | 备注 |
|------|-----|------|
| Driver | `25.5.2` | Phase 16 最小 |
| Firmware | `7.8.0.7.220` | chip flash，持久 |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `csy0225/pypto stepfun/develop:b00c8b23` | 比 origin/main 多 3 commit（DFX env hook + repros + simpler submodule pin） |
| pypto-lib | `csy0225/pypto-lib stepfun/develop:9c4773f` | 比 origin/main 多 ~9 commit（step3p5 模型 + Phase 19 padding + ST 脚手架 + dev-workflow docs；误置的 phase tracker 已撤回） |
| pto-isa | `csy0225/pto-isa stepfun/develop:e25732f0` | = origin/main（无本地 patch） |
| PTOAS | `csy0225/PTOAS stepfun/develop:da011a3d` | = origin/main；binary `ptoas-bin` `v0.45` |
| simpler | `csy0225/simpler a6e06406`（pypto submodule） | 比 origin/main 多 4 patch（zero-size view + `--no-as-needed` libhcomm + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip） |
| ptoas-bin | `v0.45` | binary release |
| Python | `3.11.14` | venv 在 `<workspace>/.venv311` |

验证证据见 [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
"2026-06-22（早段）—— 验证基线"。

## 兼容规则

### pypto / pto-isa / PTOAS / ptoas-bin

pypto codegen 产 MLIR 给 PTOAS 吃。wire format 会偶尔变；mismatched
pypto + ptoas-bin 编译时会报 parser error。

历史已知 mismatch：
- pypto 越过 `505abd64`（TCIOp `hasCustomAssemblyFormat`）之后需要
  ptoas-bin ≥ `v0.45`。Phase 19 blocker 1 就是这个 mismatch —— pypto
  跑前了，ptoas-bin 还在 `v0.44`。

规则：bump pypto 跨过会动 MLIR op 的上游 commit 时，同时 bump ptoas-bin。

### pypto / simpler

simpler 是 pypto 的 git submodule，在 `pypto/runtime/`。`pypto` 仓的
pin 决定编哪个 simpler commit。更新 simpler 时必须
`git submodule update` 并 commit pypto 侧的 submodule pin。

当前 r12 组合使用 simpler `85a82c454074c069315ed6485033c3c2b136e562` 与
pypto `14de90fd74b3c0716f94b9d4eafdd004d4eaed73`；两者的最终组合已在 0162
完成 immutable admission。r9-r11 使用同一 simpler pin，pypto 当时为
`519b588a7a6461cac0e443e853accf29479c1d15`。K8 与 Wave5 历史组合使用 simpler
`e2efebcbd190302609c0775d2984f409f5f42c76`。2026-07-29 历史发布组合使用的是 simpler
`8459d60f04b64b74322e965e0dd038ab26165124`，由 pypto `6933b1aa` 固定。
**Dockerfile 里的显式 checkout 不算**——`pip install -e pypto` 期间的
`git submodule update` 会把它切回 gitlink，所以换 simpler 必须同时 bump pypto。
下方 `a6e06406` 仅属于 2026-06-22 历史组合。

### CANN

CANN beta.1 **必需**。CANN GA 会让 simpler init 失败（见
[`phase16-three-pillars.md`](phase16-three-pillars.md) "CANN GA failure
mode"）。**不要**升级 CANN 除非 Huawei 出了新 beta 或 GA 明确修复了
AICPU `libaicpu_extend_kernels.so` push path。

### Driver + firmware

总是成对。driver-only 或 firmware-only 升级未验证。
`support_shmem_map_exbus` cap 由两者共同 gate。

## 升级顺序（全部前进时）

推荐顺序：

1. Firmware（写 chip flash；先做，其余还在老版本上）
2. Driver（重装到 host filesystem；要 daemonset drain）
3. 重启主机
4. CANN（**只**在 Huawei 出新 beta/GA 验证过兼容时）
5. simpler（pypto submodule）
6. pypto + pto-isa + PTOAS + pypto-lib（任意顺序，但重装时按
   pypto → pto-isa → PTOAS → pypto-lib 顺序）
7. ptoas-bin（binary drop-in，跟 PTOAS source pin 配对）

每一步后都跑 smoke + simpler L3 allreduce 验证。

## 项目之外但邻接的仓库

| 仓库 | 角色 | 我们跟踪的 pin |
|------|------|----------------|
| `vLLM stepcast fork` | Phase 2 集成目标 | `csy/pypto-tail-mtp-integration@1b3e538c`（gitlab.basemind.com/sys/stepcast/vllm） |
| `pypto-serving` | 早期 serving wrapper（早于本项目） | 不积极跟踪；需要时见 `<workspace>/pypto-serving/` |

## 相关文档

- [`phase16-three-pillars.md`](phase16-three-pillars.md) —— driver/
  firmware/CANN 为什么硬绑
- [`machine-recovery.md`](machine-recovery.md) —— 怎么安装/升级
- [`../STATUS.md`](../STATUS.md) —— 最新 pin snapshot 一行
- [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
  "Pin snapshot history" —— 历史 pin
