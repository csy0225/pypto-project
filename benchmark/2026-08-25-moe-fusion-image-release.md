# 2026-08-25 · packed-NZ MoE fusion r10 镜像发布

> **状态：release-admitted。**
> Registry、compile、liveness、precision、ITL、六档 correctness、
> immutable-image A/B/A、outer hidden/DFX、`stepfun/develop` exact-lease
> fast-forward 和最终 `release_contract.json` 已全部闭环。当前
> release-admitted 镜像为 r10；r9 转为历史发布基线。

## 1. 镜像与源码身份

| 字段 | 值 |
|---|---|
| 机器 | `gpu-a910x-0162.host.platform.shaipower.com` |
| Tag | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260825-r10` |
| Manifest | `sha256:8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b` |
| Config | `sha256:38ebba41d6aa0c49940c03e2e7c6fa42d85b61d631c143d38944683d0c657b5f` |
| Base | r9 manifest `sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6` |
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` |
| pypto-lib | `fe641929dbf959d887ad111f3bd7cac0b73fa34b` |
| pypto-lib tree | `5d8f7e647cab301ee5bb2f0175fec4d91bfa71e8` |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` |
| PTOAS source / binary | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` / `v0.57` |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |
| staging branch | `csy0225/pypto-lib:release/r9-moe-fusion-20260825` |
| `stepfun/develop` | `fe641929dbf959d887ad111f3bd7cac0b73fa34b`（由 `bf3ff440` exact-lease FF） |

`fe641929` 是 `bf3ff440` 的单父 fast-forward commit，包含 22 个文件的
`6720 insertions / 1338 deletions`。关键源码 SHA256：

```text
decode_fwd.py                da36c09dc275838ee364f76342d74717338ef313d912ba2b372808530489dd14
weight_loader.py             bedcef8c3e9749d2b2d8e4f64e599e363009e3ce810d6dac9dfc85a21fa4d723
runtime_tensor_compat.hpp    902143c2e70df480c16746c19f0e5ea34ba335ffc6074954883b40f90abc878f
```

## 2. 实现边界

r10 在 current-r9 路径上带入：

- routed W13/W2 `FRACTAL_NZ` packed weight；
- `GMM1 + SwiGLU + requant` mixed AIC/AIV external fused task；
- external routed-down kernel；
- down 的 BS-aware worker grid：BS1 为 `23 AIC + 46 AIV`，multi-batch
  为 `22 AIC` 对应 dual-AIV；
- routed-down task ID 到 combine 的显式依赖；
- 新 runtime `ChipTensor` / 旧 runtime `Tensor` 的 external ABI 兼容层；
- current-r9 的 INT32 route sidecar、H4 holder 和 IPC provenance 合同继续保留。

source-overlay 的 matched A/B/A 和 five-layer DFX 是本次镜像化之前的先行证据；
其阶段数字与 E5→E6 判定见
[`2026-08-21-pypto-vs-vllm-ascend-stage-table.md`](2026-08-21-pypto-vs-vllm-ascend-stage-table.md)
§0.5。那些数据不冒充本 r10 immutable image 的最终门；r10 自身的六档
correctness 与 outer DFX 结果见 §6。

## 3. 可复现构建与镜像审计

构建 spec：
[`../deployment/docker/builds/stepfun-upgrade-20260825-r10.env`](../deployment/docker/builds/stepfun-upgrade-20260825-r10.env)。

0162 构建上下文：

```text
/mnt/persist/chensiyu/workspace/moe-fusion-release-20260825/
  image-build/
  logs/build-20260825-r10.log
```

构建输入 SHA256：

```text
Dockerfile       52432adebe8092c42c992e3748d436e1eab8f14ed4217613de0f7b5146bae2d2
build.sh         f0885119ec0e1f45aebcbeda86787e7d17e1fe69ba855a9cd4e58b2264a23f12
build spec       91f9ac5a6eea4cce1ff9409f73b9b44f0fb71f2e2ffeae13b4fb73d3b5377ce2
src-pins.shas    268a62168ddc1405b024d10c4800677633e19d9097b326e951c8ef841f4c0a38
```

Dockerfile 在从已发布 r9 派生时先删除 base 里的五仓源码目录，再从
`src-pins.tgz` 按 immutable commit 重建，避免把 base 工作树与新 pin 混合。

已完成：

| 门 | 结果 |
|---|---|
| source focused unit | **PASS**：`162 passed`，rc=0 |
| base credential audit | **PASS** |
| final image credential audit | **PASS** |
| 五仓 pin + clean worktree | **PASS** |
| prepared swimlane reuse capability | **PASS** |
| ptoas | **PASS**：`0.57` |
| image smoke | **PASS** |
| r10 external extension audit | **PASS** |
| whole compile | **PASS**：rc=0，`24.769 s`，无 source/runtime overlay |

证据：

```text
source-gates/unit-r9-20260825-123609/
image-audit/
compile-r10-20260825-125029/
```

## 4. Registry 发布

```text
publication-20260825-125156/
```

- push rc=0；
- 隔离 namespace `pypto-r10-verify-20260825-125156` fresh pull rc=0；
- fresh-pull `RepoDigests` 与 config 精确匹配；
- registry raw manifest/config 精确匹配；
- `registry_verdict.json` 与 `remote_registry_verdict.json` 均为 `pass=true`。

这证明 registry 中的对象身份成立；完整 release admission 由 §7 的最终合同给出。

## 5. 整网 liveness 与 precision

### 5.1 Full liveness

Digest-only、`PYPTO_H4_RESIDENT=all`：

| stage | rc | 结果 |
|---|---:|---|
| Main hidden 8-step | 0 | token `303,1207,19384,872,428,6127,4231,2636` exact |
| MTP single | 0 | 3 层 token exact |
| MTP batch16 | 0 | 3 层 token exact |

`whole_network_report.json` 的 `ok=true`，source 为 clean
`fe641929` / tree `5d8f7e64`。

证据：

```text
liveness-r10-full-20260825-125532/
```

### 5.2 N=128 accepted-oracle replay

两臂均使用 accepted oracle：

```text
oracle SHA256 eb561cf8eb1de49cccefe2cbda91071d3fed8fa163fa84a3461c68eb70a241c2
seed          6127
threshold     95%
```

| H4 模式 | aligned | mismatch | finite | TP spread max | gate |
|---|---:|---|---|---:|---|
| `all` | `127/128 = 99.21875%` | `[94]` | true | `0.0` | PASS |
| `none` | `127/128 = 99.21875%` | `[94]` | true | `0.0` | PASS |

两臂 128 个 output token 逐项相同。严格 parity admission 进一步确认：

- `256/256` hidden/active-hidden tensor pair `torch.equal` 且文件 byte-equal；
- `512/512` tensor finite；
- 两臂 report 除 `run_sec` 外 semantic row exact；
- active tensor 与 report 的 TP spread max 均为 `0.0`。

```text
precision-r10-h4-all-none-parity-verdict.json
  sha256 ab2c186641722b9eaadedccff2810e65827693604ab87d94610b9a0f51ed33a2
precision-r10-h4-all-none-parity-verdict.txt
  sha256 b925a73308c9719dd007dfdf758a104476c4e8c7e0b4b6767e329dc4a2ea2d43
```

当前两臂原始证据：

```text
precision-replay-r10-h4-all-20260825-130058/
precision-replay-r10-h4-none-20260825-130930/
```

本轮复用了已验收 oracle，没有重新启动 vanilla 生成新 trajectory；因此只按
accepted-oracle replay 报告，不写成“fresh vanilla oracle 已重建”。

## 6. 性能、六档 BS 与 DFX

截至 2026-08-25，以下 immutable-image 门已全部闭环：

| 门 | 状态 | 证据 / 结论 |
|---|---|---|
| matched A/B/A：r9 / r10 / r9 | **PASS** | p50 `22.524 / 21.821 / 22.580 ms`；hidden/token exact |
| r10 BS1×64K 1000-iter | **PASS** | `21.492 / 22.296 / 21.742 / 27.286 / 36.385 ms`（min/mean/p50/p99/max，H4=all） |
| 四点 context curve | **PASS** | p50 `21.503 / 22.106 / 22.135 / 22.285 ms`（1K/8K/32K/64K） |
| BS `1/2/4/7/8/16` focused exact | **PASS** | `6/6` r9/r10 `torch.equal`；`12/12` tensor health PASS |
| L3/L4 outer hidden | **PASS** | exact；SHA `5aca3716…108ee8b9` / `0308be31…e400a4` |
| 8/8 chip swimlane / DFX | **PASS** | analyzer `pass=true`、`blockers=[]`；五类 8-rank artifact 均为 `8/8` |
| E3→E4 / E4→E5 image-level 阶段 | **MEASURED / structural PASS** | active-rank median `44.97/41.62 us`、`16.18/16.44 us`（L3/L4） |
| 完整 E5→E6 endpoint | **n/a** | 仍无统一 shared + TP-AR + global-fence semantic endpoint |

### 6.1 正式 ITL admission 与 immutable-image A/B/A

正式 ITL admission：

```text
itl-20260825-132536/itl_admission.json
sha256 e494134b1282c5dd62a00f699fd1e67edcc592a06ab24d3a20c8743ab8fa1214
```

结构门与性能门均为 `pass=true`。64K/1000 p50 `21.742 ms`：

- 相对 source campaign 的 matched baseline midpoint `22.6615 ms`：
  `−0.9195 ms / −4.0575%`；
- 相对 r9 published H4-all `22.253 ms`：
  `−0.511 ms / −2.2963%`。

第一项使用的是先前同源码 campaign 的 matched baseline midpoint；最终发布收益
以下面的 immutable-image r9/r10/r9 三臂为准。

immutable-image A/B/A 合同为 BS1、ctx `65536`、`num_blocks=512`、
warmup `10`、iters `100`、cards `0–7`、`PYPTO_H4_RESIDENT=all`：

| arm | 镜像 | p50 | mean | p99 | hidden / token |
|---|---|---:|---:|---:|---|
| A1 | r9 | `22.524 ms` | `22.862 ms` | `28.542 ms` | exact |
| B | r10 | **`21.821 ms`** | **`21.937 ms`** | `28.338 ms` | exact |
| A2 | r9 | `22.580 ms` | `22.633 ms` | `24.208 ms` | exact |

baseline p50 midpoint 为 `22.552 ms`，A1/A2 bracket 仅 `0.056 ms`；
r10 相对 midpoint 为 **`−0.731 ms / −3.241%`**。三臂 hidden SHA 均为
`567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e`，
tail token 均为 `14371`，finite、TP spread `0.0`。

p99 只低于 A1、但高于 A2，因此不能从这组三臂声称 tail 全面改善；发布性能
结论限定为 matched p50/mean 改善，并由 1000-iter ITL admission 旁证。

```text
image-aba-r9-r10-r9-20260825-142702/
  aba_admission.json
    sha256 8d4224e0214b71bae01efe24393e5886375e04dff5481ffd34ba19e3821ddb0e
```

先前 source-overlay A/B/A 的 p50 为 `22.786 / 21.854 / 22.537 ms`，
相对 midpoint `−0.8075 ms / −3.563%`；它只作为与最终 immutable-image
方向一致的历史旁证。

### 6.2 六档 correctness 与 diagnostic latency

immutable r9/r10 两臂合同为 cards `0–7`、`PYPTO_H4_RESIDENT=all`、
ctx `128`、`num_blocks=32`、warmup `1`、iters `1`，同一 resident holder
依次执行 BS `1/2/4/7/8/16` 并保存 next hidden。两臂均为 digest-only，
无 source/runtime overlay。

correctness verdict 为 `pass=true`：

- 六档全部 r9/r10 `torch.equal=true`、`max_abs=0`；
- `12/12` arm-batch tensor health PASS；
- shape 均为 `[8,16,4096]`、dtype `BF16`、finite；
- TP spread max `0.0`、inactive 区域 exact zero。

单次 ITL 只用于伴随诊断，不是性能准入：

| BS | r9 | r10 | r10 − r9 | correctness |
|---:|---:|---:|---:|---|
| 1 | `22.673 ms` | `21.404 ms` | `−1.269 ms` | exact |
| 2 | `23.240 ms` | `21.828 ms` | `−1.412 ms` | exact |
| 4 | `27.103 ms` | `26.376 ms` | `−0.727 ms` | exact |
| 7 | `34.305 ms` | `33.112 ms` | `−1.193 ms` | exact |
| 8 | `35.124 ms` | `40.801 ms` | **`+5.677 ms`** | exact |
| 16 | `51.317 ms` | `51.868 ms` | **`+0.551 ms`** | exact |

因此本轮只能声明“六档 correctness PASS”，不能声明“六档性能全部提升”。
BS8 与 BS16 的单样本方向为回退，后续若作为性能门，必须增加 warmup/iters
并做 matched 重复采样。

证据：

```text
six-batch-r9-r10-h4-20260825-140158/
  six_batch_r9_r10_verdict.json
    sha256 9e58959ff49fe0bf663b69a89ab0ddf0f28c550a1eb3449ee544f884c3a66a85
  six_batch_r9_r10_verdict.txt
    sha256 526efd1dc05b12c0eadb7cacd443c38209e27fc58c0907e96bdd036aabde5bc4
  evidence_manifest.json
    sha256 d4cc411ba45ca9d2dc64ee49a9fe042b3ee6a27510d0192c2a29f890fd4eb2b2
```

### 6.3 Immutable r10 outer hidden / DFX

合同：BS1、ctx `65536`、`num_blocks=512`、warmup `3`、iters `20`、
cards `0–7`、`PYPTO_H4_RESIDENT=all`、`PYPTO_MOE_DFX_PROFILE=packed-nz`，
无 source/runtime overlay。route sidecar SHA 为
`d0f0713d1b635123c9588c65a7238ee2a28869045b309df1ceef673d4d74bea0`。

outer admission `pass=true`：

- `container.rc=0`、`outer_admission.rc=0`、`gate.rc=0`；
- L3/L4 hidden 对 accepted r9 golden `torch.equal` 且 file SHA 相同；
- `chip_swimlane_records.json` / `deps.json` / `name_map.json` /
  `critical_path_report.md` / merged swimlane 均为 `8/8`；
- analyzer `profile=packed-nz`、`pass=true`、`blockers=[]`；
- fused grid `24 AIC + 48 AIV`、down grid `23 AIC + 46 AIV` 精确；
- route-empty ranks 按 predicate skip，无伪造空计算 task。

五层运行 p50/mean 为 `10.620/10.623 ms`。阶段值是同一 physical task
在 active ranks 上的 wall envelope，不是尚未结构化导出的 per-stage CP：

| 阶段 | L3 median（min–max） | L4 median（min–max） | worker grid |
|---|---:|---:|---|
| `E3→E4` fused GMM1+SwiGLU+requant | `44.97 us`（`44.36–70.74`） | `41.62 us`（`40.84–69.72`） | `24 AIC + 48 AIV` |
| `E4→E5` routed down | `16.18 us`（`14.58–28.22`） | `16.44 us`（`14.68–29.14`） | `23 AIC + 46 AIV` |
| fused→down task gap | `7.30 us`（`6.82–7.86`） | `7.23 us`（`6.80–7.98`） | one remaining boundary |

这组 immutable-image 数字复现了 source-overlay 的方向：`E3→E4` 已进入
历史 vLLM `45.25 us` 的同量级；`E4→E5` 不是当前瓶颈。per-stage numeric
critical-path contribution 仍未由 analyzer sidecar 导出，不能用全 rank
makespan 代替。

证据根：

```text
r10-outer-swimlane-dfx-20260825-141817/
  outer_admission.json
    sha256 e1eab1c6179f1ade55bed81160ef106c1345d25bcd63f0a6fd92d7dcdad20a7d
  critical_path_metrics.json
    sha256 0bbe5a409a445a0f3a916ea54676dd24b370537454446f6baaabd2b30009e722
  runtime/dfx_analysis/moe_dfx_report.json
    sha256 c5b07841dd03d5a5d8522b1a25f0525720db289ab3dea9294af28b4fe9c3f435
  runtime/dfx_analysis/moe_critical_path_report.md
    sha256 9c05e84cc17791ede816c954b75c6a6f2e964932434b130b20b6141f7ffde42b
  runtime/hidden_l3.pt
    sha256 5aca3716156b190ece14780bc32316e23423ab0c4f9525ba50b4730f108ee8b9
  runtime/hidden_l4.pt
    sha256 0308be3197bfe1921215c2082146946a625350896949f63572b52145afe400a4
```

设备执行和原始 DFX 已完整成功；outer runner 的 heredoc 管道错误发生在
容器结束后的 admission shell 后处理。上述 admission 从同一原始 OUT
只读解析并离线重放，没有重新运行设备。

### 6.4 E5→E6 边界

完整 `E5→E6` 仍为 `n/a`：当前报告没有把 routed combine、shared branch、
TP AllReduce 与 global completion fence 绑定为统一 semantic endpoint。

只读 routed combine partial：

| partial `combine_scatter→combine_reduce` | L3 | L4 |
|---|---:|---:|
| median（8 ranks） | `55.63 us` | `62.54 us` |
| max | `155.66 us` | `158.98 us` |

最大 partial 出现在 route-empty ranks，其 `combine_wait` 为约
`137.62–146.72 us`；本地 scatter/reduce 本体仅数微秒到十余微秒。
所以长尾来自 remote producer completion wait，不是本地 unpermute/reduce
算术。该 partial 不能与 vLLM 完整 `E5→E6 = 57.50 us` 直接作差，也不能支持
用 local `MoeTokenUnpermute` 替换 PyPTO 的 EP8 return/completion 协议。

## 7. `stepfun/develop` 与最终准入

五仓同步 verdict `pass=true`。其中前四仓是 verified no-op；pypto-lib 使用
exact lease 从 `bf3ff440` fast-forward 到 `fe641929`，并再次读取远端 ref
确认精确一致：

```text
git-sync-r10-20260825-144155/
  verdict.json
    sha256 5102b7f6b8a28af1908becdaccb69d20abb0e1c5d6d29e939ef46fbc852fe526

pypto-lib refs/heads/stepfun/develop:
  bf3ff4400082f74b35fbdb5b3e0f5f4bf51ce373
  -> fe641929dbf959d887ad111f3bd7cac0b73fa34b
```

最终合同：

```text
r10-release-admission-20260825-150350/
  release_contract.json
    schema step3p5.r10-release-admission.v2
    pass   true
    checks 71/71
    sha256 bcdd0b11d346e450dca49b8434544de5566b7fc0ad1a38c715815a41958dafca
```

## 8. 当前判定

```text
BUILD / AUDIT / REGISTRY / COMPILE / LIVENESS / N=128 / H4 STRICT PARITY = PASS
ITL LONG / CURVE / ADMISSION = PASS
SIX-BS CORRECTNESS / OUTER HIDDEN / 8-RANK DFX = PASS
MATCHED IMMUTABLE A/B/A / DEVELOP SYNC / FINAL CONTRACT = PASS
FINAL CONTRACT = 71/71 PASS, schema v2
```

所以当前准确称谓是：

> **r10 immutable image 已 release-admitted。**

当前 release-admitted manifest 为 `sha256:8510f30e…e9907b`，性能合同仍须
显式使用 `PYPTO_H4_RESIDENT=all`。完整 `E5→E6` semantic endpoint 继续为
`n/a`；六档 campaign 只证明 correctness，BS8/BS16 的 warmup1/iters1
单样本回退仍保留为后续多 BS 性能采样 caveat。
