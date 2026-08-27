# 2026-08-26 · replicated-input local-owner MoE r11 镜像发布

> **状态：release-admitted，最终合同 20/20 PASS。**
> r11 以已发布 r10 为 base，仅将 pypto-lib 从 `fe641929` 前进到
> `e6c7d8ec`，落地 replicated-input local-owner MoE；其余四仓 pin
> 与 r10 相同。Registry、fresh pull、raw identity、H4 all/none precision
> 与 parity、64K ITL、immutable r10/r11/r10 A/B/A 均已闭环。

## 1. 镜像与源码身份

| 字段 | 值 |
|---|---|
| 机器 | `gpu-a910x-0162.host.platform.shaipower.com` |
| Tag | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-upgrade-20260826-r11` |
| Digest ref | `hub.i.basemind.com/stepcast/vllm-pypto@sha256:401ead7da4f957f6532e380fa1a138eda733fe1dc04b40eabc67d79d62a67b12` |
| Manifest | `sha256:401ead7da4f957f6532e380fa1a138eda733fe1dc04b40eabc67d79d62a67b12` |
| Config | `sha256:35c42510a64ce3e1c8e899e15c36ab8b534d091ea03a085ec663f18df8706876` |
| Base | r10 manifest `sha256:8510f30e1f2a2f2edcaa834c831165b349a4aca1212b655ca2a02ed6b3e9907b` |
| pypto | `519b588a7a6461cac0e443e853accf29479c1d15` |
| pypto-lib | `e6c7d8ec34a05c3051ccf0dd169639f40f041a57` |
| pypto-lib tree | `22a40673ababd069fa79a5890f7f95722a00b527` |
| pto-isa | `cd4a3d3f7a1a27fcfe536f617e9bca3008929664` |
| PTOAS source / binary | `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` / `v0.57` |
| simpler | `85a82c454074c069315ed6485033c3c2b136e562` |
| vLLM patch | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` |

r11 构建 spec 明确写明：

```text
The published r10 image is the reproducible base; only pypto-lib advances.
```

因此 r10 到 r11 的源码变量只有：

```text
pypto-lib:
  fe641929dbf959d887ad111f3bd7cac0b73fa34b
  -> e6c7d8ec34a05c3051ccf0dd169639f40f041a57
```

最终合同封存的 pypto-lib 源码身份：

```text
commit                   e6c7d8ec34a05c3051ccf0dd169639f40f041a57
parent                   2be5dcad8077f9090cc31dbc7377b66c9a15a067
tree                     22a40673ababd069fa79a5890f7f95722a00b527
bundle sha256            49dd2b0c17190b72901b59adb85aaf50f039e2189bcc04c46f1220c9fe75b5ee
frozen manifest entries  495
frozen manifest sha256   c19c6fe7e3deb4f81c6c4d99ab7638d75512b82fa48c1868adb4569ec0491c92
decode_fwd.py sha256     91d677a874a5a9a4ac394e8a0e1d5e44fe7eccd87fa83dc3715a7ae20d392e41
```

## 2. 可复现构建

0162 证据根：

```text
/mnt/persist/chensiyu/workspace/moe-fusion-release-20260826/
```

关键构建工件：

| 工件 | SHA256 |
|---|---|
| `image-build/builds/stepfun-upgrade-20260826-r11.env` | `9c272afe17545277f475a12859d86cb33633ef9a5df140bbf51c0a0ff5218c62` |
| `image-build/src-pins.shas` | `c6db303ef61ad3c5309517090a1f9a9ae8d7bce2da4b5249bf015a19dce2a79e` |
| `image-build-r11-20260826-094652/run_contract.txt` | `17b74a225046f596e62e4b63a4151074e137ed2f05d48186fc70e6231c84ee5d` |
| `image-build-r11-20260826-094652/evidence.sha256` | `a61ed6ac5e378f3daa53724bbd0cf40f2676d26af65bb562e019b0622479ac55` |

构建、manifest/config identity、source clean、source commit/tree、
frozen source manifest 和 image audit 均由最终合同逐项校验为 `true`。

## 3. Immutable r10/r11/r10 A/B/A

证据目录：

```text
/mnt/persist/chensiyu/workspace/moe-fusion-release-20260826/
  itl/image-aba-r10-r11-r10-local-20260826-105857/
```

运行合同：

```text
arm order       A1 -> B -> A2
devices         0,1,2,3,4,5,6,7
protected       8,9,10,11,12,13,14,15
active batch    1
context         65536
num_blocks      512
warmup / iters  10 / 100
H4              all
source overlay  false
runtime overlay false
fresh container/build directory per arm
```

Arm 定义：

| Arm | 镜像 | pypto-lib | Config |
|---|---|---|---|
| A1 | r10 digest `sha256:8510f30e…e9907b` | `fe641929` | `sha256:38ebba41…657b5f` |
| B | r11 manifest `sha256:401ead7d…a67b12` | `e6c7d8ec` | `sha256:35c42510…06876` |
| A2 | r10 digest `sha256:8510f30e…e9907b` | `fe641929` | `sha256:38ebba41…657b5f` |

ITL 结果：

| Arm | p50 | mean | p99 |
|---|---:|---:|---:|
| A1 r10 | `21.751 ms` | `21.884 ms` | `26.052 ms` |
| B r11 | `21.745 ms` | `21.938 ms` | `32.285 ms` |
| A2 r10 | `21.752 ms` | `21.877 ms` | `26.828 ms` |

Baseline p50 midpoint 为 `21.7515 ms`，r11 相对 midpoint：

```text
-0.0065 ms / -0.0299%
```

这个差值远小于可解释为显著收益的量级。准确结论是：

> **r11 相对 r10 性能中性，未观察到 p50 回退。**

p99 高于两个 r10 baseline arm，因此不能声明 tail latency 改善。

A1/A2 使用旧 r10 oracle，hidden SHA 均为
`567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e`、
tail token `14371` exact。r11 的 local-owner 语义对应新的 accepted output：
hidden SHA 为
`ee8ae6b4b3083112d397e5e91cc63fb0e2edfb705eb7a535aceb232f1a7db96a`、
tail token `43640`。因此该 A/B/A 的 correctness 条件是：

- A1/A2 baseline identity 与旧 oracle exact；
- B 通过独立 r11 128-step precision gate 和 H4 parity gate；
- 不把跨版本 hidden 不同误写成 correctness regression。

关键 SHA：

```text
run_contract.json  7bc24697105e2dadc00b3aac702f606930de6351e8d7b656df50268e61b2cdd3
aba_admission.json 611aa7c2ce4243615cec521d4ffd826e2b1937c2835336fc7ceb90558d6fcd41
artifacts.sha256   051401c0d91af95014391f9142f01bc09ead5a7941c069738b96aefc290d2c74
```

## 4. Precision 与 H4 parity

r11 digest-only precision：

| H4 模式 | aligned | mismatch steps | finite | TP spread | 结果 |
|---|---:|---|---|---:|---|
| `all` | `126/128 = 98.4375%` | `[20, 69]` | true | `0.0` | PASS |
| `none` | `126/128 = 98.4375%` | `[20, 69]` | true | `0.0` | PASS |

严格 parity 进一步确认：

- H4 all/none output tokens `128/128` exact；
- active hidden `128/128` pair byte-exact；
- row0 hidden `128/128` pair byte-exact；
- 所有 active rows 存在、所有 hidden finite；
- TP spread max 为 `0.0`。

证据：

```text
precision-r11-image-h4-all-20260826-103901/
  evidence.sha256
  sha256 c61c45a8f15b125586b91a80bf7700fc2874d60a576ce5896531339fca169a0e

precision-r11-image-h4-none-20260826-104714/
  evidence.sha256
  sha256 2c971015d4dc4ece42842f8b948c6951629599f64ed83a8831d1b89ec57cdaaa
```

## 5. ITL

四点 context curve：

| Context | Iters | p50 | mean | p99 |
|---:|---:|---:|---:|---:|
| 1K | 20 | `20.680 ms` | `20.756 ms` | `21.747 ms` |
| 8K | 20 | `21.205 ms` | `21.370 ms` | `23.267 ms` |
| 32K | 20 | `21.471 ms` | `21.480 ms` | `21.591 ms` |
| 64K | 20 | `21.390 ms` | `21.891 ms` | `28.084 ms` |

64K / 1000-iter：

```text
min   21.332 ms
mean  22.262 ms
p50   21.477 ms
p99   35.882 ms
max   55.427 ms
```

这些数字绑定 `PYPTO_H4_RESIDENT=all`。镜像 Config 未把该值作为可替代
launcher 合同的默认性能承诺。

证据：

```text
itl/itl-r11-local-20260826-111531/evidence.sha256
sha256 86de2999aa0596edfc586dfe6322b706f5f7e79d7cde83e2d55d2f66ea7cb915
```

## 6. Registry 发布

证据目录：

```text
/mnt/persist/chensiyu/workspace/moe-fusion-release-20260826/
  publication/publication-r11-20260826-112906/
```

发布结果：

- push rc=0；
- 隔离 namespace `pypto-r11-fresh-20260826-112906` fresh pull PASS；
- published manifest/config 与期望 identity exact；
- registry raw manifest/config PASS；
- image audit、smoke、source-bake audit PASS。

关键 SHA：

```text
run_contract.txt  9cb8875cf3948dedeb7380e0d376074b8d36589610511570c25c137cf5811c04
evidence.sha256   f4852a771129f2e2626edaaf8a1987515a17bf8fc4c5c4dd946048d9c700d0d1
```

## 7. 最终 release contract

```text
/mnt/persist/chensiyu/workspace/moe-fusion-release-20260826/
  release-admission-r11-20260826-113923/release_contract.json
```

```text
schema  step3p5.r11-release-admission.v1
pass    true
checks  20/20
sha256  570bb04ef761e66fa12fb246f3482973294fe282688d967c76e119fcda740af7
```

20 项检查覆盖 build、source identity、publication、fresh pull、raw registry、
H4 all/none precision、H4 parity、ITL 和 immutable A/B/A。

准确称谓是：

> **r11 immutable image 已 release-admitted；相对 r10 的性能结论为中性。**
