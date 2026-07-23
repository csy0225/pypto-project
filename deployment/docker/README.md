# vllm-pypto 可复现镜像 — 构建 / 部署 / 验证

基于 0162 验证过的环境 + `stepfun/develop` 分支,做成一个自包含、可复现的
vllm + pypto 集成镜像。**构建于 devbox(有 docker),部署验证在 0162(NPU 机, 只有
containerd/nerdctl)。**

---

## 1. 内容与 pin

- **base**: `hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938`
  (自带 CANN 8.5.1 + CANN 9.0.0-beta.1 + vLLM 0.19.0 + vllm-ascend + python3.11.14)
- 本镜像在其上:
  1. **删 CANN 8.5.1**,只留 `cann-9.0.0-beta.1`;并修好 base 把 8.5.1 设成默认后留下的悬空引用
     (ENTRYPOINT / `/etc/profile` / ENV 里 hardcode 的 `cann-8.5.1` → `beta.1`,见 §7)
  2. clone pypto 栈到 `/workspace`,切到验证过的可编译 pin(下表)
  3. `ptoas-bin v0.45`(含 `$PTOAS_ROOT/ptoas` 顶层符号链接 → `bin/ptoas`,codegen 需要)
  4. 编译 `pypto` + `runtime`(`build_runtimes --platforms a2a3`)
  5. **vLLM Track-B 补丁**:`step3p5.py`(tail-only 主网 + `PyPtoMetadataOnlyStep3p5DecoderLayer`)+
     `step3p5_mtp.py`(MTP-proposer 挂点 + MTP3 `hf_overrides` boot fix),来自 gitlab
     `sys/stepcast/vllm:csy/pypto-tail-mtp-integration`
  6. env(CANN beta.1 / PTO_ISA_ROOT / PTOAS / PYTHONPATH / PTO2_RING_*)写进 `/etc/profile.d/pypto-env.sh`
  7. 冒烟脚本 bake 在 `/workspace/pypto-smoke.sh`

| 仓库 | pin | 说明 |
|------|-----|------|
| pypto | `8af501fc` | = stepfun/develop `9ec303f6` + runtime submodule gitlink 回退到 `36957c6b` |
| pypto-lib | `4c48215b` | stepfun/develop |
| pto-isa | `ecb6c303` | stepfun/develop(本会话 FF-push 对齐) |
| PTOAS(src) | `72ada0a1` | stepfun/develop(本会话 FF-push 对齐) |
| simpler(pypto/runtime submodule) | `36957c6b` | **可编译**版;develop tip `c7fdc574` 的 Phase-24 import_ipc 半成品编译不过(见 §7),已回退 |
| ptoas-bin | `v0.45` | 二进制 |

> **镜像 tag**: `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260723`

---

## 2. 构建(devbox)

```bash
cd deployment/docker
# 每次构建的 pins+tag 在一个 spec 文件里(builds/<tag>.env),配方共用单一 Dockerfile。
GH=/data/chensiyu/secrets/github.env GL=/data/chensiyu/secrets/gitlab.env \
  bash build.sh builds/stepfun-develop-20260723.env
docker push hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260723
```

- `build.sh <spec>` 读 spec 里的 pins,以 `--build-arg` 传进 Dockerfile,`-t` 用 `IMAGE_TAG`。
  不带参数默认用最新 spec。**加新 build 见 [§9 组织方式](#9-组织方式--加新-build)**。

- `GH`/`GL` 是含 PAT 的文件,以 BuildKit `--secret` 传入,**不落镜像层**。
- build.sh 做了三件网络相关的事(devbox 内网特性):
  1. `DOCKER_BUILDKIT=1 docker build --network=host`(走宿主路由到代理);
  2. 从官方入口 `deploy.i.shaipower.com/httpproxy` 取代理并以 `--build-arg` 传入
     (github clone/release 经 `proxy.i.shaipower.com:3128`;内网 pip 镜像/gitlab/hub 直连不走代理);
  3. `ptoas-bin` 从 0162 验证过的二进制打进 build context(fork 无 release asset)。
- 编译限并行 `CMAKE_BUILD_PARALLEL_LEVEL=2 / MAX_JOBS=2`(devbox dockerd 在 memcg 下, 17GB/5 核,
  全并行编 pypto 会 OOM 打挂 dockerd)。

---

## 3. 部署到 0162(containerd / nerdctl)

0162 **没有 docker**,用 containerd 自带的 `nerdctl`(路径 `/mnt/persist/k8s-install/containerd/bin/`)。
0-7 卡通常被 vanilla vLLM oracle(8000)占用,pypto 用 **8-15 卡**。

```bash
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260723
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

**期望输出**(2026-07-23 0162 实测):

```
[smoke] ptoas   : ptoas 0.45
[smoke] pypto   : 0.1.0
[smoke] simpler : OK
[smoke] runtime : /workspace/pypto/runtime/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
[smoke] vllm-pypto CI dir: OK
[smoke] PASS
```

---

## 5. 整网精度验证(canonical, 8 卡)

权威 runner:`tests/step3p5/ci/run_whole_network_ci.py`(preflight → Main 45 层 hidden-only
8-step → MTP45/46/47 → 清理)。用 §3 的 8 卡 run 命令,把最后一行换成:

```bash
"$IMG" bash -lc "cd /workspace/vllm-pypto && \
  python -m tests.step3p5.ci.run_whole_network_ci \
    --ckpt $CKPT --devices 8,9,10,11,12,13,14,15 --out /tmp/n1_ci"
```

> 要留存 artifact/日志时,加 `-v <宿主目录>:/tmp/n1_ci -v <宿主目录2>:/tmp/n1_ci_artifacts`。

**canonical 金标准**:token `6127` → argmax `303`。

**2026-07-23 0162 实测**(整网 decode 逐步):

| step | input | pypto 输出 | 说明 |
|------|-------|-----------|------|
| 0 | 6127 | **303** | canonical 金标准 ✅ |
| 1 | 303 | 1207 | ✅ |
| 2 | 1207 | **6127** | 与**在跑的 8000 vanilla vLLM** 逐 token 一致(见下) |

`hidden_finite=true`、TP `spread=0.0`(确定性)。**结论:镜像内 pypto 整网 decode 与
vanilla vLLM 逐 token 对齐,精度正常。**

> ⚠ **run_whole_network_ci 会在 step2 报 FAIL**:harness 里 `DEFAULT_ORACLE_TOKENS[2]=19384`
> 是**过时常量**(vanilla 的 #2 "题目")。直接查在跑的 8000 vanilla oracle:
> `curl .../v1/completions -d '{"prompt":[6127,303,1207],"max_tokens":1,"temperature":0}'`
> → 返回 **"北京"(token 6127)**,与 pypto 一致。即 FAIL 是 stale-oracle,不是精度问题。
> 真正的门禁是下面的 live A/B。

**live 逐 token A/B(≥95% 门禁, 替代 stale 常量)**:`tests/step3p5/ci/run_live_precision_ab.sh`
(见 `tests/step3p5/ci/LIVE_PRECISION_AB.md`;bare-metal 已验证 124/128=96.9%)。两段式:
oracle-gen 在 vanilla 容器里跑(pypto `.venv311` 无 transformers),pypto teacher-forced 在本镜像跑。

---

## 6. vLLM serving(Track-B 后端 + MTP3)

镜像已把 vLLM 侧 Track-B 补丁覆盖到 base 的 `/vllm-workspace/vllm`。启动 MTP3 speculative
serving(接受率从 `/metrics` 读):

```bash
sudo $NC run -d --name vllm-pypto-serve --net host --ipc host --privileged \
  --security-opt apparmor=unconfined \
  $DEVS --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v "$CKPT":"$CKPT":ro \
  --shm-size 32g "$IMG" bash -lc "
    vllm serve $CKPT \
      --speculative-config '{\"method\":\"step3p5_mtp\",\"num_speculative_tokens\":3,\"enable_multi_layers_mtp\":true}' \
      --enforce-eager --port 8001"
# 接受率: curl -s http://127.0.0.1:8001/metrics | grep -i spec_decode
```

> `--enforce-eager` 必需(pypto kernel 与 vLLM aclgraph 互斥)。`--speculative-config` 用连字符。
> 注:pypto 作为 live 主网 backend(KV bridge + 动态 batch 映射)仍是 Phase 20 在建项;
> 当前镜像 serving 走 vanilla + MTP3 路径,pypto 整网 decode 走 §5 的 offline canonical。

---

## 6.5 decode ITL 性能(64k context)

harness 的 perf-only 模式(pypto-lib ≥ `7cb2a6b3`):固定 context 长度 pin metadata 到
`seq_len=L`,计每步 `holder.run()` 耗时。attention 计算/带宽与 KV *内容* 无关,故无需
prefill 即可测大 context 的稳态 ITL;**KV-IPC 全程开启**(holder `kv_ipc=True`,`block_table`
在 `--num-blocks 512` 下可寻址到 512 块 → 64k),所以 attention 真实遍历 64k KV。

```bash
# 已发布镜像 stepfun-develop-20260723 bake 的是 pypto-lib 4c48215b(无此 harness);
# 用 -v 挂载新 harness, 或用带 7cb2a6b3 的新 build。
sudo $NC run --rm --net host --ipc host --privileged --security-opt apparmor=unconfined \
  $DEVS --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v "$CKPT":"$CKPT":ro \
  -v /data/chensiyu/itl_out:/tmp/itl --shm-size 32g \
  "$IMG" bash -lc "cd /workspace/vllm-pypto && python -m tests.step3p5.harnesses._stage_main_hidden_only \
    --device 8,9,10,11,12,13,14,15 --ckpt $CKPT --out /tmp/itl --num-blocks 512 \
    --itl-context-lens 1024,4096,16384,32768,65536 --itl-iters 20 --itl-warmup 3"
```

**2026-07-23 0162 实测**(整网 45 层 hidden-only decode,W8A8,TP=8,active batch=1):

| context | ITL mean (ms) | p50 | min | max |
|--------:|:-:|:-:|:-:|:-:|
| 1024 | 635.3 | 644.4 | 617.4 | 648.6 |
| 4096 | 639.3 | 645.3 | 617.5 | 652.3 |
| 16384 | 640.0 | 640.3 | 620.7 | 658.2 |
| 32768 | 646.7 | 651.2 | 624.5 | 688.7 |
| **65536** | **654.0** | 658.9 | 632.1 | 677.6 |

**结论:64k decode ITL ≈ 654 ms/step;1k→64k 仅 +19ms,整网 decode 计算受限
(45 层 W8A8 MLP/MoE),非 attention/KV 受限。** 报告 `itl_report.json`。
> 注:这是**未做性能调优**的整网 baseline(Phase 22 才做 tuning),数值仅作当前参考。

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
│   └── stepfun-develop-20260723.env
├── README.md           # 本文档 + 下方「构建登记表」
├── .dockerignore / .gitignore
└── (ptoas-bin.tgz / build_*.log 由 build.sh 生成, gitignored)
```

**加一个新 build**(例:新 pypto commit):

1. `cp builds/stepfun-develop-20260723.env builds/<新tag>.env`,改 `IMAGE_TAG` + 变动的 `*_COMMIT`。
2. `bash build.sh builds/<新tag>.env && docker push hub.i.basemind.com/stepcast/vllm-pypto:<新tag>`。
3. 在下方**构建登记表**加一行(tag / 日期 / pins 摘要 / 验证状态)。
4. Dockerfile **不动**(除非配方本身要改,如新踩坑修复——那属于所有 build 共享的配方演进)。

> 旧 spec 文件保留(可复现历史某次镜像);Dockerfile 单一、不随 build 复制,避免配方漂移。

## 构建登记表

| IMAGE_TAG | 日期 | pypto / pypto-lib / pto-isa / PTOAS / simpler / ptoas-bin | 验证(0162) |
|-----------|------|----------------------------------------------------------|-------------|
| `stepfun-develop-20260723` | 2026-07-23 | `8af501fc` / `4c48215b` / `ecb6c303` / `72ada0a1` / `36957c6b` / `v0.45` | 冒烟 PASS + 整网 decode `6127→303` / step2→`6127`(与 vanilla 逐 token 一致)✅ |

## Pin 依据

见 [`../version-matrix.md`](../version-matrix.md) 与 [`../../STATUS.md`](../../STATUS.md)
的 Pin Snapshot(2026-07-23 行)+「两条线(项目结构)」。
