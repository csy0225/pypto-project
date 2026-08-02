# vllm-pypto 可复现镜像 — 构建 / 部署 / 验证

基于 0162 验证过的 0724 环境 + immutable repository pins,做成一个自包含、可复现的
vllm + pypto 集成镜像。**构建于 devbox(有 docker),部署验证在 0162(NPU 机, 只有
containerd/nerdctl)。**

---

## 0. 当前镜像状态（2026-08-02）

最新构建是 **clean canonical candidate，不是正式 release**：

```text
tag:
  hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260802-attn-final-canonical

manifest:
  sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d

config/image ID:
  sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea
```

pins：

```text
pypto      defa97c526fec7e8f032dbbfcc39c820add02bf7
pypto-lib  76d96bdbeac280f12ecf626b1bbd722b9278719e
pto-isa    ecb6c303f797749f811a494742c3c08156aacabb
PTOAS      fc8c6caee561914b4fb991dfc8427bb63194269e
simpler    e2efebcbd190302609c0775d2984f409f5f42c76
ptoas-bin  v0.50
vLLM       1b3e538c35999e62b6d24e0651b3a85b7d16c826
```

镜像内容 audit、smoke、64K ITL、DFX 均 PASS；64K p50 为 `50.563 ms`。但同一
fresh oracle 的 N=128 三轮均为 `121/128=94.53125%`，低于 `>=95%` raw gate，
所以当前状态是 **release blocked**。部署/回归时必须按 manifest 核对，不能只看 tag，
也不能引用 v2 的历史 `123/128` 作为该 clean 镜像的结果。

本轮 immutable 验证只挂载 driver(ro)、checkpoint(ro)、output(rw)，没有宿主源码挂载。
0162 只使用 cards `0–7`，未操作 cards `8–15` 及 PID `2045390–2045397`。

拉取 candidate（仅用于继续定位/复核，不作为 production rollout）：

```bash
IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260802-attn-final-canonical
sudo "$NC" pull "$IMG"
# 拉取后必须核对 manifest sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d
```

> 本文后续 2026-07-29 镜像段落保留为历史 runbook。当前 pin、tag 和发布判定以本节、
> [`../version-matrix.md`](../version-matrix.md) 与
> [`../../benchmark/2026-08-02-step3p5-attention-final.md`](../../benchmark/2026-08-02-step3p5-attention-final.md)
> 为准。

## 1. 历史 2026-07-29 镜像内容与 pin

> 本节只描述 `stepfun-develop-20260729-allreduce-push` 这一历史组合，保留作
> 复现/回退 runbook；它不是 2026-08-02 clean canonical candidate 的当前 pin。
> 当前 candidate 的 pin、digest 和发布判定以 §0、[`../version-matrix.md`](../version-matrix.md)
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

## 2. 构建(devbox；通用流程，示例以当前 candidate 为准)

> 下面的构建流程适用于所有 immutable spec。旧的 2026-07-29 baseline 仍保留在
> 历史 spec 中；若目标是复现当前 attention candidate，应使用
> `builds/stepfun-develop-20260802-attn-final-canonical.env`。该 candidate 的
> raw precision gate 尚未通过，重建/复核不等于正式发布。

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
bash build.sh builds/stepfun-develop-20260802-attn-final-canonical.env
# ⚠ 先做 §2(a) 的本地核对，PASS 之后才 push
# 可以推 candidate tag 供定位/复核，但不得 retag 或 rollout 为 production release。
docker push hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260802-attn-final-canonical
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
> `attn-final-canonical` 的 N=128 gate 结果。当前 candidate 的精度 blocker、
> 64K ITL 和 DFX 证据见 §0 及
> [`../../benchmark/2026-08-02-step3p5-attention-final.md`](../../benchmark/2026-08-02-step3p5-attention-final.md)。

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
4. Dockerfile **不动**(除非配方本身要改,如新踩坑修复——那属于所有 build 共享的配方演进)。

> 旧 spec 文件保留(可复现历史某次镜像);Dockerfile 单一、不随 build 复制,避免配方漂移。

## 构建登记表

| IMAGE_TAG | 日期 | pypto / pypto-lib / pto-isa / PTOAS / simpler / ptoas-bin | 验证(0162) |
|-----------|------|----------------------------------------------------------|-------------|
| `stepfun-develop-20260723` | 2026-07-23 | `8af501fc` / `4c48215b` / `ecb6c303` / `72ada0a1` / `36957c6b` / `v0.45` | 冒烟 PASS + 整网 decode `6127→303` / step2→`6127`(与 vanilla 逐 token 一致)✅ |
| `stepfun-develop-20260724` | 2026-07-24 | `ca21ab5f` / `fd26b1be` / `ecb6c303` / `fc8c6cae` / `216e7632` / `v0.50` | 合并 origin/main + IPC 权重 interior 指针 provenance 修复（解 `submit_next_level child_memory` 卡点）。冒烟 PASS(ptoas 0.50) + 整网 8 步 decode `6127→303→1207→6127`(与 live vanilla 逐 token 一致)✅ |
| `stepfun-develop-20260729-allreduce-push` | 2026-07-29 | `6933b1aa` / `cfbdcce8` / `ecb6c303` / `fc8c6cae` / `8459d60f` / `v0.50` | **2026-07-29 正式发布基线**。registry digest `sha256:7924925f4b281…`（config `sha256:5402e07ba0d19…`）；PERF-C4 TP all-reduce → reduce-scatter + push all-gather，simpler span-aware provenance 入库。0162 immutable-image 回归：audit 5 pin 一致 + `IMAGE_GIT_CREDENTIAL_AUDIT` / `IMAGE_WORKTREE_CLEAN_AUDIT` / `CANONICAL_ONLY_AUDIT` / `ALLREDUCE_PUSH_PRESENT` 全 PASS；smoke PASS；整网 CI `rc=0` 198.3 s，6 项 check 全 true（`tokens_exact` / `eight_steps` / `result_clean` / `pypto_hidden_only` / `step0_hidden_saved` / `process_rc_zero`），token `303,1207,19384,872,428,6127,4231,2636`；`hidden_tp_spread` 在 ci/main + rep1/rep2/rep3 共 **4×8 = 32 步全 `0.0`**；ITL p50 `65.942 ms`(ctx=1024) / `66.455 ms`(ctx=4096)（`--num-blocks 32`），权威 64k 工作点 `--num-blocks 512` 为 p50 `83.349 ms`，DFX 拆解见 benchmark/2026-07-29 |
| `stepfun-develop-20260726-step3p5-only` | 2026-07-26 | `ca21ab5f` / `53eb7212` / `ecb6c303` / `fc8c6cae` / `216e7632` / `v0.50` | registry digest `sha256:99b2b971…`（config `sha256:d2964610…`）；0162 credential/symbol/ldd audit、smoke、unit `136 passed/4 skipped`、contract `15 passed`；唯一 program=`whole_decode_step3p5`；N=256 raw `240/256=93.75%`，与清理前 canonical token/hidden `256/256` exact、`max_abs_diff=0`、TP spread `0`，step127/128/255 PASS |
| `stepfun-develop-20260729-perf-h1` | 2026-07-29 | `1f704616` / `4513007d` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **PERF-H1 device-memset 优化镜像**（perf 上取代 C4 发布，正确性等价）。registry digest `sha256:b4e8c8a457a5…`。在 C4 上前进 pypto→`1f704616`(gitlink→simpler `e2efebcb`)、pypto-lib→`4513007d`(=`cfbdcce8` + ITL `--active-batch` + MTP CI oracle-dir 可配置化)。0162 回归：smoke PASS；整网 CI `ok=true`（Main token `303,1207,19384,872,428,6127,4231,2636` exact + MTP single/batch16 `6178,410,303` exact，`hidden_tp_spread=0`）；N=256 H1 vs C4 **token 256/256 exact**（step127/128/255 含），全步 finite（raw-hidden run-to-run 抖动=C4 push all-reduce 归约顺序，非 H1 回归，H1a-vs-H1b 复跑证实）。**ITL p50(`--num-blocks 512`)：1024 `50.9` / 8192 `52.0` / 32768 `58.0` / 65536 `64.1` ms，较 C4 同工作点降 23–27%**。PMU/scope 与 C4 逐项一致（cube_int8 46.35%，ring heap 79.9%，dropped=0）。benchmark: [`../../benchmark/2026-07-29-perf-h1-image-itl-dfx.md`](../../benchmark/2026-07-29-perf-h1-image-itl-dfx.md)。⚠ MTP CI oracle-wiring 修复 `0f3650c7`(test-only) 为 mount 验证、未 rebuild 烤进本镜像 |
| `stepfun-develop-20260802-attn-final` | 2026-08-02 | `1f704616` / `76d96bdb` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **v1 失败证据**：缺 `defa97c5` 动态 SPMD codegen 修复，immutable-image compile 报 launch-bound 变量未声明；非 candidate |
| `stepfun-develop-20260802-attn-final-v2` | 2026-08-02 | `defa97c5` / `76d96bdb` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **v2 非 canonical**：代码可运行，但 image config 仍含旧 CANN 8.5.1 字符串；历史 `123/128` 不得借给 clean 镜像 |
| `stepfun-develop-20260802-attn-final-canonical` | 2026-08-02 | `defa97c5` / `76d96bdb` / `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50` | **clean canonical candidate / release blocked**。manifest `sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d`，config `sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea`；audit/smoke/64K ITL/DFX PASS，p50 `50.563 ms`；fresh-oracle 三轮均 `121/128=94.53125%`，低于 raw gate |

## Pin 依据

见 [`../version-matrix.md`](../version-matrix.md) 与 [`../../STATUS.md`](../../STATUS.md)
的 Pin Snapshot(2026-07-23 行)+「两条线(项目结构)」。
