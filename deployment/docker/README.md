# vllm-pypto 可复现镜像

基于 0162 验证过的环境 + `stepfun/develop` 分支,做成一个自包含、可复现的
vllm + pypto 集成镜像。

## 内容

- **base**: `hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938`
  (自带 CANN 8.5.1 + CANN 9.0.0-beta.1 + vLLM 0.19.0 + vllm-ascend + python3.11.14)
- 本镜像在其上:
  1. **删 CANN 8.5.1**,只留 `cann-9.0.0-beta.1`(避免 8.5 干扰;`/usr/local/Ascend/cann` 指 beta.1)
  2. clone pypto 栈到 `/workspace`(stepfun/develop pins):`pypto 9ec303f6` / `pypto-lib(stepfun/develop) 4c48215b` / `pto-isa ecb6c303` / `PTOAS 72ada0a1` / simpler submodule `c7fdc574`
  3. `ptoas-bin v0.45`
  4. 编译 `pypto` + `runtime`(`build_runtimes --platforms a2a3`,`cmake==3.31.6`)
  5. **vLLM Track-B 补丁**:`step3p5.py`(tail-only 主网 + `PyPtoMetadataOnlyStep3p5DecoderLayer`)+ `step3p5_mtp.py`(MTP-proposer 挂点 + MTP3 `hf_overrides` boot fix),来自 gitlab `sys/stepcast/vllm:csy/pypto-tail-mtp-integration`
  6. env(CANN beta.1 / PTO_ISA_ROOT / PTOAS / PYTHONPATH / PTO2_RING_*)写进 `/etc/profile.d/pypto-env.sh`

> 注:`--mount=type=secret` 用 BuildKit 内置 frontend(不写 `# syntax=` 以免从 docker.io 拉;
> 内网只有 hub.i.basemind.com)。

## Build

```bash
GH=/data/chensiyu/secrets/github.env \
GL=/data/chensiyu/secrets/gitlab.env \
IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260723 \
bash build.sh
```

`GH`/`GL` 是含 PAT 的文件,以 `docker buildx --secret` 传入,**不落镜像层**。

## Run（含 `/workspace` 挂载,统一管理）

镜像已把仓库 bake 在 `/workspace`(可独立运行);做**统一管理**时把宿主仓库挂载覆盖:

```bash
docker run -it --rm \
  --device /dev/davinci0 ... --device /dev/davinci15 \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v <宿主>/workspace:/workspace \            # 挂载宿主仓库 -> 统一管理(可选)
  -v <宿主>/ckpt:/mnt/hw910test/models/... \  # checkpoint
  hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260723
```

> 挂载 `/workspace` 时,宿主仓库需已编译(`pypto_core*.so` + runtime `.so`),且与镜像
> 的 python3.11.14 + CANN beta.1 ABI 一致(宿主用同环境编译即可)。不挂载则用镜像内 bake 的。

## 验证(0162)

登录 shell 后 env 已就绪(`/etc/profile.d/pypto-env.sh`)。

- 主网整网精度(8-step multi-decode):`cd /workspace/vllm-pypto && python -m tests.step3p5.harnesses._stage_main_hidden_only --device 8,9,10,11,12,13,14,15 --out /tmp/main-hidden --ckpt <ckpt> --steps 8`
- live-oracle 逐 token 对齐 A/B:`tests/step3p5/ci/run_live_precision_ab.sh`(见 `tests/step3p5/ci/LIVE_PRECISION_AB.md`;已验证 124/128=96.9%)
- MTP3 vLLM(接受率从 `/metrics` 读):`vllm serve <ckpt> ... --speculative-config '{"method":"step3p5_mtp","num_speculative_tokens":3,"enable_multi_layers_mtp":true}' --enforce-eager`

## Pin 依据

见 [`../version-matrix.md`](../version-matrix.md) 与 [`../../STATUS.md`](../../STATUS.md)
的「两条线（项目结构）」+ 集成现状快照。
