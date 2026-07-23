---
name: pypto-image-verify
description: >
  用户拿到 vllm-pypto 可复现镜像后,在一台 Ascend 910B NPU 机上快速验证"环境是否可用"的
  分步 runbook。当用户问"镜像能不能用 / 拿到镜像怎么验证 / 验证一下这个镜像 / 部署后怎么
  确认 pypto 跑得通"时使用。覆盖:host 三件套核验、nerdctl 拉取、bake 冒烟脚本、8 卡整网
  canonical 精度、结果判读(含 stale-oracle 陷阱)与"可用/不可用"反馈模板。区别于
  pypto-runtime-install(那是从零裸机装 5 仓;本 skill 是拿现成镜像快速验证)。
---

# vllm-pypto 镜像快速验证 runbook(Ascend 910B)

> 目标:**用户拿到 `hub.i.basemind.com/stepcast/vllm-pypto:<tag>` 镜像 → 在 NPU 机上
> 3 步确认可用**:① 冒烟(镜像+native 库加载)② 整网 decode 精度(canonical)。
> 完整构建/部署文档:[`deployment/docker/README.md`](../../deployment/docker/README.md)。

镜像自包含(pypto 栈 + runtime .so + ptoas + vLLM Track-B 补丁 + CANN beta.1),**不需要**
宿主装 pypto;宿主只需满足 driver/firmware/CANN 硬前提 + 有 containerd/nerdctl。

---

## 用法(给 agent:分步执行,最终反馈"可用/不可用")

- 按 Step 顺序执行,每步先看"检查",达标才继续。
- Step 0 的一次性/权限问题(升 driver 等)**先停下告诉用户**,不要擅自装驱动。
- 最终**反馈模板**:
  ```
  镜像: 可用 ✅ / 不可用 ❌
  卡在: <Step N / 无>
  证据: 冒烟 [smoke] PASS + 整网 step0 6127->303 / step2->6127(与 vanilla 一致)
  待办: <需用户做的:升 driver / 挂 ckpt / 释放卡 …>
  ```

---

## Step 0 · 前提核验(宿主)

```bash
# 1) Phase 16 三件套(缺则多卡必挂)
npu-smi info -t board -i 0 | grep -E "Software|Firmware"   # 期望 25.5.2 / 7.8.0.7.220
ls /usr/local/Ascend/cann/set_env.sh                        # CANN 9.0.0-beta.1(非 GA)
# 2) 容器运行时(0162 无 docker, 用 containerd 自带 nerdctl)
NC=$(command -v nerdctl || echo /mnt/persist/k8s-install/containerd/bin/nerdctl); ls "$NC"
# 3) 空闲卡(0-7 常被 8000 oracle 占, 用 8-15)+ checkpoint
npu-smi info | grep -E "^\| [0-9]"                          # 看 8-15 HBM 占用低
ls -d /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp  # W8A8 ckpt
```

- ❌ driver/firmware 低 → 先按 [`deployment/machine-recovery.md`](../../deployment/machine-recovery.md) 升级(成对升,重启),别继续。
- ❌ CANN 是 GA → 换 beta.1(见 [`deployment/phase16-three-pillars.md`](../../deployment/phase16-three-pillars.md))。
- ❌ 8-15 卡有残留占用 → `npu-smi info -t usages` 确认 <10%;有残留进程先 `pkill`(**禁 `-9`**、**禁 `npu-smi reset`**)。

---

## Step 1 · 拉镜像

```bash
IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260723
sudo $NC pull "$IMG"        # base blob 已在 content store, 只下增量
sudo $NC images | grep vllm-pypto
```

> 拉不动 → 确认宿主能到 `hub.i.basemind.com`(内网);或让用户 `nerdctl login`。

---

## Step 2 · 冒烟(镜像 + native 库在本机可加载)—— 单卡即可

镜像 bake 了 `/workspace/pypto-smoke.sh`:

```bash
sudo $NC run --rm --net host --security-opt apparmor=unconfined \
  --device /dev/davinci8 --device /dev/davinci_manager \
  --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  "$IMG" bash -lc 'bash /workspace/pypto-smoke.sh'
```

**期望**(全绿即 Step 2 通过):

```
[smoke] ptoas   : ptoas 0.45
[smoke] pypto   : 0.1.0
[smoke] simpler : OK
[smoke] runtime : .../a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
[smoke] vllm-pypto CI dir: OK
[smoke] PASS
```

- ❌ 容器 `rc=1` 起不来、`cann-8.5.1 ... No such file` → 用的是旧镜像(未修 8.5.1 残留),换新 tag。
- ❌ `pypto`/`simpler` import 失败 → native .so 与宿主 driver ABI 不匹配(核对 Step 0 CANN/driver)。

---

## Step 3 · 整网 decode 精度(canonical, 8 卡)

```bash
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
DEVS=""; for i in 8 9 10 11 12 13 14 15; do DEVS="$DEVS --device /dev/davinci$i"; done
mkdir -p /data/chensiyu/ci_out
sudo $NC run --rm --net host --ipc host --privileged --security-opt apparmor=unconfined \
  $DEVS --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro -v /data/chensiyu/ci_out:/tmp/n1_ci --shm-size 32g \
  "$IMG" bash -lc "cd /workspace/vllm-pypto && \
    python -m tests.step3p5.ci.run_whole_network_ci \
      --ckpt $CKPT --devices 8,9,10,11,12,13,14,15 --out /tmp/n1_ci"
```

**判读(关键)**——canonical 金标准 = token `6127` → argmax `303`:

- step0 `6127→303`、step1 `303→1207`、step2 `1207→6127` = **pypto 与 vanilla 逐 token 一致,精度正常 ✅**。
- ⚠ **runner 可能在 step2 报 `SINGLE_CHIP_HIDDEN_CI=FAIL`**:harness 里
  `DEFAULT_ORACLE_TOKENS[2]=19384` 是**过时常量**(vanilla 的 #2 "题目"),**不是精度问题**。
  独立确认:查在跑的 8000 vanilla oracle
  `curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"step3.5-flash","prompt":[6127,303,1207],"max_tokens":1,"temperature":0}'`
  → 返回 **"北京"(token 6127)**,与 pypto 一致 → 判"可用"。
- 权威 ≥95% 门禁走 live A/B:`tests/step3p5/ci/run_live_precision_ab.sh`(见 LIVE_PRECISION_AB.md)。

> 失败时读留存的 exporter 日志:`/data/chensiyu/ci_out/../..` 或加
> `-v <宿主>:/tmp/n1_ci_artifacts` 后看 `logs/main_hidden_8step.log`。

---

## 踩坑速查(nerdctl 起容器)

| 现象 | 根因 | 修法 |
|------|------|------|
| `apparmor_parser resolves to executable in current directory` | nerdctl apparmor 探测 quirk | `--security-opt apparmor=unconfined` |
| `needs CNI plugin "bridge"` | 0162 无 CNI bridge | `--net host` |
| 整网 exporter `exited before readiness` / 跨卡 IPC 失败 | 缺 IPC/设备权限 | `--privileged --ipc host --shm-size 32g` |
| `ptoas does not exist`(codegen) | 旧镜像缺 `$PTOAS_ROOT/ptoas` 符号链接 | 换新 tag(已修) |
| 容器 `rc=1` + `cann-8.5.1 No such file` | 旧镜像 8.5.1 残留 | 换新 tag(ENTRYPOINT/profile/ENV 已改 beta.1) |
| ptoas/pypto 命令找不到 | 没走登录 shell | 命令用 `bash -lc '...'`(source `/etc/profile.d/pypto-env.sh`) |
| runner step2 FAIL | `DEFAULT_ORACLE_TOKENS` stale(19384) | 非精度问题,见 Step 3 判读 |

## 铁律

1. **宿主 Phase 16 三件套缺一不可**(driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1)。
2. **8 卡用 8-15**(0-7 常被 8000 oracle 占);launch 前确认目标卡空闲。
3. **命令走 `bash -lc`**(登录 shell 才有 pypto env);多卡加 `--privileged --ipc host --shm-size`。
4. **runner step2 FAIL ≠ 精度问题**:是 harness stale oracle,用 8000 vanilla 或 live A/B 佐证。
5. **禁 `-9` 强杀 device 进程 / 禁 `npu-smi reset`**(netboot 机重启锁死 / card poison)。
