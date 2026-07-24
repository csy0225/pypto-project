---
name: pypto-image-verify
description: >
  用户拿到 vllm-pypto 可复现镜像后,在一台 Ascend 910B NPU 机上快速验证"环境是否可用"的
  分步 runbook。当用户问"镜像能不能用 / 拿到镜像怎么验证 / 验证一下这个镜像 / 部署后怎么
  确认 pypto 跑得通"时使用。覆盖:host 三件套核验、nerdctl 拉取、bake 冒烟脚本、8 卡整网
  canonical 精度、结果判读(含 stale-oracle 陷阱)与"可用/不可用"反馈模板。同时覆盖
  "已 exec 进容器内"(无 nerdctl)的快速验证路径。区别于 pypto-runtime-install(那是从零
  裸机装 5 仓;本 skill 是拿现成镜像/容器快速验证)。
---

# vllm-pypto 镜像快速验证 runbook(Ascend 910B)

> 目标:**用户拿到 `hub.i.basemind.com/stepcast/vllm-pypto:<tag>` 镜像 → 在 NPU 机上
> 3 步确认可用**:① 冒烟(镜像+native 库加载)② 整网 decode 精度(canonical)。
> 完整构建/部署文档:[`deployment/docker/README.md`](../../deployment/docker/README.md)。

镜像自包含(pypto 栈 + runtime .so + ptoas + vLLM Track-B 补丁 + CANN beta.1),**不需要**
宿主装 pypto;宿主只需满足 driver/firmware/CANN 硬前提 + 有 containerd/nerdctl。

> **两种执行路径,先识别用哪条**:
> - **宿主场景**(有 nerdctl,镜像在远端 registry):走 Step 0→1→2→3 完整流程。
> - **容器内场景**(`ls /workspace/pypto-smoke.sh` 存在、无 nerdctl、已在 `vllm-pypto` 容器里):
>   跳过 Step 1,直接走下面的[「容器环境内场景」](#容器环境内场景已经-nerdctl-exec--kubectl-exec-进-vllm-pypto-容器)小节。

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

## 容器环境内场景(已经 `nerdctl exec` / `kubectl exec` 进 `vllm-pypto` 容器)

> 触发信号:`ls /workspace/pypto-smoke.sh` 存在、`command -v nerdctl` 无输出、
> `LD_LIBRARY_PATH` 已含 CANN/driver 路径——此时**已经在镜像里**,不要再装/拉 nerdctl。
> 也适用于拿到 k8s pod 直接 `kubectl exec` 进去的情况。

### 与宿主场景的差异(对照执行)

| 环节 | 宿主场景 | 容器内场景 |
|------|----------|------------|
| Step 0 容器运行时检查 | 必查 nerdctl/containerd | **跳过**(已在容器内,`command -v nerdctl` 无所谓) |
| Step 1 拉镜像 | `sudo $NC pull` | **整步跳过**(已在该镜像里) |
| Step 2 冒烟 | `sudo $NC run ... bash -lc 'bash /workspace/pypto-smoke.sh'` | `bash -lc 'bash /workspace/pypto-smoke.sh'`(**裸跑,无 `$NC run` 外壳**) |
| Step 3 整网 CI | `sudo $NC run ... python -m tests.step3p5.ci.run_whole_network_ci ...` | `cd /workspace/vllm-pypto && bash -lc 'python -m tests.step3p5.ci.run_whole_network_ci ...'`(**裸跑,无 `--device` / `-v` 挂载**) |
| CANN 版本探针 | `ls /usr/local/Ascend/cann/set_env.sh` | 同上;但 `version.info` 可能不存在,改读 `/usr/local/Ascend/cann-9.0.0-beta.1/x86_64-linux/ascend_toolkit_install.info` 里的 `version=9.0.0-beta.1` |
| Checkpoint 路径 | `/data/chensiyu/...` | 可能是 `/mnt/hw910test-jfs/models/...`(jfs 挂载),按用户指给的实际路径 |

### Step 0(容器内)· 三件套 + 设备 + ckpt

```bash
# 1) Phase 16 三件套
npu-smi info -t board -i 0 | grep -E "Software|Firmware"   # 25.5.2 / 7.8.0.7.220
ls /usr/local/Ascend/cann-9.0.0-beta.1/x86_64-linux/ascend_toolkit_install.info  # version=9.0.0-beta.1
# 2) 已在容器内(此时 nerdctl 找不到是正常的,不是缺失)
ls /workspace/pypto-smoke.sh && ls /workspace/ptoas-bin/bin/ptoas  # 镜像内 bake 的关键文件
# 3) 容器内可见的空闲卡 + ckpt(路径以容器内挂载为准)
npu-smi info | grep -E "^\| [0-9]"
ls -d /mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp  # 或用户给的实际路径
```

### Step 2(容器内)· 冒烟

```bash
bash -lc 'bash /workspace/pypto-smoke.sh'
```

期望同宿主场景:`[smoke] PASS` 全绿。

> **注意**:smoke 只验 ptoas binary 能打出版本字符串,**不跑真实 codegen**。
> 即使 smoke PASS,若 LD_LIBRARY_PATH 漏 `/workspace/ptoas-bin/lib`,Step 3 一进
> codegen 立即崩 `libMLIRMlirOptMain.so.21.1: cannot open shared object file`。
> 保险起见 smoke 后追加一句自检:
> `bash -lc 'ldd /workspace/ptoas-bin/ptoas | grep -c "not found"'` → 应输出 `0`。

### Step 3(容器内)· 整网 canonical CI

```bash
CKPT=/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp  # 按用户实际路径
OUT=/tmp/n1_ci
mkdir -p "$OUT"
# 注意:容器内空闲卡常是 0-7(不是宿主场景默认的 8-15),需 --allow-protected-devices 覆盖
cd /workspace/vllm-pypto && nohup bash -lc "python -m tests.step3p5.ci.run_whole_network_ci \
  --ckpt $CKPT --devices 0,1,2,3,4,5,6,7 --allow-protected-devices --out $OUT" \
  > $OUT/run.log 2>&1 &
echo "PID=$!"
```

**关键差异:用 `0-7` 卡时必须 `--allow-protected-devices`**

`tests/step3p5/ci/run_whole_network_ci.py` 硬编码:
- `CANONICAL_DEVICES = tuple(range(8, 16))` —— 默认 8-15
- `PROTECTED_FRONT_DEVICES = tuple(range(8))` —— 0-7 默认保护(给 8000 oracle 留位)

容器内空闲卡常是 0-7,直接 `--devices 0,1,2,3,4,5,6,7` 会被 harness 拒绝
(`devices overlap with protected front devices`),必须加 `--allow-protected-devices`。

**轮询(Bash 默认 2 分钟超时,不能 `sleep 600` 一直等)**

```bash
# 每 90s 轮询一次,直到 run.log 出现 "SINGLE_CHIP_HIDDEN_CI=PASS/FAIL"
ps -eo pid,etime,stat,cmd | grep -E "run_whole_network|stage_main|stage_mtp" | grep -v grep
tail -30 $OUT/run.log
tail -5 $OUT/main/export_rank0.log     # 看 weight pool 是否 ready
```

整网 Main 45 层 × 8 步 × 8 rank,典型时长 7-8 分钟(Main) + MTP + 可选 batch16。

**容器内判读:8000 vanilla 不可用时,从 worker log 自证**

容器内通常没有 8000 vanilla oracle 在跑(`curl 127.0.0.1:8000` 不通),skill 原版
"curl 8000 佐证"路径 N/A。改用 worker 日志直接看 pypto 每步实际 token:

```bash
grep '"output_token"' /tmp/n1_ci_artifacts/logs/main_hidden_8step.log | grep -v STRACE
```

判定矩阵(满足即判"可用"):

| step | pypto 实际 output | harness expected | token_exact | 判定 |
|------|------|------|------|------|
| 0 | 303 | 303 | true | ✅ |
| 1 | 1207 | 1207 | true | ✅ |
| 2 | **6127** | 19384(stale) | false | ✅(非精度问题,见铁律 4) |

配套健康度(任一不满足需排查):
- `hidden_finite=true`(无 NaN/Inf)
- `hidden_tp_spread=0.0`(8 rank TP 完全对齐)
- `hidden_row0_abs_max` 在百量级(660→274→368,非 0/非爆炸)

step0/step1 `token_exact=true` + step2 `output=6127`(= skill 金标准 `1207→6127`)
即足以判"可用",不需要 8000 vanilla 佐证。要更权威再跑
`tests/step3p5/ci/run_live_precision_ab.sh`(见 LIVE_PRECISION_AB.md)。

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

## 踩坑速查(容器内场景)

| 现象 | 根因 | 修法 |
|------|------|------|
| `devices overlap with protected front devices` | harness 默认保护 0-7 给 8000 oracle | 加 `--allow-protected-devices`(容器内常用 0-7) |
| codegen 一启动全崩 `libMLIRMlirOptMain.so.21.1: cannot open shared object file` | 命令没走 login shell,`/etc/profile.d/pypto-env.sh` 没 source,`LD_LIBRARY_PATH` 漏 `/workspace/ptoas-bin/lib` | **所有命令用 `bash -lc '...'`**(smoke 也要);自检 `ldd /workspace/ptoas-bin/ptoas \| grep -c "not found"` 应为 0 |
| smoke PASS 但 Step 3 codegen 崩 | smoke 只打版本字符串不真跑 codegen | 见上一行(`bash -lc` + `ldd` 自检) |
| 容器内 `curl 127.0.0.1:8000` 不通,无法佐证 step2 | 容器内通常不跑 8000 vanilla oracle | 改从 worker log `grep '"output_token"' logs/main_hidden_8step.log` 自证 step0/1 `token_exact=true` + step2 output=6127 |
| Bash 工具 sleep 600 被 SIGTERM(143) | Bash 默认 2 分钟超时 | 改 90s 轮询循环,或用 `run_in_background=true` |
| weight pool 卡在 `import_ipc_all` 不 ready | 容器没起 `--ipc host` / shm 不够 | 起容器时(外层 nerdctl)加 `--ipc host --shm-size 32g`;容器内已 inherit 则检查 `/dev/shm` 大小 |
| rank0 log 出 `bundle has 3 unexpected keys: moe_w_*_r_scale` | ckpt 多 3 个 MoE 量化 scale key,loader 容忍跳过 | 非阻塞,不影响精度;若做 W8A8 精度对齐再核对 key 来源 |

## 铁律

1. **宿主 Phase 16 三件套缺一不可**(driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1)。
2. **宿主场景 8 卡用 8-15**(0-7 常被 8000 oracle 占);**容器内场景**若空闲卡是 0-7,需 `--devices 0,1,2,3,4,5,6,7 --allow-protected-devices`(harness 默认保护 0-7)。launch 前确认目标卡空闲。
3. **命令走 `bash -lc`**(登录 shell 才 source `/etc/profile.d/pypto-env.sh`,把 `/workspace/ptoas-bin/lib` 加进 `LD_LIBRARY_PATH`);**容器内场景尤其重要**——smoke PASS ≠ codegen 能跑,必须 `bash -lc` 才能过 codegen。多卡(宿主场景)加 `--privileged --ipc host --shm-size`。
4. **runner step2 FAIL ≠ 精度问题**:是 harness stale oracle(`DEFAULT_ORACLE_TOKENS[2]=19384`),用 8000 vanilla、live A/B 或容器内 worker log 自证 step0/1 `token_exact=true` + step2 output=6127 佐证。
5. **禁 `-9` 强杀 device 进程 / 禁 `npu-smi reset`**(netboot 机重启锁死 / card poison)。
6. **容器内场景先识别**:若 `ls /workspace/pypto-smoke.sh` 存在且无 nerdctl,**不要再装容器运行时**——直接走"容器环境内场景"小节,跳过 Step 1。 
