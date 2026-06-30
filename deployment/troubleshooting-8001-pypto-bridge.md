# Troubleshooting —— 8001 PyPTO bridge（dense/shared/attention live 服务）

> 在 `gpu-a910x-0162` 上把 PyPTO 真实 kernel 接进 live vLLM（patched 8001，
> dense 0-2 + shared 3-44，未来 + attention）时的**运维排障**。撞到下面任一
> 症状先查这里。baseline 8000（cards 0-7，含 MTP `--speculative_config`）与
> patched 8001（cards 8-15）**共享 HOST pid namespace** —— 千万别 blanket-kill。

## 拓扑速查

| 服务 | 容器 | 端口 | 卡 | vllm serve 特征 |
|------|------|------|-----|----------------|
| baseline 8000 | `stepcast-vllm-isolated` | 8000 | 0-7 | 有 `--speculative_config {step3p5_mtp}` |
| patched 8001 | `stepcast-vllm-w8a8` | 8001 | 8-15 | **无** speculative；`PYPTO_DENSE_MLP_BACKEND=1` |

- 8 个 pypto worker（host venv，每 rank 一个，cards 8-15）：`python -m tools.step3p5.pypto_mlp_worker --device 8+r --rank r ...`，socket `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/pypto_mlp_rank{r}.sock`。
- 重启脚本：`/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/restart_workers.sh`（8 worker，setsid，4s stagger）。
- vLLM 启动脚本：`/logs/start_8001.sh`（容器内 `/logs` = host 上面那个目录）。

---

## ⚠ 症状 1：恢复 8001 时 vLLM `rtBinaryGetFunction 107000` / HCCL error 15，EngineCore 起不来

```
RuntimeError: create_config:...HCCLUtils.cpp:130 HCCL function error:
hcclCommInitRootInfoConfig(...), error code is 15
        rtBinaryGetFunction failed, runtime result = 107000.
```
8 个 TP worker 全报，EngineCore failed to start。

**根因**：pypto worker 占着 cards 8-15 的时候，vLLM 做 TP=8 HCCL init 会失败。
不是硬件（`npu-smi info -t health` 报 OK），`aclrtResetDeviceForce` 也清不掉。

**修复 —— 恢复顺序铁律（2026-06-30 花了 ~1h 才定位）**：

> **先起 8001 让它把 HCCL init 做完（等到 `Application startup complete` / `:8001/health=200`），再起 pypto worker。**

vLLM 在 worker 未就绪时 dense/shared 走 fail-safe fallback（原生 kernel），等 worker
起来后续请求自动切到 pypto。顺序反了（worker 先占卡）必挂。

```bash
# 1. 确认 cards 8-15 干净：无 pypto worker、无残留 8001 vLLM
pgrep -fc '[p]ypto_mlp_worker'                       # 期望 0
pgrep -af 'port 8001' | grep -v grep                 # 期望空
# 2. 先起 8001
sudo docker exec -d stepcast-vllm-w8a8 bash -c "bash /logs/start_8001.sh > /logs/boot_8001.log 2>&1"
#    盯日志直到 Application startup complete（约 1-3 min）
tail -F /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/boot_8001.log | grep -E "Application startup complete|HCCL function error|EngineCore failed"
curl -s -m5 -o /dev/null -w '%{http_code}\n' http://localhost:8001/health   # 等到 200
# 3. 再起 8 个 worker
setsid bash /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/restart_workers.sh </dev/null >/tmp/rw.log 2>&1
```

---

## ⚠ 症状 2：`pkill -f pypto_mlp_worker` 之后 worker 没起来 / 日志没生成 / ssh 命令"无输出"

**根因**：`pkill -9 -f pypto_mlp_worker` 的**命令行本身含 `pypto_mlp_worker` 字串，
`-f` 自匹配把当前 ssh shell 也 `-9` 掉了**，后续 launch 命令根本没执行。

**修复**：括号 trick（regex `[p]` 不匹配它自己 cmdline 里的字面 `[p]`）：
```bash
pkill -9 -f '[p]ypto_mlp_worker'
pgrep -fc '[p]ypto_mlp_worker'      # 计数同理用括号
```
另一个相关坑：**ssh 远程命令结尾带 `&` 后台化 worker 往往 no-op**（ssh 通道处理）。
用 `setsid bash restart_workers.sh </dev/null >LOG 2>&1`（**不带**结尾 `&`，setsid 前台
跑 ~32s，子 worker 自己 detach，ssh 正常返回 LOG）。

---

## ⚠ 症状 3：停 8001 后 / 跑完设备 IPC e2e 后，卡 8-15 进入脏状态

**根因**：(a) `kill -9` 整个 TP=8 vLLM 树时 HCCL 没机会优雅拆掉跨卡 exbus 映射；
(b) IPC exporter 调了 `aclrtIpcMemGetExportKey` 却**没调 `aclrtIpcMemClose`**，泄漏
exbus 句柄（Phase22 P5 记过）。两者都会让后续 vLLM HCCL init 报症状 1。

**预防**：任何 device-IPC exporter（含 e2e/KV-export patch）export 后必须
`aclrtIpcMemClose`（或复用长生命周期 key）。

**已知**：`aclrtResetDeviceForce` 清不掉这个状态；当前唯一可靠解法是
**按症状 1 的顺序在干净的卡上重起**（worker 全停 → 先起 vLLM → 再起 worker）。

---

## 安全停 8001（开维护窗口腾卡时）

只杀 8001 树，**保 baseline 8000（pid 见 `pgrep -af 'port 8000'`，cards 0-7）不动**：

```bash
SERVE=$(pgrep -f 'port 8001' | head -1)             # 8001 的 vllm serve 根 pid
sudo kill -9 $(pstree -p "$SERVE" | grep -oP '[(]\K[0-9]+')   # 杀整棵 8001 树
kill -9 $(pgrep -f '[p]ypto_mlp_worker')                       # 杀 8 个 worker（infra-owned）
curl -s -m5 -o /dev/null -w '8000=%{http_code}\n' http://localhost:8000/health   # 确认 baseline 仍 200
```
失败的 vLLM boot 会留一堆 `[VLLM::Worker_TP] <defunct>` 僵尸（PPID 是容器 shim）——
无害（已死），不持卡，不用管。
