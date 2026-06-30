# Phase 22 P1 device-IPC 验证环境变量记录（gpu-a910x-0234 / pypto-ascend）

> 生成时间：2026-06-29T06:50:21  
> 生成方式：在 `tmux attach -t pypto-ascend` 当前容器 shell 中执行 `/data/chensiyu/hw_project/pypto/workspace/capture_p1_env_doc.py`。  
> 注意：下方环境变量为实际运行 P1 探针的容器环境快照；包含敏感含义的变量名已脱敏为 `<REDACTED>`。

## 1. P1 验证结论

P1 同卡 device-IPC 跨进程 / 跨 ACL context 探针已通过：

```text
IMPORTER got_head=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] exp_head=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] nbytes=4096 ok=True
EXPORTER exitcode=0 cleanup=(0, 0)
P1_IPC_PROBE_PASS device=0 nbytes=4096
```

- 探针脚本：`/data/chensiyu/hw_project/pypto/workspace/p1_ipc_probe.py`
- 运行脚本：`/data/chensiyu/hw_project/pypto/workspace/run_p1_ipc_0234.sh`
- 验证日志：`/data/chensiyu/hw_project/pypto/workspace/p1_ipc_probe_0234_20260629_064226.log`
- 实际命令：

```bash
/data/chensiyu/hw_project/pypto/workspace/run_p1_ipc_0234.sh
```

## 2. 运行上下文

```text
hostname: gpu-a910x-0234.host.platform.shaipower.com
whoami: root
pwd: /workspace
date: 2026-06-29 06:50:21 UTC
python3: /usr/local/python3.11.14/bin/python3 (Python 3.11.14)
node name env: gpu-a910x-0234.host.platform.shaipower.com
gpu type env: A910X
proc per node env: 8
job id env: ws-cccc6d6920a0bd13-jlaunch-xhv7z
```

## 3. Ascend ACL IPC API / flag 确认

```text
65:#define ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT                0x0UL
66:#define ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION 0x1UL
68:#define ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT            0x0UL
69:#define ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS 0x1UL
3867: *                       ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT : Default behavior.
3868: *                       ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION : Remove whitelist verification for PID.
3889: *                        ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT : Default behavior.
3890: *                        ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS : Enables direct access to memory allocations on a peer device.
```

```text
221:} aclrtIpcMemAttrType;
3872:ACL_FUNC_VISIBILITY aclError aclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key, size_t len, uint64_t flags);
3881:ACL_FUNC_VISIBILITY aclError aclrtIpcMemClose(const char *key);
3894:ACL_FUNC_VISIBILITY aclError aclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flags);
3905:ACL_FUNC_VISIBILITY aclError aclrtIpcMemSetImportPid(const char *key, int32_t *pid, size_t num);
3916:ACL_FUNC_VISIBILITY aclError aclrtIpcMemSetAttr(const char *key, aclrtIpcMemAttrType type, uint64_t attr);
3927:ACL_FUNC_VISIBILITY aclError aclrtIpcMemImportPidInterServer(const char *key, aclrtServerPid *serverPids, size_t num);
```

P1 探针实际使用：

- export：`ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION = 0x1`
- import：`ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT = 0x0`
- memcpy：`ACL_MEMCPY_HOST_TO_DEVICE = 1`，`ACL_MEMCPY_DEVICE_TO_HOST = 2`

## 4. 实际环境变量快照（已脱敏）

```bash
ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.0.0-beta.1
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0-beta.1
ASCEND_NNAL_ENV_SET=true
ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0-beta.1/opp
ASCEND_PROCESS_LOG_PATH=/mnt/host0/ws-cccc6d6920a0bd13-jlaunch-xhv7z-cfcd2084
ASCEND_RUNTIME_OPTIONS=
ASCEND_TOOLKIT_ENV_SET=true
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0-beta.1
ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ASCEND_WORK_PATH=/mnt/host0/ws-cccc6d6920a0bd13-jlaunch-xhv7z-cfcd2084
ATB_COMPARE_TILING_EVERY_KERNEL=0
ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
ATB_MATMUL_SHUFFLE_K_ENABLE=1
ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5
ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
ATB_SHARE_MEMORY_NAME_SUFFIX=
ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0
ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
AWS_ACCESS_KEY_ID=<REDACTED>
AWS_SECRET_ACCESS_KEY=<REDACTED>
BRAIN_TOPOLOGY_LEAF_ROCE=SH-QB-G0-H3CS9825-04001
BRAIN_USERNAME=chensiyu
CMAKE_PREFIX_PATH=/usr/local/Ascend/cann-9.0.0-beta.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-9.0.0-beta.1/lib64/cmake:/usr/local/Ascend/cann-9.0.0-beta.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-9.0.0-beta.1/lib64/cmake:/usr/local/Ascend/cann-8.5.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-8.5.1/lib64/cmake
CONTAINER_WORKER_PORT_PIDMONITOR=9205
CPU_AFFINITY_CONF=1
DEBIAN_FRONTEND=noninteractive
DISTRIBUTED_JOB_ENVS_INITIALIZED=true
GPU_COUNT=8
GPU_TYPE=A910X
GPU_VENDOR=ASCEND
HCCL_CONNECT_TIMEOUT=3600
HCCL_INTRA_PCIE_ENABLE=0
HCCL_INTRA_ROCE_ENABLE=1
HCCL_OP_EXPANSION_MODE=AIV
HCCL_RDMA_RETRY_CNT=7
HCCL_RDMA_SL=4
HCCL_RDMA_TC=144
HCCL_RDMA_TIMEOUT=18
HCCL_WHITELIST_DISABLE=1
HOME=/root
HOSTNAME=gpu-a910x-0234.host.platform.shaipower.com
HOST_NETWORK=true
JOB_ID=ws-cccc6d6920a0bd13-jlaunch-xhv7z
KUBEBRAIN_CLUSTER_ENTRY=https://platform.shaipower.com
KUBEBRAIN_HOSTNAME_PREFIX=rjob-8997f16c69ecd2ca-52045d57e7718f09
KUBEBRAIN_NAMESPACE=shai-core
KUBEBRAIN_NODE_NAME=gpu-a910x-0234.host.platform.shaipower.com
KUBEBRAIN_QUOTA_GROUP=hw910test
KUBEBRAIN_REPLICA=0
KUBEBRAIN_REPLICA_NAME=ws-cccc6d6920a0bd13-jlaunch-xhv7z-cfcd2084
KUBEBRAIN_REPLICA_TOTAL=1
KUBEBRAIN_RESOURCE_TYPE=rjob
KUBEBRAIN_RJOB_NAME=ws-cccc6d6920a0bd13-jlaunch-xhv7z
KUBEBRAIN_WORKSPACE_NAME=ws-cccc6d6920a0bd13
KUBEBRAIN_WORKSPACE_NAMESPACE=shai-core
KUBERNETES_PORT=tcp://10.70.0.1:443
KUBERNETES_PORT_443_TCP=tcp://10.70.0.1:443
KUBERNETES_PORT_443_TCP_ADDR=10.70.0.1
KUBERNETES_PORT_443_TCP_PORT=443
KUBERNETES_PORT_443_TCP_PROTO=tcp
KUBERNETES_SERVICE_HOST=10.70.0.1
KUBERNETES_SERVICE_PORT=443
KUBERNETES_SERVICE_PORT_HTTPS=443
LCCL_DETERMINISTIC=0
LCCL_PARALLEL=0
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python3.11.14/lib:/usr/local/lib
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.webp=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:
MASTER_ADDR=10.201.33.28
METRICS_PUSH_GATEWAY=10.130.32.31
NCCL_IB_DISABLE=0
NCCL_IB_PCI_RELAXED_ORDERING=1
NCCL_IB_QPS_PER_CONNECTION=8
NCCL_IB_RETRY_CNT=7
NCCL_IB_TC=186
NCCL_IB_TIMEOUT=21
NCCL_NVLS_ENABLE=0
NCCL_PXN_DISABLE=1
NCCL_RAS_ADDR=0.0.0.0:28028
NCCL_SET_THREAD_NAME=1
NODE_COUNT=1
NODE_NAME=gpu-a910x-0234.host.platform.shaipower.com
NODE_RANK=0
NVIDIA_GDRCOPY=enabled
OLDPWD=/workspace
OMP_NUM_THREADS=1
PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/bishengir/bin:/usr/local/Ascend/cann-9.0.0-beta.1/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/ccec_compiler/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/profiler/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/ascend_system_advisor/asys:/usr/local/Ascend/cann-9.0.0-beta.1/tools/show_kernel_debug_data:/usr/local/Ascend/cann-9.0.0-beta.1/tools/msobjdump:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/python3.11.14/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kubebrain:/usr/local/nvidia/bin:/usr/local/go/bin
PCIDEVICE_MELLANOX_COM_MLNX_RDMA=
POD_CPU_LIMIT=60
POD_MEM_LIMIT=838860800000
PORT_0=9205
PORT_PIDMONITOR=9205
PROC_PER_NODE=8
PWD=/workspace
PYTHONPATH=/usr/local/Ascend/cann-9.0.0-beta.1/python/site-packages:/usr/local/Ascend/cann-9.0.0-beta.1/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:
RDMA_NETWORK_LINK_TYPE=none
REDIS_NODELIST=10.130.17.182:6379,10.130.17.179:6379,10.130.17.178:6379,10.130.17.181:6379,10.130.17.174:6379,10.130.17.180:6379
RJOB_TASK_INDEX=0
RLAUNCH_WORKER=1
SHLVL=1
SOC_VERSION=ascend910b1
TASK_QUEUE_ENABLE=0
TERM=xterm
TOOLCHAIN_HOME=/usr/local/Ascend/cann-9.0.0-beta.1/toolkit
TOPOLOGY_ZONE=shai-cn-shanghai-qb
TRACING_ENDPOINT=10.130.33.6
TerminationGracePeriodSeconds=30
VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
VLLM_USE_V1=1
VLLM_VERSION=0.19.0
_=/usr/local/python3.11.14/bin/python3
```

## 5. 复现命令

```bash
cd /data/chensiyu/hw_project/pypto/workspace
python3 p1_ipc_probe.py --device 0 --nbytes 4096
```

推荐使用带日志的 wrapper：

```bash
/data/chensiyu/hw_project/pypto/workspace/run_p1_ipc_0234.sh
```
