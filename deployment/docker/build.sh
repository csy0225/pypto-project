#!/usr/bin/env bash
# 构建 vllm-pypto 可复现镜像。配方在单一 Dockerfile;每次构建的 pins+tag 在
# builds/<spec>.env(见 README「组织方式」)。
# 用法:
#   bash build.sh [builds/<spec>.env]        # 默认最新 spec
#   GH=.../github.env GL=.../gitlab.env bash build.sh builds/stepfun-develop-20260726-step3p5-only.env
# 需要: docker (BuildKit --secret)。GH/GL 是含 PAT 的文件 (不落镜像层)。
set -euo pipefail
cd "$(dirname "$0")"

SPEC=${1:-${SPEC:-builds/stepfun-develop-20260726-step3p5-only.env}}
[ -f "$SPEC" ] || { echo "缺 build spec: $SPEC (见 builds/)"; exit 1; }
# shellcheck disable=SC1090
source "$SPEC"    # IMAGE_TAG + immutable source pins
: "${IMAGE_TAG:?spec 缺 IMAGE_TAG}"
BUILD_JOBS=${BUILD_JOBS:-2}
ATTN_TASK_PROFILE=${ATTN_TASK_PROFILE:-portable}
REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN=${REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN:-0}
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "BUILD_JOBS 必须是正整数，当前值: $BUILD_JOBS"
  exit 1
}
case "$ATTN_TASK_PROFILE" in
  portable|a2a3) ;;
  *)
    echo "ATTN_TASK_PROFILE 仅支持 portable/a2a3，当前值: $ATTN_TASK_PROFILE"
    exit 1
    ;;
esac
case "$REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN" in
  0|1) ;;
  *)
    echo "REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN 必须是 0 或 1，当前值: $REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN"
    exit 1
    ;;
esac

GH=${GH:-/data/chensiyu/secrets/github.env}
GL=${GL:-/data/chensiyu/secrets/gitlab.env}
IMG=${IMG:-hub.i.basemind.com/stepcast/vllm-pypto:${IMAGE_TAG}}
BASE=${BASE:-hub.i.basemind.com/stepcast/vllm-pypto@sha256:3d6392588fe9fb6ce4f5852100667d24f09d70f262dbd0ebe6c45b380f49573a}

# 两台构建机的容器运行时不同,但吃同一个 Dockerfile:
#   devbox — docker + BuildKit,`--network=host` 走宿主到代理
#   0162   — 无 docker,只有 containerd + nerdctl,buildkitd 已带
#            `--containerd-worker-net=host`(所以不需要也不支持 `--network`)
# 设 CONTAINER_CLI 覆盖;默认按现场探测。
if [ -z "${CONTAINER_CLI:-}" ]; then
  if command -v docker >/dev/null 2>&1; then
    CONTAINER_CLI="docker"
  elif [ -x /mnt/persist/k8s-install/containerd/bin/nerdctl ]; then
    CONTAINER_CLI="sudo -n /mnt/persist/k8s-install/containerd/bin/nerdctl"
  else
    echo "找不到 docker 也找不到 nerdctl;设 CONTAINER_CLI"
    exit 1
  fi
fi
export CONTAINER_CLI            # audit_image_credentials.py 也用它
export DOCKER_BUILDKIT=1        # docker 用;nerdctl 忽略
read -r -a CLI <<< "$CONTAINER_CLI"
case "$CONTAINER_CLI" in *docker) IS_DOCKER=1 ;; *) IS_DOCKER=0 ;; esac
echo "[build] 容器 CLI: $CONTAINER_CLI"

BUILD_ARGS=()
if [ "$IS_DOCKER" = 1 ]; then
  # BuildKit 的 sandbox 网络在 devbox 上到不了代理,必须借宿主路由。
  BUILD_ARGS+=(--network=host)
else
  # nerdctl build 没有 --network;buildkitd 已经 host 网络。
  BUILD_ARGS+=(--buildkit-host "${BUILDKIT_HOST:-unix:///run/buildkit/moe-r2.sock}")
  # nerdctl build shells out to buildctl, which is not on the default PATH here.
  # Exporting PATH is not enough: nerdctl runs under sudo, and sudo's secure_path
  # discards it. Insert `env PATH=...` right before the runtime binary (the last
  # element) so it applies to whatever wrapper prefix CONTAINER_CLI carries.
  if ! command -v buildctl >/dev/null 2>&1; then
    BUILDCTL_DIR=${BUILDCTL_DIR:-/mnt/persist/chensiyu/buildkit/extracted/bin}
    [ -x "$BUILDCTL_DIR/buildctl" ] || {
      echo "nerdctl build 需要 buildctl,未找到: $BUILDCTL_DIR/buildctl"; exit 1; }
    CLI=("${CLI[@]:0:${#CLI[@]}-1}" env "PATH=$BUILDCTL_DIR:$PATH" "${CLI[@]: -1}")
    echo "[build] buildctl: $BUILDCTL_DIR"
  fi
fi

# GH 只给 audit_image_credentials.py 扫历史用;源码走 src-pins.tgz,构建期不需要它。
[ -f "$GH" ] || { echo "缺 GitHub token 文件: $GH (凭据审计需要)"; exit 1; }
[ -f "$GL" ] || { echo "缺 GitLab token 文件: $GL"; exit 1; }

# The historical StepCast base recorded a literal GitLab token in OCI history.
# Pull and audit the digest-pinned sanitized base before allowing a build.
"${CLI[@]}" pull "$BASE" >/dev/null
python3 audit_image_credentials.py "$BASE" "$GH" "$GL"

# ptoas-bin: fork 无 release asset,从 0162 验证过的二进制 bake 进 context。
# 从 v0.51 起发布树是 `bin lib ptoas ptodsl .ptoas-python-version`,必须整棵打包:
# `ptodsl/` 被 ptoas 自己的 `_runtime/share/ptoas/{SoftOps,TileOps}/*.py` 引用;
# `.ptoas-python-version` 缺失会让 launcher **静默跳过**解释器版本校验。
PTOAS_BIN_SRC=${PTOAS_BIN_SRC:-/data/chensiyu/hw_project/pypto/workspace/ptoas-bin}
if [ ! -f ptoas-bin.tgz ]; then
  [ -x "$PTOAS_BIN_SRC/bin/ptoas" ] || { echo "缺 ptoas 二进制: $PTOAS_BIN_SRC/bin/ptoas (设 PTOAS_BIN_SRC)"; exit 1; }
  echo "[build] 打包 ptoas-bin (整棵发布树) 从 $PTOAS_BIN_SRC ..."
  tar czf ptoas-bin.tgz -C "$PTOAS_BIN_SRC" .
  echo "[build] ptoas-bin.tgz = $(du -h ptoas-bin.tgz | cut -f1)"
fi
if [ -n "${PTOAS_BIN_SHA256:-}" ]; then
  actual_ptoas_sha=$(
    tar -xOzf ptoas-bin.tgz ./bin/ptoas | sha256sum | awk '{print $1}'
  )
  [ "$actual_ptoas_sha" = "$PTOAS_BIN_SHA256" ] || {
    echo "ptoas payload SHA256 mismatch: expected=$PTOAS_BIN_SHA256 actual=$actual_ptoas_sha"
    exit 1
  }
fi
if [ -n "${PTOAS_BIN_ARCHIVE_SHA256:-}" ]; then
  actual_ptoas_archive_sha=$(sha256sum ptoas-bin.tgz | awk '{print $1}')
  [ "$actual_ptoas_archive_sha" = "$PTOAS_BIN_ARCHIVE_SHA256" ] || {
    echo "ptoas archive SHA256 mismatch: expected=$PTOAS_BIN_ARCHIVE_SHA256 actual=$actual_ptoas_archive_sha"
    exit 1
  }
fi

echo "[build] SPEC=$SPEC  IMG=$IMG"
echo "[build] pins: pypto=$PYPTO_COMMIT pypto-lib=$PYPTO_LIB_COMMIT pto-isa=$PTO_ISA_COMMIT PTOAS=$PTOAS_COMMIT simpler=$SIMPLER_COMMIT ptoas-bin=$PTOAS_BIN_VER"
echo "[build] profile: attention=$ATTN_TASK_PROFILE build-jobs=$BUILD_JOBS l2-swimlane-reuse-required=$REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN"

# src-pins.tgz: 七个 bare mirror 打进 build context,Dockerfile 从它 clone,构建期
# 完全不碰 github。原因是唯一有足够核数编 simpler 的机器 (0162) **没有 github
# 出口** —— 直连 TCP 超时 130s,代理主机名连 DNS 都解析不了。
# 这不降低可复现性: git 对象名就是内容哈希,带着 commit <sha> 的 mirror 就是
# github 的 <sha>,Dockerfile 里的 rev-parse 断言照旧成立。
# 每个仓的路径可单独覆盖 (SRC_PYPTO 等);默认落在 SRC_PINS_DIR 下的同名目录。
# 3rdparty 两个 submodule 要的 sha 直接从 pypto 的 tree 读,不另设 pin。
SRC_PINS_DIR=${SRC_PINS_DIR:-}
src_pypto=${SRC_PYPTO:-$SRC_PINS_DIR/pypto}
if [ ! -f src-pins.tgz ]; then
  [ -n "$SRC_PINS_DIR" ] || [ -n "${SRC_PYPTO:-}" ] || {
    echo "缺 src-pins.tgz;设 SRC_PINS_DIR 指向含五仓 checkout 的目录 (可用 SRC_<仓> 单独覆盖)"
    exit 1
  }
  sub_sha() { git -C "$src_pypto" ls-tree "$PYPTO_COMMIT" "$1" | awk '{print $3}'; }
  src_stage=$(mktemp -d)
  trap 'rm -rf "$src_stage"' EXIT
  mirror_pin() {   # name  src_path  wanted_sha
    local n=$1 p=$2 s=$3
    [ -e "$p/.git" ] || [ -d "$p/objects" ] || { echo "缺源码仓: $p (设 SRC_* 覆盖)"; exit 1; }
    [ -n "$s" ] || { echo "$n 的目标 sha 为空"; exit 1; }
    [ "$(git -C "$p" cat-file -t "$s" 2>/dev/null)" = commit ] || {
      echo "$p 里没有 commit $s ($n) —— 先在那边 fetch 到再来构建"; exit 1; }
    git clone -q --mirror --no-hardlinks "$p" "$src_stage/$n.git"
    [ "$(git -C "$src_stage/$n.git" cat-file -t "$s" 2>/dev/null)" = commit ] || {
      echo "$n 的 mirror 没带上 $s"; exit 1; }
    printf '[build]   %-14s %s  %s\n' "$n" "${s:0:12}" "$p"
  }
  echo "[build] 组装 src-pins.tgz (bare mirror,构建期不碰 github) ..."
  mirror_pin pypto        "$src_pypto"                                          "$PYPTO_COMMIT"
  mirror_pin pypto-lib    "${SRC_PYPTO_LIB:-$SRC_PINS_DIR/pypto-lib}"           "$PYPTO_LIB_COMMIT"
  mirror_pin pto-isa      "${SRC_PTO_ISA:-$SRC_PINS_DIR/pto-isa}"               "$PTO_ISA_COMMIT"
  mirror_pin PTOAS        "${SRC_PTOAS:-$SRC_PINS_DIR/PTOAS}"                   "$PTOAS_COMMIT"
  mirror_pin simpler      "${SRC_SIMPLER:-$src_pypto/runtime}"                  "$SIMPLER_COMMIT"
  mirror_pin libbacktrace "${SRC_LIBBACKTRACE:-$src_pypto/3rdparty/libbacktrace}" "$(sub_sha 3rdparty/libbacktrace)"
  mirror_pin msgpack-c    "${SRC_MSGPACK:-$src_pypto/3rdparty/msgpack-c}"       "$(sub_sha 3rdparty/msgpack-c)"
  # pypto 的 gitlink 必须和 SIMPLER_COMMIT 一致,否则镜像里 submodule 会指到别处
  gitlink_simpler=$(git -C "$src_pypto" ls-tree "$PYPTO_COMMIT" runtime | awk '{print $3}')
  [ "$gitlink_simpler" = "$SIMPLER_COMMIT" ] || {
    echo "pypto@$PYPTO_COMMIT 的 runtime gitlink=$gitlink_simpler != SIMPLER_COMMIT=$SIMPLER_COMMIT"
    exit 1
  }
  tar czf src-pins.tgz -C "$src_stage" .
  rm -rf "$src_stage"; trap - EXIT
  echo "[build] src-pins.tgz = $(du -h src-pins.tgz | cut -f1)"
fi

PROXY_ARGS=()
if [ "$IS_DOCKER" = 1 ]; then
  # 只给 pip 用: 内网 (pip/gitlab/hub) 直连不走代理,但 pypto 的构建依赖里有走
  # 公网的。优先官方入口 (deploy.i.shaipower.com/httpproxy),拿不到回落 Dockerfile
  # 内置默认。源码不再在构建期 clone,所以这里跟 github 无关。
  if eval "$(curl -fsS http://deploy.i.shaipower.com/httpproxy 2>/dev/null)" 2>/dev/null && [ -n "${http_proxy:-}" ]; then
    echo "[build] 代理(官方入口): $http_proxy"
    PROXY_ARGS=(--build-arg GH_PROXY="$http_proxy" --build-arg NO_PROXY_HOSTS="${no_proxy:-basemind.com,shaipower.com,127.0.0.1,localhost}")
  else
    echo "[build] 官方代理入口不可达, 用 Dockerfile 内置默认"
  fi
else
  # 0162 完全没有公网出口: proxy.i.shaipower.com 连 DNS 都解析不了,直连 github
  # TCP 超时 130s。传空代理,否则每个请求都会挂在死代理上。内网 hub/gitlab 直连可达。
  echo "[build] 不走代理(0162 无公网出口,内网直连)"
  PROXY_ARGS=(--build-arg GH_PROXY=)
fi

"${CLI[@]}" build \
  "${BUILD_ARGS[@]}" \
  --build-arg BASE="$BASE" \
  --build-arg PYPTO_COMMIT="$PYPTO_COMMIT" \
  --build-arg PYPTO_LIB_COMMIT="$PYPTO_LIB_COMMIT" \
  --build-arg PTO_ISA_COMMIT="$PTO_ISA_COMMIT" \
  --build-arg PTOAS_COMMIT="$PTOAS_COMMIT" \
  --build-arg SIMPLER_COMMIT="$SIMPLER_COMMIT" \
  --build-arg PTOAS_BIN_VER="$PTOAS_BIN_VER" \
  --build-arg ATTN_TASK_PROFILE="$ATTN_TASK_PROFILE" \
  --build-arg BUILD_JOBS="$BUILD_JOBS" \
  --build-arg REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN="$REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN" \
  --build-arg VLLM_PATCH_BRANCH="$VLLM_PATCH_BRANCH" \
  --build-arg VLLM_PATCH_COMMIT="$VLLM_PATCH_COMMIT" \
  "${PROXY_ARGS[@]}" \
  --secret id=gl_token,src="$GL" \
  --progress=plain \
  -t "$IMG" \
  -f Dockerfile \
  .
echo "[build] done: $IMG"
python3 audit_image_credentials.py "$IMG" "$GH" "$GL"
echo "[build] 推送 (可选): $CONTAINER_CLI push $IMG"
