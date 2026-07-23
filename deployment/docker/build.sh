#!/usr/bin/env bash
# 构建 vllm-pypto 可复现镜像。配方在单一 Dockerfile;每次构建的 pins+tag 在
# builds/<spec>.env(见 README「组织方式」)。
# 用法:
#   bash build.sh [builds/<spec>.env]        # 默认最新 spec
#   GH=.../github.env GL=.../gitlab.env bash build.sh builds/stepfun-develop-20260723.env
# 需要: docker (BuildKit --secret)。GH/GL 是含 PAT 的文件 (不落镜像层)。
set -euo pipefail
cd "$(dirname "$0")"

SPEC=${1:-${SPEC:-builds/stepfun-develop-20260723.env}}
[ -f "$SPEC" ] || { echo "缺 build spec: $SPEC (见 builds/)"; exit 1; }
# shellcheck disable=SC1090
source "$SPEC"    # IMAGE_TAG + PYPTO_COMMIT/PYPTO_LIB_COMMIT/PTO_ISA_COMMIT/PTOAS_COMMIT/SIMPLER_COMMIT/PTOAS_BIN_VER/VLLM_PATCH_BRANCH
: "${IMAGE_TAG:?spec 缺 IMAGE_TAG}"

GH=${GH:-/data/chensiyu/secrets/github.env}
GL=${GL:-/data/chensiyu/secrets/gitlab.env}
IMG=${IMG:-hub.i.basemind.com/stepcast/vllm-pypto:${IMAGE_TAG}}
BASE=${BASE:-hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938}

[ -f "$GH" ] || { echo "缺 GitHub token 文件: $GH"; exit 1; }
[ -f "$GL" ] || { echo "缺 GitLab token 文件: $GL"; exit 1; }

# ptoas-bin: fork 无 release asset,从 0162 验证过的二进制 bake 进 context。
PTOAS_BIN_SRC=${PTOAS_BIN_SRC:-/data/chensiyu/hw_project/pypto/workspace/ptoas-bin}
if [ ! -f ptoas-bin.tgz ]; then
  [ -x "$PTOAS_BIN_SRC/bin/ptoas" ] || { echo "缺 ptoas 二进制: $PTOAS_BIN_SRC/bin/ptoas (设 PTOAS_BIN_SRC)"; exit 1; }
  echo "[build] 打包 ptoas-bin (bin+lib+顶层 ptoas 符号链接) 从 $PTOAS_BIN_SRC ..."
  tar czf ptoas-bin.tgz -C "$PTOAS_BIN_SRC" bin lib $([ -e "$PTOAS_BIN_SRC/ptoas" ] && echo ptoas)
  echo "[build] ptoas-bin.tgz = $(du -h ptoas-bin.tgz | cut -f1)"
fi

echo "[build] SPEC=$SPEC  IMG=$IMG"
echo "[build] pins: pypto=$PYPTO_COMMIT pypto-lib=$PYPTO_LIB_COMMIT pto-isa=$PTO_ISA_COMMIT PTOAS=$PTOAS_COMMIT simpler=$SIMPLER_COMMIT ptoas-bin=$PTOAS_BIN_VER"

# github clone 需经宿主可达的代理; 优先官方入口 (deploy.i.shaipower.com/httpproxy),
# 拿不到回落 Dockerfile 内置默认。内网 (pip/gitlab/hub) 直连不走代理。
PROXY_ARGS=()
if eval "$(curl -fsS http://deploy.i.shaipower.com/httpproxy 2>/dev/null)" 2>/dev/null && [ -n "${http_proxy:-}" ]; then
  echo "[build] 代理(官方入口): $http_proxy"
  PROXY_ARGS=(--build-arg GH_PROXY="$http_proxy" --build-arg NO_PROXY_HOSTS="${no_proxy:-basemind.com,shaipower.com,127.0.0.1,localhost}")
else
  echo "[build] 官方代理入口不可达, 用 Dockerfile 内置默认"
fi

# --network=host 走宿主路由到代理; DOCKER_BUILDKIT=1 docker build 同时支持 --secret + --network=host。
DOCKER_BUILDKIT=1 docker build \
  --network=host \
  --build-arg BASE="$BASE" \
  --build-arg PYPTO_COMMIT="$PYPTO_COMMIT" \
  --build-arg PYPTO_LIB_COMMIT="$PYPTO_LIB_COMMIT" \
  --build-arg PTO_ISA_COMMIT="$PTO_ISA_COMMIT" \
  --build-arg PTOAS_COMMIT="$PTOAS_COMMIT" \
  --build-arg SIMPLER_COMMIT="$SIMPLER_COMMIT" \
  --build-arg PTOAS_BIN_VER="$PTOAS_BIN_VER" \
  --build-arg VLLM_PATCH_BRANCH="$VLLM_PATCH_BRANCH" \
  "${PROXY_ARGS[@]}" \
  --secret id=gh_token,src="$GH" \
  --secret id=gl_token,src="$GL" \
  --progress=plain \
  -t "$IMG" \
  -f Dockerfile \
  .
echo "[build] done: $IMG"
echo "[build] 推送 (可选): docker push $IMG"
