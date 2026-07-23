#!/usr/bin/env bash
# 构建 vllm-pypto 可复现镜像。
# 用法:
#   GH=/data/chensiyu/secrets/github.env GL=/data/chensiyu/secrets/gitlab.env \
#   IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260723 \
#   bash build.sh
#
# 需要: docker (支持 buildx --secret)。GH/GL 是含 PAT 的文件 (不落镜像层)。
set -euo pipefail
cd "$(dirname "$0")"

GH=${GH:-/data/chensiyu/secrets/github.env}
GL=${GL:-/data/chensiyu/secrets/gitlab.env}
IMG=${IMG:-vllm-pypto:stepfun-develop-$(date +%Y%m%d)}
BASE=${BASE:-hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938}

[ -f "$GH" ] || { echo "缺 GitHub token 文件: $GH"; exit 1; }
[ -f "$GL" ] || { echo "缺 GitLab token 文件: $GL"; exit 1; }

# ptoas-bin v0.45: fork 无 release asset,从 0162 验证过的二进制 bake 进 context。
PTOAS_BIN_SRC=${PTOAS_BIN_SRC:-/data/chensiyu/hw_project/pypto/workspace/ptoas-bin}
if [ ! -f ptoas-bin.tgz ]; then
  [ -x "$PTOAS_BIN_SRC/bin/ptoas" ] || { echo "缺 ptoas 二进制: $PTOAS_BIN_SRC/bin/ptoas (设 PTOAS_BIN_SRC)"; exit 1; }
  echo "[build] 打包 ptoas-bin (bin+lib+顶层 ptoas 符号链接) 从 $PTOAS_BIN_SRC ..."
  tar czf ptoas-bin.tgz -C "$PTOAS_BIN_SRC" bin lib $([ -e "$PTOAS_BIN_SRC/ptoas" ] && echo ptoas)
  echo "[build] ptoas-bin.tgz = $(du -h ptoas-bin.tgz | cut -f1)"
fi

echo "[build] BASE=$BASE"
echo "[build] IMG=$IMG"
# github clone 需经宿主可达的代理; 优先用官方入口 (deploy.i.shaipower.com/httpproxy),
# 拿不到就回落 Dockerfile 内置默认 (proxy.i.shaipower.com:3128)。内网 (pip/gitlab/hub) 直连。
PROXY_ARGS=()
if eval "$(curl -fsS http://deploy.i.shaipower.com/httpproxy 2>/dev/null)" 2>/dev/null && [ -n "${http_proxy:-}" ]; then
  echo "[build] 代理(官方入口): $http_proxy"
  PROXY_ARGS=(--build-arg GH_PROXY="$http_proxy" --build-arg NO_PROXY_HOSTS="${no_proxy:-basemind.com,shaipower.com,127.0.0.1,localhost}")
else
  echo "[build] 官方代理入口不可达, 用 Dockerfile 内置默认"
fi
# 用 --network=host: 走宿主路由到代理; 用 DOCKER_BUILDKIT=1 docker build 同时支持 --secret + --network=host。
DOCKER_BUILDKIT=1 docker build \
  --network=host \
  --build-arg BASE="$BASE" \
  "${PROXY_ARGS[@]}" \
  --secret id=gh_token,src="$GH" \
  --secret id=gl_token,src="$GL" \
  --progress=plain \
  -t "$IMG" \
  -f Dockerfile \
  .
echo "[build] done: $IMG"
echo "[build] 推送 (可选): docker push $IMG"
