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

echo "[build] BASE=$BASE"
echo "[build] IMG=$IMG"
DOCKER_BUILDKIT=1 docker buildx build \
  --build-arg BASE="$BASE" \
  --secret id=gh_token,src="$GH" \
  --secret id=gl_token,src="$GL" \
  --progress=plain \
  --load \
  -t "$IMG" \
  -f Dockerfile \
  .
echo "[build] done: $IMG"
echo "[build] 推送 (可选): docker push $IMG"
