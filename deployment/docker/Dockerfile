# =============================================================================
# vllm-pypto 可复现环境 (Ascend 910B / step3p5)
# 基于 0162 验证过的环境 + stepfun/develop 分支 pins。
#
# base 镜像自带: CANN 8.5.1 + CANN 9.0.0-beta.1 + vLLM 0.19.0 + vllm-ascend
#                + python3.11.14 (/usr/local/python3.11.14)。
# 本 Dockerfile 在其上: 删 CANN 8.5.1 (只留 beta.1) -> clone pypto 栈 (pins)
#                       -> ptoas-bin v0.45 -> 编译 pypto + runtime(a2a3)
#                       -> 打 vLLM Track-B 补丁 -> 设 env。
#
# build: 见同目录 build.sh (用 docker buildx --secret 传 GH_TOKEN / GL_TOKEN)
# =============================================================================
ARG BASE=hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938
FROM ${BASE}

SHELL ["/bin/bash", "-lc"]

# ---- stepfun/develop pins (2026-07-23 验证) --------------------------------
ARG PYPTO_COMMIT=9ec303f6f984b96a70d467cd46185b7b9157d3ec
ARG PYPTO_LIB_COMMIT=4c48215b
ARG PTO_ISA_COMMIT=ecb6c303
ARG PTOAS_COMMIT=72ada0a1
ARG SIMPLER_COMMIT=c7fdc574fb0637fda581398e5064994734d913c5
ARG PTOAS_BIN_VER=v0.45
ARG VLLM_PATCH_BRANCH=csy/pypto-tail-mtp-integration

ENV WS=/workspace
ENV PYTHON=/usr/local/python3.11.14/bin/python3
ENV PIP="/usr/local/python3.11.14/bin/python3 -m pip"

# ---- 1) 只保留 CANN 9.0.0-beta.1 (删 8.5.1, 避免干扰后续任务) --------------
RUN rm -rf /usr/local/Ascend/cann-8.5.1 && \
    ln -sfn /usr/local/Ascend/cann-9.0.0-beta.1 /usr/local/Ascend/cann

# ---- 2) clone pypto 栈 @ stepfun/develop pins (github csy0225 forks) --------
WORKDIR ${WS}
RUN --mount=type=secret,id=gh_token \
    GH="$(cat /run/secrets/gh_token)" && \
    for r in pypto pypto-lib pto-isa PTOAS; do \
      git -c http.version=HTTP/1.1 clone "https://${GH}@github.com/csy0225/${r}.git" "${WS}/${r}"; \
    done && \
    cd ${WS}/pypto && git checkout ${PYPTO_COMMIT} && \
      git config submodule.runtime.url "https://${GH}@github.com/csy0225/simpler.git" && \
      git submodule update --init runtime && \
      ( cd runtime && git checkout ${SIMPLER_COMMIT} ) && \
    cd ${WS}/pypto-lib && git fetch -q "https://${GH}@github.com/csy0225/pypto-lib.git" 'stepfun/develop:refs/heads/stepfun/develop' && git checkout stepfun/develop && \
    cd ${WS}/pto-isa && git checkout ${PTO_ISA_COMMIT} && \
    cd ${WS}/PTOAS && git checkout ${PTOAS_COMMIT} && \
    ( cd ${WS}/pypto-lib && git worktree add ${WS}/vllm-pypto stepfun/develop )

# ---- 3) ptoas-bin v0.45 (binary release) -----------------------------------
RUN --mount=type=secret,id=gh_token \
    GH="$(cat /run/secrets/gh_token)" && \
    mkdir -p ${WS}/ptoas-bin && cd ${WS}/ptoas-bin && \
    ( curl -fsSL -H "Authorization: token ${GH}" \
        "https://github.com/csy0225/PTOAS/releases/download/${PTOAS_BIN_VER}/ptoas-bin-x86_64.tar.gz" \
        -o /tmp/ptoas-bin.tgz && \
      tar xzf /tmp/ptoas-bin.tgz -C ${WS}/ptoas-bin --strip-components=1 && rm -f /tmp/ptoas-bin.tgz ) || \
    echo "WARN: ptoas-bin 下载失败, 请在 build 前手动放置 ${WS}/ptoas-bin (bin/ lib/ share/)"

# ---- 4) 编译 pypto + runtime (CANN beta.1 env) -----------------------------
ENV PTO_ISA_ROOT=${WS}/pto-isa
ENV PTOAS_ROOT=${WS}/ptoas-bin
ENV LD_LIBRARY_PATH=${WS}/ptoas-bin/lib:${LD_LIBRARY_PATH}
ENV PATH=${WS}/ptoas-bin/bin:${PATH}
RUN source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh && \
    export PTO_ISA_ROOT=${WS}/pto-isa && \
    ${PIP} install --no-cache-dir "cmake==3.31.6" && \
    ${PIP} install --no-cache-dir --no-build-isolation -e ${WS}/pypto && \
    ${PIP} install --no-cache-dir --no-build-isolation -e ${WS}/pypto/runtime && \
    cd ${WS}/pypto && ${PYTHON} -m simpler_setup.build_runtimes --platforms a2a3 --clone-protocol https

# ---- 5) vLLM Track-B 补丁 (tail-only 主网 + MTP-proposer + MTP3 boot fix) ---
#     从 gitlab 分支取 step3p5.py + step3p5_mtp.py 覆盖 base 的 /vllm-workspace/vllm
RUN --mount=type=secret,id=gl_token \
    GL="$(cat /run/secrets/gl_token)" && \
    tmp=$(mktemp -d) && \
    git -c http.version=HTTP/1.1 clone --depth 1 -b ${VLLM_PATCH_BRANCH} \
      "https://oauth2:${GL}@gitlab.basemind.com/sys/stepcast/vllm.git" "$tmp/vllm" && \
    cp "$tmp/vllm/vllm/model_executor/models/step3p5.py"     /vllm-workspace/vllm/vllm/model_executor/models/step3p5.py && \
    cp "$tmp/vllm/vllm/model_executor/models/step3p5_mtp.py" /vllm-workspace/vllm/vllm/model_executor/models/step3p5_mtp.py && \
    rm -rf "$tmp"

# ---- 6) runtime env (每次登录 shell / serve 生效) --------------------------
ENV PYPTO_PROG_BUILD_DIR=/tmp/pypto_build_output
ENV PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072
RUN printf '%s\n' \
  'source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh' \
  'export WS=/workspace' \
  'export PTO_ISA_ROOT=$WS/pto-isa PTOAS_ROOT=$WS/ptoas-bin' \
  'export LD_LIBRARY_PATH=$WS/ptoas-bin/lib:$LD_LIBRARY_PATH' \
  'export PATH=$WS/ptoas-bin/bin:/usr/local/python3.11.14/bin:$PATH' \
  'export PYTHONPATH=$WS/vllm-pypto:$WS/pypto/python:$WS/pypto/runtime/python:$PYTHONPATH' \
  'export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072' \
  > /etc/profile.d/pypto-env.sh

WORKDIR ${WS}
CMD ["/bin/bash", "-l"]
