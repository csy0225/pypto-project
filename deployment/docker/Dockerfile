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
# pypto develop 8af501fc = 9ec303f6 + runtime submodule gitlink 回退到可编译的 36957c6b
# (原 9ec303f6 gitlink 指向 c7fdc574, 其 Phase-24 import_ipc 半成品编译不过)。
ARG PYPTO_COMMIT=8af501fc01c91d6f02da8e6a29ef7d73c8049082
ARG PYPTO_LIB_COMMIT=4c48215b
ARG PTO_ISA_COMMIT=ecb6c303
ARG PTOAS_COMMIT=72ada0a1
# simpler stepfun/develop 已回退到 36957c6b (proven-buildable, 0162 验证过的 .so 就是它)。
# 原 c7fdc574 = 36957c6b + 9 WIP commit, 其中 Phase-24 import_ipc 半成品编译不过
# (get_worker 笔误 + control_import_ipc 缺头声明; 该 facade 已被 Python 路径取代)。
# 原 c7fdc574 保留在 simpler tag backup/stepfun-develop-c7fdc574-20260723。
# pypto develop 8af501fc 的 runtime gitlink 已指向 36957c6b, submodule update 直接落对。
ARG SIMPLER_COMMIT=36957c6b56700ecba3aeb8dbbedd6240594e01de
ARG PTOAS_BIN_VER=v0.45
ARG VLLM_PATCH_BRANCH=csy/pypto-tail-mtp-integration
# github 不在内网直连,需经代理; 内网 (basemind.com: 镜像/pip/gitlab) 直连不走代理。
ARG GH_PROXY=http://proxy.i.shaipower.com:3128
ARG NO_PROXY_HOSTS=basemind.com,shaipower.com,shaipower,127.0.0.1,localhost,.svc

ENV WS=/workspace
ENV PYTHON=/usr/local/python3.11.14/bin/python3
ENV PIP="/usr/local/python3.11.14/bin/python3 -m pip"

# ---- 1) 只保留 CANN 9.0.0-beta.1 (删 8.5.1, 避免干扰后续任务) --------------
RUN rm -rf /usr/local/Ascend/cann-8.5.1 && \
    ln -sfn /usr/local/Ascend/cann-9.0.0-beta.1 /usr/local/Ascend/cann

# ---- 2) clone pypto 栈 @ stepfun/develop pins (github csy0225 forks) --------
WORKDIR ${WS}
RUN --mount=type=secret,id=gh_token \
    export https_proxy=${GH_PROXY} http_proxy=${GH_PROXY} no_proxy=${NO_PROXY_HOSTS} && \
    GH="$(cat /run/secrets/gh_token)" && \
    for r in pypto pypto-lib pto-isa PTOAS; do \
      git -c http.version=HTTP/1.1 clone "https://${GH}@github.com/csy0225/${r}.git" "${WS}/${r}"; \
    done && \
    cd ${WS}/pypto && git checkout ${PYPTO_COMMIT} && \
      git config submodule.runtime.url "https://${GH}@github.com/csy0225/simpler.git" && \
      git submodule update --init && \
      ( cd runtime && git checkout ${SIMPLER_COMMIT} ) && \
    cd ${WS}/pypto-lib && git fetch -q "https://${GH}@github.com/csy0225/pypto-lib.git" 'stepfun/develop:refs/heads/stepfun/develop' && git checkout stepfun/develop && \
    cd ${WS}/pto-isa && git checkout ${PTO_ISA_COMMIT} && \
    cd ${WS}/PTOAS && git checkout ${PTOAS_COMMIT} && \
    ln -sfn ${WS}/pypto-lib ${WS}/vllm-pypto

# ---- 3) ptoas-bin v0.45 (从 0162 验证过的二进制 bake; fork 无 release asset) ---
#     build.sh 从 $PTOAS_BIN_SRC ($WS/ptoas-bin/{bin,lib}) 打 ptoas-bin.tgz 进 context。
COPY ptoas-bin.tgz /tmp/ptoas-bin.tgz
RUN mkdir -p ${WS}/ptoas-bin && \
    tar xzf /tmp/ptoas-bin.tgz -C ${WS}/ptoas-bin && rm -f /tmp/ptoas-bin.tgz && \
    LD_LIBRARY_PATH=${WS}/ptoas-bin/lib ${WS}/ptoas-bin/bin/ptoas --version

# ---- 4) 编译 pypto + runtime (CANN beta.1 env) -----------------------------
ENV PTO_ISA_ROOT=${WS}/pto-isa
ENV PTOAS_ROOT=${WS}/ptoas-bin
ENV LD_LIBRARY_PATH=${WS}/ptoas-bin/lib:${LD_LIBRARY_PATH}
ENV PATH=${WS}/ptoas-bin/bin:${PATH}
# pypto/runtime 用 scikit-build-core 后端; --no-build-isolation 不装后端,须先备好。
# 限制编译并行度 (devbox dockerd 在 memcg 下, 17GB/5核; 模板重的 C++ 全并行会 OOM 打挂 dockerd)。
ENV CMAKE_BUILD_PARALLEL_LEVEL=2
ENV MAX_JOBS=2
# 4a) 编译后端 + pypto (慢, ~14min; 单独一层, 改 runtime 步不重编 pypto)
RUN source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh && \
    export https_proxy=${GH_PROXY} http_proxy=${GH_PROXY} no_proxy=${NO_PROXY_HOSTS} && \
    git config --global http.version HTTP/1.1 && \
    git config --global --add safe.directory '*' && \
    export PTO_ISA_ROOT=${WS}/pto-isa && \
    ${PIP} install --no-cache-dir "cmake==3.31.6" \
        "scikit-build-core>=0.10.0" "nanobind>=2.0.0" "ninja>=1.11.0" && \
    ${PIP} install --no-cache-dir --no-build-isolation -e ${WS}/pypto

# 4b) simpler runtime + build_runtimes(a2a3); build_runtimes 内部会 clone
#     hw-native-sys/pto-isa 到 runtime/build/pto-isa(managed), 故需代理 + HTTP/1.1。
RUN source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh && \
    export https_proxy=${GH_PROXY} http_proxy=${GH_PROXY} no_proxy=${NO_PROXY_HOSTS} && \
    git config --global http.version HTTP/1.1 && \
    git config --global --add safe.directory '*' && \
    export PTO_ISA_ROOT=${WS}/pto-isa && \
    ${PIP} install --no-cache-dir --no-build-isolation -e ${WS}/pypto/runtime && \
    cd ${WS}/pypto && ${PYTHON} -m simpler_setup.build_runtimes --platforms a2a3

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

# ---- 6) 修 base 镜像 cann-8.5.1 残留 + runtime env --------------------------
# base 镜像把 CANN 8.5.1 设为默认 (ENV + /etc/profile:29 都硬编码 cann-8.5.1)。
# 删了 8.5.1 后这些悬空引用会让每个 bash 登录 shell 报错退出 (rc=1)。
# 覆盖 baked 标量 ASCEND 环境 -> beta.1 (PATH/LD 里 8.5.1 死条目无害, 由 set_env 覆盖)。
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0-beta.1 \
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0-beta.1 \
    ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.0.0-beta.1 \
    ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0-beta.1/opp \
    TOOLCHAIN_HOME=/usr/local/Ascend/cann-9.0.0-beta.1/toolkit
ENV PYPTO_PROG_BUILD_DIR=/tmp/pypto_build_output
ENV PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072
# /etc/profile:29 hardcode 的 cann-8.5.1 set_env.sh -> beta.1 (beta.1 有同名文件),
# 让登录 shell 不再因缺文件 abort; 然后写 pypto 专属 env (login shell 生效)。
RUN sed -i 's#/usr/local/Ascend/cann-8.5.1#/usr/local/Ascend/cann-9.0.0-beta.1#g' /etc/profile 2>/dev/null || true; \
    printf '%s\n' \
  'source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh' \
  'export WS=/workspace' \
  'export PTO_ISA_ROOT=$WS/pto-isa PTOAS_ROOT=$WS/ptoas-bin' \
  'export LD_LIBRARY_PATH=$WS/ptoas-bin/lib:$LD_LIBRARY_PATH' \
  'export PATH=$WS/ptoas-bin/bin:/usr/local/python3.11.14/bin:$PATH' \
  'export PYTHONPATH=$WS/vllm-pypto:$WS/pypto/python:$WS/pypto/runtime/python:$PYTHONPATH' \
  'export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072' \
  > /etc/profile.d/pypto-env.sh

WORKDIR ${WS}
# base 镜像的 ENTRYPOINT 在 && 链里 hardcode source cann-8.5.1/.../ascendnpu-ir/set_env.sh,
# 删了 8.5.1 后该 source 失败 -> && 链中断 -> exec "$@" 不执行 -> 容器 rc=1 起不来。
# 覆盖为 beta.1 (有同名文件); 另两处 (ascend-toolkit/latest->cann->beta.1, nnal/atb) 本就 OK。
ENTRYPOINT ["/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/cann-9.0.0-beta.1/share/info/ascendnpu-ir/bin/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && exec \"$@\"", "--"]
CMD ["/bin/bash", "-l"]
