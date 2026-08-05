# =============================================================================
# vllm-pypto 可复现环境 (Ascend 910B / step3p5)
#
# 0724 blessed image 的固定环境 + canonical-only Step3p5 loop-form Main。
# 构建规格由 builds/*.env 提供；所有仓 pin 都必须通过 build args 传入，
# 禁止把可变的 stepfun/develop branch 当作 release source。
# =============================================================================
ARG BASE=hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938
FROM ${BASE}

SHELL ["/bin/bash", "-lc"]

# Release pins are overridden by the selected build spec.
ARG PYPTO_COMMIT=ca21ab5fcfd8203165928428302d273c377db5c6
ARG PYPTO_LIB_COMMIT=53eb7212c29c9bd015ee060cd9924a13ea781ae0
ARG PTO_ISA_COMMIT=ecb6c303f797749f811a494742c3c08156aacabb
ARG PTOAS_COMMIT=fc8c6caee561914b4fb991dfc8427bb63194269e
ARG SIMPLER_COMMIT=216e7632267ae815c484cdeba7991c87fabf3086
ARG PTOAS_BIN_VER=v0.50
ARG ATTN_TASK_PROFILE=portable
ARG BUILD_JOBS=2
ARG VLLM_PATCH_BRANCH=csy/pypto-tail-mtp-integration
ARG VLLM_PATCH_COMMIT=1b3e538c35999e62b6d24e0651b3a85b7d16c826
ARG GH_PROXY=http://proxy.i.shaipower.com:3128
ARG NO_PROXY_HOSTS=basemind.com,shaipower.com,shaipower,127.0.0.1,localhost,.svc

ENV WS=/workspace
ENV PYTHON=/usr/local/python3.11.14/bin/python3
ENV PIP="/usr/local/python3.11.14/bin/python3 -m pip"

# Keep only the CANN release used by the 0724 image.
RUN rm -rf /usr/local/Ascend/cann-8.5.1 && \
    ln -sfn /usr/local/Ascend/cann-9.0.0-beta.1 /usr/local/Ascend/cann

# Clone the complete stack and checkout immutable pins. pypto-lib and
# vllm-pypto intentionally point at the same current replacement commit.
# Scrub every credential-bearing clone URL before this layer is committed.
WORKDIR ${WS}
RUN --mount=type=secret,id=gh_token \
    export https_proxy=${GH_PROXY} http_proxy=${GH_PROXY} no_proxy=${NO_PROXY_HOSTS} && \
    GH="$(cat /run/secrets/gh_token)" && \
    for r in pypto pypto-lib pto-isa PTOAS; do \
      git -c http.version=HTTP/1.1 clone "https://${GH}@github.com/csy0225/${r}.git" "${WS}/${r}"; \
    done && \
    cd ${WS}/pypto && git checkout ${PYPTO_COMMIT} && \
      git config submodule.runtime.url "https://${GH}@github.com/csy0225/simpler.git" && \
      git submodule update --init runtime && \
      ( cd runtime && git fetch -q origin ${SIMPLER_COMMIT} && git checkout ${SIMPLER_COMMIT} ) && \
    cd ${WS}/pypto-lib && git fetch -q "https://${GH}@github.com/csy0225/pypto-lib.git" ${PYPTO_LIB_COMMIT} && git checkout ${PYPTO_LIB_COMMIT} && \
    cd ${WS}/pto-isa && git fetch -q "https://${GH}@github.com/csy0225/pto-isa.git" ${PTO_ISA_COMMIT} && git checkout ${PTO_ISA_COMMIT} && \
    cd ${WS}/PTOAS && git fetch -q "https://${GH}@github.com/csy0225/PTOAS.git" ${PTOAS_COMMIT} && git checkout ${PTOAS_COMMIT} && \
    ln -sfn ${WS}/pypto-lib ${WS}/vllm-pypto && \
    git -C ${WS}/pypto config --local submodule.runtime.url "https://github.com/csy0225/simpler.git" && \
    git -C ${WS}/pypto remote set-url origin "https://github.com/csy0225/pypto.git" && \
    git -C ${WS}/pypto/runtime remote set-url origin "https://github.com/csy0225/simpler.git" && \
    git -C ${WS}/pypto-lib remote set-url origin "https://github.com/csy0225/pypto-lib.git" && \
    git -C ${WS}/pto-isa remote set-url origin "https://github.com/csy0225/pto-isa.git" && \
    git -C ${WS}/PTOAS remote set-url origin "https://github.com/csy0225/PTOAS.git" && \
    test "$(git -C ${WS}/pypto-lib rev-parse HEAD)" = "${PYPTO_LIB_COMMIT}" && \
    test "$(git -C ${WS}/pypto/runtime rev-parse HEAD)" = "${SIMPLER_COMMIT}"

# ptoas-bin is the v0.50 payload verified with the 0724 environment.
COPY ptoas-bin.tgz /tmp/ptoas-bin.tgz
RUN mkdir -p ${WS}/ptoas-bin && \
    tar xzf /tmp/ptoas-bin.tgz -C ${WS}/ptoas-bin && rm -f /tmp/ptoas-bin.tgz && \
    LD_LIBRARY_PATH=${WS}/ptoas-bin/lib ${WS}/ptoas-bin/bin/ptoas --version

# Build pypto and simpler runtime with the pinned CANN/toolchain.
ENV PTO_ISA_ROOT=${WS}/pto-isa
ENV PTOAS_ROOT=${WS}/ptoas-bin
ENV LD_LIBRARY_PATH=${WS}/ptoas-bin/lib:${LD_LIBRARY_PATH}
ENV PATH=${WS}/ptoas-bin/bin:${PATH}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV MAX_JOBS=${BUILD_JOBS}
RUN source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh && \
    export https_proxy=${GH_PROXY} http_proxy=${GH_PROXY} no_proxy=${NO_PROXY_HOSTS} && \
    git config --global http.version HTTP/1.1 && \
    git config --global --add safe.directory '*' && \
    export PTO_ISA_ROOT=${WS}/pto-isa && \
    ${PIP} install --no-cache-dir "cmake==3.31.6" \
        "scikit-build-core>=0.10.0" "nanobind>=2.0.0" "ninja>=1.11.0" && \
    ${PIP} install --no-cache-dir --no-build-isolation -e ${WS}/pypto
# simpler has three nested concurrency levels: runtime variants, host/aicpu/
# aicore targets, and each target's cmake build. Bridge all three to the
# release build setting only while compiling, then restore the pinned source
# so the image stays clean.
RUN runtime_compiler=${WS}/pypto/runtime/simpler_setup/runtime_compiler.py && \
    build_runtimes=${WS}/pypto/runtime/simpler_setup/build_runtimes.py && \
    runtime_builder=${WS}/pypto/runtime/simpler_setup/runtime_builder.py && \
    grep -Fq 'str(min(multiprocessing.cpu_count(), 32)),' "${runtime_compiler}" && \
    grep -Fq 'ThreadPoolExecutor(max_workers=len(tasks) or 1)' "${build_runtimes}" && \
    grep -Fq 'ThreadPoolExecutor(max_workers=4)' "${runtime_builder}" && \
    sed -i \
      's#str(min(multiprocessing.cpu_count(), 32)),#os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(min(multiprocessing.cpu_count(), 32))),#' \
      "${runtime_compiler}" && \
    sed -i \
      's#ThreadPoolExecutor(max_workers=len(tasks) or 1)#ThreadPoolExecutor(max_workers=min(len(tasks) or 1, int(os.environ.get("MAX_JOBS", str(len(tasks) or 1)))))#' \
      "${build_runtimes}" && \
    sed -i \
      's#ThreadPoolExecutor(max_workers=4)#ThreadPoolExecutor(max_workers=min(4, int(os.environ.get("MAX_JOBS", "4"))))#' \
      "${runtime_builder}" && \
    source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh && \
    export https_proxy=${GH_PROXY} http_proxy=${GH_PROXY} no_proxy=${NO_PROXY_HOSTS} && \
    git config --global http.version HTTP/1.1 && \
    git config --global --add safe.directory '*' && \
    export PTO_ISA_ROOT=${WS}/pto-isa && \
    ${PIP} install --no-cache-dir --no-build-isolation -e ${WS}/pypto/runtime && \
    cd ${WS}/pypto && ${PYTHON} -m simpler_setup.build_runtimes --platforms a2a3 && \
    git -C ${WS}/pypto/runtime checkout -- \
      simpler_setup/runtime_compiler.py \
      simpler_setup/build_runtimes.py \
      simpler_setup/runtime_builder.py && \
    git -C ${WS}/pypto/runtime diff --exit-code -- \
      simpler_setup/runtime_compiler.py \
      simpler_setup/build_runtimes.py \
      simpler_setup/runtime_builder.py

# vLLM Track-B patch (tail-only Main + MTP proposer/boot fixes).
RUN --mount=type=secret,id=gl_token \
    GL="$(cat /run/secrets/gl_token)" && \
    tmp=$(mktemp -d) && \
    git -c http.version=HTTP/1.1 init "$tmp/vllm" && \
    git -C "$tmp/vllm" remote add origin \
      "https://oauth2:${GL}@gitlab.basemind.com/sys/stepcast/vllm.git" && \
    git -C "$tmp/vllm" -c http.version=HTTP/1.1 fetch --depth 1 origin ${VLLM_PATCH_COMMIT} && \
    git -C "$tmp/vllm" checkout --detach ${VLLM_PATCH_COMMIT} && \
    test "$(git -C "$tmp/vllm" rev-parse HEAD)" = "${VLLM_PATCH_COMMIT}" && \
    cp "$tmp/vllm/vllm/model_executor/models/step3p5.py"     /vllm-workspace/vllm/vllm/model_executor/models/step3p5.py && \
    cp "$tmp/vllm/vllm/model_executor/models/step3p5_mtp.py" /vllm-workspace/vllm/vllm/model_executor/models/step3p5_mtp.py && \
    rm -rf "$tmp"

# Runtime environment and base-image CANN 8.5.1 cleanup.
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0-beta.1 \
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0-beta.1 \
    ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.0.0-beta.1 \
    ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0-beta.1/opp \
    TOOLCHAIN_HOME=/usr/local/Ascend/cann-9.0.0-beta.1/toolkit
ENV PATH=${WS}/ptoas-bin/bin:/usr/local/python3.11.14/bin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/cann-9.0.0-beta.1/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/ccec_compiler/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=${WS}/vllm-pypto:${WS}/pypto/python:${WS}/pypto/runtime/python:/usr/local/Ascend/cann-9.0.0-beta.1/python/site-packages:/usr/local/Ascend/cann-9.0.0-beta.1/opp/built-in/op_impl/ai_core/tbe \
    CMAKE_PREFIX_PATH=/usr/local/Ascend/cann-9.0.0-beta.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-9.0.0-beta.1/lib64/cmake
ENV PYPTO_PROG_BUILD_DIR=/tmp/pypto_build_output
ENV PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072
ENV PYPTO_STEP3P5_ATTN_TASK_PROFILE=${ATTN_TASK_PROFILE}
RUN for name in \
      PYPTO_STEP3P5_FULL_ATTN_QK_BLOCKS_PER_TASK \
      PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_BLOCKS_PER_TASK \
      PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_BLOCKS_PER_TASK \
      PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_PARTIALS_PER_REDUCE_TASK \
      PYPTO_STEP3P5_FULL_ATTN_QK_UNIFORM_O1 \
      PYPTO_STEP3P5_FULL_ATTN_SOFTMAX_UNIFORM_O1 \
      PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_UNIFORM_O1 \
      PYPTO_STEP3P5_FULL_ATTN_ONLINE_SOFTMAX_REDUCE_UNIFORM_O1; do \
      test -z "${!name+x}"; \
    done
RUN ln -sfn bin/ptoas ${WS}/ptoas-bin/ptoas && \
    sed -i 's#/usr/local/Ascend/cann-8.5.1#/usr/local/Ascend/cann-9.0.0-beta.1#g' /etc/profile 2>/dev/null || true; \
    printf '%s\n' \
  'source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh' \
  'export WS=/workspace' \
  'export PTO_ISA_ROOT=$WS/pto-isa PTOAS_ROOT=$WS/ptoas-bin' \
  'export LD_LIBRARY_PATH=$WS/ptoas-bin/lib:$LD_LIBRARY_PATH' \
  'export PATH=$WS/ptoas-bin/bin:/usr/local/python3.11.14/bin:$PATH' \
  'export PYTHONPATH=$WS/vllm-pypto:$WS/pypto/python:$WS/pypto/runtime/python:$PYTHONPATH' \
  'export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072' \
  > /etc/profile.d/pypto-env.sh; \
    test ! -e /usr/local/Ascend/cann-8.5.1; \
    ! env | grep -q '/usr/local/Ascend/cann-8\.5\.1'; \
    ! grep -Rqs '/usr/local/Ascend/cann-8\.5\.1' \
        /etc/profile /etc/profile.d /root/.bashrc /root/.profile 2>/dev/null

WORKDIR ${WS}
COPY pypto-smoke.sh /workspace/pypto-smoke.sh
ENTRYPOINT ["/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/cann-9.0.0-beta.1/share/info/ascendnpu-ir/bin/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && exec \"$@\"", "--"]
CMD ["/bin/bash", "-l"]
