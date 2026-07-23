#!/bin/bash
# vllm-pypto image smoke test — validates the baked pypto stack loads on this host.
# Run inside the container with a login shell so env is ready:
#   nerdctl run --rm --net host --security-opt apparmor=unconfined \
#     --device /dev/davinci8 --device /dev/davinci_manager \
#     --device /dev/hisi_hdc --device /dev/devmm_svm \
#     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
#     <IMG> bash -lc 'bash /workspace/pypto-smoke.sh'
set -e
echo "[smoke] ptoas   : $(ptoas --version 2>&1 | head -1)"
python -c "import pypto; print('[smoke] pypto   :', getattr(pypto,'__version__','?'))"
python -c "import simpler; print('[smoke] simpler : OK')"
ls /workspace/pypto/runtime/build/lib/a2a3/dispatcher/*.so >/dev/null 2>&1 \
  && echo "[smoke] runtime : $(ls /workspace/pypto/runtime/build/lib/a2a3/dispatcher/*.so | head -1)"
test -d /workspace/vllm-pypto/tests/step3p5/ci && echo "[smoke] vllm-pypto CI dir: OK"
echo "[smoke] PASS"
