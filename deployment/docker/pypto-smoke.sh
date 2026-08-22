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
python -c "import os; from models.step3p5 import config as c; expected=os.environ.get('PYPTO_STEP3P5_ATTN_TASK_PROFILE','portable'); assert c.ATTN_TASK_PROFILE == expected, (c.ATTN_TASK_PROFILE, expected); print('[smoke] attention profile:', c.ATTN_TASK_PROFILE)"
python - <<'PY'
import os
from dataclasses import fields

from pypto.runtime.runner import RunConfig

requirement = os.environ.get("PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN")
assert requirement in (None, "0", "1"), requirement
names = {field.name for field in fields(RunConfig)}
required = {
    "enable_chip_swimlane",
    "enable_dep_gen",
    "l2_swimlane_reuse_dep_gen",
}
available = required <= names
constructed = False
if available:
    config = RunConfig(
        platform="a2a3",
        enable_chip_swimlane=True,
        enable_dep_gen=False,
        l2_swimlane_reuse_dep_gen=True,
    )
    constructed = config.l2_swimlane_reuse_dep_gen is True
if requirement == "1":
    assert available and constructed
print(
    "[smoke] l2 swimlane dep reuse:",
    {"available": available, "constructed": constructed, "required": requirement},
)
PY
ls /workspace/pypto/runtime/build/lib/a2a3/dispatcher/*.so >/dev/null 2>&1 \
  && echo "[smoke] runtime : $(ls /workspace/pypto/runtime/build/lib/a2a3/dispatcher/*.so | head -1)"
test -d /workspace/vllm-pypto/tests/step3p5/ci && echo "[smoke] vllm-pypto CI dir: OK"
echo "[smoke] PASS"
