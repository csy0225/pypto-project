# Step3p5 W8A8 end-to-end precision report

Date: 2026-06-26
Host: gpu-a910x-0162.host.platform.shaipower.com

## Model and service

- Checkpoint: `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`
- vLLM image: `hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938`
- vLLM command: eager mode, TP=EP=8, `--quantization ascend`, `--enable-expert-parallel`, NPU `8..15`, port `8001`
- Golden root: `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/golden_step3p5_w8a8_vllm_20260626_004648`

## Code changes

- `models/step3p5/weight_loader.py`
  - supports `quant_model_weights.safetensors.index.json`
  - detects W8A8_DYNAMIC routed experts
  - dequantizes per-expert INT8 `weight` using per-output-row `weight_scale` and optional `weight_offset`
  - preserves the existing BF16 PyPTO bundle layout for kernels and reference tools
- `tools/step3p5/decode_acceptance.py`
  - accepts BF16 and W8A8 index layouts and reports `is_w8a8_dynamic`
- `tools/step3p5/pypto_all_layers_detail_compare.py`
  - auto-enables routed-MoE W8A8 dynamic activation quantization for W8A8 checkpoints
  - compares PyPTO reference against the newly dumped W8A8 vLLM detail tensors
- Tests:
  - `tests/step3p5/test_weight_loader_w8a8.py`
  - `tests/step3p5/test_step3p5_w8a8_e2e_st.py`

## Golden data

Collected by redeploying vLLM with the W8A8 checkpoint, not by reusing BF16 dumps.
The request is `beijing_1tok` (`max_tokens=1`, temperature 0).  Dump includes model input, per-layer detail tensors for layers 0..44, MTP decoder-layer tensors emitted by vLLM, and final `main_logits`.

- Manifest: `golden_step3p5_w8a8_vllm_20260626_004648/manifest.json`
- Response: `golden_step3p5_w8a8_vllm_20260626_004648/beijing_1tok/response.json`
- Dump files: 5944 `.pt` tensors

## Validation results

| Gate | Report | Result |
|---|---|---|
| W8A8 weight acceptance | `decode_acceptance_w8a8_rank0.json` | PASS, `ok=true`, 48 layers observed |
| Main 45-layer detail alignment | `pypto_all_layers_detail_compare_w8a8_beijing1_atol1_report.json` | PASS, 3960 checks, worst pass rate `0.9995659589767456` |
| Final logits e2e | `pypto_final_logits_from_vllm_w8a8/final_logits_report.json` | PASS, full-vocab logits pass rate `1.0`, argmax token match (`3648`) |
| ST tests | `pytest -q tests/step3p5/test_weight_loader_w8a8.py tests/step3p5/test_step3p5_w8a8_e2e_st.py` | PASS, `6 passed in 1.30s` |

Tolerance note: W8A8 routed MoE is validated with BF16-visible tensor dumps and dynamic activation quantization in the PyPTO reference.  The main detail gate uses `mlp_atol=1.0`, `mlp_rtol=5e-3`, `pass_rate>=0.999`; non-MLP tensors use `atol=5e-3`, `rtol=5e-3`.

## Reproduction commands

```bash
cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
source ../activate.sh
PYTHONPATH=. python tools/step3p5/decode_acceptance.py \
  --ckpt-dir /mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --rank 0 --tp-world-size 8 --batch 1 --json

PYTHONPATH=. python tools/step3p5/pypto_all_layers_detail_compare.py \
  --dump-root /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/golden_step3p5_w8a8_vllm_20260626_004648/beijing_1tok/dump \
  --ckpt-dir /mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --tp-world-size 8 --rtol 5e-3 --atol 5e-3 \
  --mlp-rtol 5e-3 --mlp-atol 1.0 --pass-rate 0.999

PYTHONPATH=. pytest -q tests/step3p5/test_weight_loader_w8a8.py tests/step3p5/test_step3p5_w8a8_e2e_st.py
```
