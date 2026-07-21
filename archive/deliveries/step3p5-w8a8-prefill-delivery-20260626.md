# Step3p5 W8A8 prefill precision report

Date: 2026-06-26
Host: `gpu-a910x-0162.host.platform.shaipower.com`

## Model and service

- Checkpoint: `/mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`
- vLLM image: `hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938`
- vLLM mode: W8A8 (`--quantization ascend`), eager, TP=EP=8, NPU `8..15`, port `8001`
- Golden root: `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/golden_step3p5_w8a8_prefill_vllm_sampled`
- PyPTO report root: `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision`

## Code changes

- `tools/step3p5/collect_w8a8_prefill_golden.py`
  - collects W8A8 vLLM prefill detail golden for multiple sequence lengths;
  - expects the Step3p5 vLLM debug dump hook with sampled/pruned dumping enabled.
- `tools/step3p5/prefill_precision_suite.py`
  - validates multi-length prefill detail and final logits;
  - emits JSON/Markdown reports and packages artifacts.
- `tests/step3p5/test_step3p5_w8a8_prefill_st.py`
  - pytest ST wrapper for precomputed reports or live golden replay.
- `docs/step3p5-w8a8-prefill-delivery.md`
  - delivery summary and reproduction notes.

PyPTO code commit: `pypto-lib` `81252e9` (`test(step3p5): add w8a8 prefill precision suite`).

## Golden data

Prefill golden was collected with sampled detail rows because full 128k layer dumps are too large. The dump keeps tensors required by the PyPTO detail/final-logits comparators and samples up to 128 rows per forward.

Covered cases:

| Case | Seq len | Dump files |
|---|---:|---:|
| `prefill_1k` | 1024 | 3232 |
| `prefill_4k` | 4096 | 3232 |
| `prefill_8k` | 8192 | 3232 |
| `prefill_32k` | 32768 | 3232 |
| `prefill_64k` | 65536 | 3232 |
| `prefill_128k` | 131072 | 6464 |

## Validation results

Acceptance for sampled W8A8-prefill detail is `pass_rate >= 0.997`; final logits must pass for every case.

| Case | Seq len | Detail | Final logits | Worst pass rate |
|---|---:|---|---|---:|
| `prefill_1k` | 1024 | PASS | PASS | 0.999349 |
| `prefill_4k` | 4096 | PASS | PASS | 0.998698 |
| `prefill_8k` | 8192 | PASS | PASS | 0.999023 |
| `prefill_32k` | 32768 | PASS | PASS | 0.999349 |
| `prefill_64k` | 65536 | PASS | PASS | 0.999756 |
| `prefill_128k` | 131072 | PASS | PASS | 0.997559 |

Pytest:

```bash
STEP3P5_PREFILL_REPORT_ROOT=/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision \
PYTHONPATH=. pytest -q tests/step3p5/test_step3p5_w8a8_prefill_st.py
# 1 passed in 0.01s
```

## Artifact package

- Tar: `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/step3p5_w8a8_prefill_regression_20260626.tar`
- Size: `20G`
- SHA256: `cd34f034e017c68437547e5f7f453a2f6b481a1e97e162a89ac21c422fe76b6e`

## Reproduction

```bash
cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
source ../activate.sh

PYTHONPATH=. python tools/step3p5/prefill_precision_suite.py \
  --golden-root /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/golden_step3p5_w8a8_prefill_vllm_sampled \
  --ckpt-dir /mnt/nvme1/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --output-root /mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/pypto_prefill_precision \
  --seq-len 1024 --seq-len 4096 --seq-len 8192 \
  --seq-len 32768 --seq-len 65536 --seq-len 131072 \
  --max-detail-tokens 16 \
  --mlp-atol 1.0 --mlp-rtol 5e-3 --pass-rate 0.997
```
