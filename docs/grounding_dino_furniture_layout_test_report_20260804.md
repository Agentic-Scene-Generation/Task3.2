# GroundingDINO Furniture Layout Test Report (2026-08-04)

## Scope

- Baseline: `dev_ljx@e7efd590fe19c2ea3ee0306e8fb73ad815cd992d`
- Implementation branch: `feat/grounding-dino-furniture-layout`
- GPU node: 2 × NVIDIA H100 80GB HBM3
- GroundingDINO weights:
  `/mnt/afs-p3/task3_2/visitor33_ljx/checkpoints/grounding-dino-base`
- VLM launcher: `/mnt/afs/visitor33/Task3.2/start_llama.sh`
- VLM model: `unsloth/Qwen3.6-27B-GGUF` (27B Q8 GGUF + F16 vision projector)
- Test image: 768 × 768 final furniture context image from the existing
  `formal_dining_gallery` output.

## GPU results

| Service / phase | GPU 0 | GPU 1 | Latency |
|---|---:|---:|---:|
| GroundingDINO ready, process memory | ~1,435 MiB | 0 MiB | model load ~32 s from AFS |
| GroundingDINO 115-phrase inference peak | 3,113 MiB | 0 MiB | 2.533 s request |
| GroundingDINO post-inference cached state | 3,104 MiB | 0 MiB | — |
| Qwen3.6-27B VLM ready, process memory | 22,904 MiB | 26,506 MiB | model load ~134 s from AFS |
| Qwen3.6-27B single-image inference peak | 23,085 MiB | 26,605 MiB | 2.692 s |
| Combined grounded-layout E2E peak | 26,229 MiB | 26,609 MiB | 29.813 s wall time |

GroundingDINO health reported 890.02 MiB model tensor allocation and 910 MiB
reserved immediately after load. The server keeps FP32 weights and uses FP16 CUDA
autocast. Directly converting the full HF model to FP16 was rejected during smoke
testing because an internal fusion path returns FP32 tensors; autocast resolved the
mixed-dtype matmul while remaining well within the H100 memory budget.

The fixed 115-phrase vocabulary was split by the model tokenizer exactly as planned:

- batch 1: 79 phrases, 239 tokens, 36 raw detections, 2,306.555 ms model time;
- batch 2: 36 phrases, 111 tokens, 2 raw detections, 117.123 ms model time.

## Full grounded-layout E2E

The final bounded test ran the actual sidecar and actual `VLMService` against the
Qwen3.6 endpoint. It completed the complete image-analysis path:

1. two tokenizer-safe GroundingDINO batches;
2. clipping, duplicate-region merging, stable IDs and annotation rendering;
3. original image + annotation image + region JSON VLM request;
4. one allowed coverage reground for `dining chair`;
5. a second and final VLM analysis;
6. strict normalization and language-only Designer contract generation.

Final audit summary:

- merged regions: 20;
- normalized Furniture-stage items: 8;
- VLM calls: 2;
- coverage regrounds: 1 (hard limit respected);
- fallback reason: `null`;
- explicit image-top/image-left/viewer-relative text in contract: false;
- combined peak memory: 26,229 MiB on GPU 0 and 26,609 MiB on GPU 1.

The later final text filter also rejects `foreground`/`background` camera-relative
notes; this small normalization change is covered by unit tests and does not alter
model calls or GPU measurements.

The E2E produced and validated all three runtime artifacts:
`context_grounding_raw.json`, `context_grounding_annotated.png`, and
`context_furniture_layout.json`. Transient test copies were removed after their
metrics and validation summary were recorded here.

## Automated tests

- Final combined regression command: **48 passed** in 61.24 seconds (4 existing
  warnings).
- Grounding client, batching, region merge, annotation, VLM normalization, cache,
  orchestration and sidecar tests: passed.
- Existing context-image generation, context-image quality gate and legacy
  placement-order regressions: passed.
- Rectangle/polygon adapter, authoritative polygon prompt and polygon-only repair
  isolation checks: 5 passed.

The full Blender-backed rectangle and polygon scene-generation runs were not launched
from this shell because its Blender import is missing `libXrender.so.1`. The feature
itself accepts no room-coordinate or room-shape input, and the geometry regression
tests confirm that authoritative rectangle/polygon handling remains isolated.
