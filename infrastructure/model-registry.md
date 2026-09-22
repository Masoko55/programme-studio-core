# Runtime model registry

Captured on 22 September 2026 using `capture_runtime_inventory.py`. The source
of record is `evidence/runtime-inventory.json`; it preserves the full remote
Ollama inventory and local ComfyUI workflow hashes.

## Active prompt-direction configuration

| Direction | Configured model | Host-reported size | SHA-256 digest |
| --- | --- | ---: | --- |
| A | `qwen3:14b` | 14.8B | `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8` |
| B | `gemma3:latest` | 4.3B | `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a` |
| C | `mistral-small3.2:24b` | 24.0B | `5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b` |

## Image workflow files

| Engine | Workflow file | SHA-256 |
| --- | --- | --- |
| FLUX.2 | `image-service/workflows/flux2.json` | `3e884f4586a26c76563bfafb4fcee9b32f1aa9b033ebac6e507d4363aca5cb64` |
| SDXL 1.0 | `image-service/workflows/sdxl.json` | `8b06e6351f552277e42db04d47707cfac89b31c983ef3e9950c612cb9cc99ec1` |
| SD 3.5 Medium | `image-service/workflows/sd35-medium.json` | `8eb918621814cad619eb2bf3959e7b80512cc7e307507fdf2820a208441b2aa3` |

The ComfyUI HTTP API exposes installed node types but does not expose model
weight file paths or their checksums. The GPU-host operator must supply those
file digests and confirm the applicable commercial licences before final
handover.

## Specification variance requiring resolution

The requested model set names Qwen 2.5 14B, Gemma 3 27B and Mistral Small 3.1
24B. The active configuration above is a different set: Qwen 3 14B, Gemma 3
4.3B and Mistral Small 3.2 24B. The recorded demonstration proves three
independent models ran, but it does **not** prove compliance with those exact
requested model versions. Resolve the variance by either installing and
configuring the named models, then rerunning the three samples, or obtaining
written approval for these substitutions.
