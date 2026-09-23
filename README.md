# Programme Studio

Programme Studio produces print-ready A4 event programmes through three local
services:

| Service | Port | Responsibility |
| --- | ---: | --- |
| Prompt Service | 8001 | Validates and freezes an event brief, creates directions A–C with Qwen, Gemma and Mistral, then starts image generation. |
| Image Service | 8002 | Runs FLUX.2, SDXL 1.0 and SD 3.5 Medium for each direction; composes, validates and publishes nine PNGs. |
| Repository Service | 8003 | Persists manifests and final PNG artifacts in Jackrabbit Oak. |

## Run locally

Follow [the Podman runbook](infrastructure/podman-runbook.md) to build and
start the three services. The Prompt and Image Service API documents are then
available at:

```text
http://127.0.0.1:8001/docs
http://127.0.0.1:8002/docs
```

## Verify delivery

With all services ready, verify the required funeral, celebration and 15-row
programme samples, their nine artifacts each, and SHA-256 checksums:

```bash
python3 infrastructure/verify_acceptance.py
```

Export the live API schemas with:

```bash
bash infrastructure/export_openapi.sh
```

## Handover material

- [Handover guide](infrastructure/handover.md)
- [Acceptance evidence](infrastructure/acceptance-evidence.md)
- [Model registry](infrastructure/model-registry.md)
- [QA review sheet](infrastructure/qa-review.md)

Before a client delivery, replace the demonstration assets in
`/data/programmes/demo-assets`, complete the human visual/proof-print review,
and capture GPU-host model-file hashes and licence evidence.
