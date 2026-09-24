# Programme Studio handover

## Services

| Service | Source | Port | Purpose |
| --- | --- | --- | --- |
| Prompt Service | `prompt-service` | 8001 | Validates and freezes briefs, generates directions A/B/C with Qwen, Gemma, and Mistral, then hands off automatically. |
| Image Service | `image-service` | 8002 | Runs the engine-first workflow: FLUX.2 A/B/C, SDXL 1.0 A/B/C, then SD 3.5 Medium A/B/C; composes, validates and publishes. |
| Repository Service | `repository-service` | 8003 | Stores manifests and PNG artifacts in Jackrabbit Oak and serves retrieval endpoints. |

Build and run commands are in `podman-runbook.md`. The Prompt and Image
services expose interactive OpenAPI documents at `/docs` while running:

```text
http://127.0.0.1:8001/docs
http://127.0.0.1:8002/docs
```

## Main API flow

1. `POST /v1/prompt-jobs` accepts a brief and returns `202 Accepted` with a
   durable reference.
2. The Prompt Service writes `brief.json`, direction checkpoints and
   `prompts.json`, then calls `POST /v1/image-jobs/{reference}/workflow`.
3. The Image Service creates the ordered nine-candidate plan, generates and
   validates backgrounds, composes A4 PNGs, writes `manifest.json`, uploads
   artifacts and verifies repository retrieval.
4. `GET /v1/prompt-jobs/{reference}` and
   `GET /v1/image-jobs/{reference}/workflow` report durable workflow state.

## Reproducible evidence

Run the repository-restart and acceptance checks:

```bash
podman restart programme-repository
sleep 5
python3 infrastructure/verify_acceptance.py
```

The checker verifies all three agreed samples, all 27 stored PNGs, their
checksums and the three service readiness endpoints. See
`acceptance-evidence.md` for the sample references.

Export the live Prompt and Image Service OpenAPI definitions with
`bash infrastructure/export_openapi.sh`. The Repository Service contract is
versioned at `repository-service/openapi.yaml`.

Capture the configured Ollama model digests, ComfyUI node inventory and local
workflow hashes with `python3 infrastructure/capture_runtime_inventory.py`.
See `model-registry.md` for the captured active models, workflow hashes and an
explicit record of the remaining model-version and GPU-host evidence gap.

To demonstrate an intentional interruption and recovery for a fresh accepted
reference, use `verify_resume.py`. Its header documents the stop, restart,
resume and artifact-stability sequence. Run `capture_gpu_host_evidence.py` on
the GPU host (with `COMFYUI_MODELS_DIR` set when required) to capture VRAM,
weight file hashes and the licence-approval record.

## Reviewer inputs still required

Supply approved client headshot and logo files when submitting a final brief;
the services reject retired demo-asset paths. A human reviewer must also
proof-print one final PNG and record the visual-quality scores required by the
acceptance checklist.
