# Programme Studio Podman runbook

The application has exactly three services. Ollama and ComfyUI run on the
separate GPU host and are infrastructure, not containers in this stack.

Create the shared network once:

```bash
podman network create programme-studio
```

Build the three images from their service directories:

```bash
podman build -t localhost/programme-prompt-service:1.0 prompt-service
podman build -t localhost/programme-image-service:1.0 image-service
podman build -t localhost/programme-repository-service:1.0 repository-service
```

Run the repository first. Its `/data/repository` bind mount is its persistent
Jackrabbit Oak store.

```bash
podman run -d --name programme-repository --network programme-studio -p 8003:8003 \
  -v /data/repository:/data/repository:rw \
  localhost/programme-repository-service:1.0
```

Run the prompt service. Its programme mount contains frozen briefs, direction
checkpoints, and `prompts.json`.

```bash
podman run -d --name programme-prompt --network programme-studio -p 8001:8001 \
  -v /data/programmes:/data/programmes:rw \
  -e PROGRAMME_DATA_PATH=/data/programmes \
  -e OLLAMA_BASE_URL=http://192.168.68.115:11434 \
  -e DIRECTION_A_MODEL=qwen2.5:14b \
  -e DIRECTION_B_MODEL=gemma3:27b \
  -e DIRECTION_C_MODEL=mistral-small3.1:24b \
  -e COMFYUI_BASE_URL=http://192.168.68.115:8188 \
  -e IMAGE_SERVICE_BASE_URL=http://programme-image:8002 \
  localhost/programme-prompt-service:1.0
```

Run the image service. The repository URL must use Podman DNS, never
`localhost`.

```bash
podman run -d --name programme-image --network programme-studio -p 8002:8002 \
  -v /data/programmes:/data/programmes:rw \
  -e PROGRAMME_DATA_PATH=/data/programmes \
  -e COMFYUI_BASE_URL=http://192.168.68.115:8188 \
  -e REPOSITORY_SERVICE_BASE_URL=http://programme-repository:8003 \
  localhost/programme-image-service:1.0
```

Check service readiness:

```bash
curl http://localhost:8001/ready
curl http://localhost:8002/ready
curl http://localhost:8003/ready
```

Image generation is always engine-first: FLUX.2 A/B/C, SDXL 1.0 A/B/C, then
Stable Diffusion 3.5 Medium A/B/C. The image service uses ComfyUI HTTP
endpoints `/object_info`, `/prompt`, `/history/{prompt_id}`, and `/view`.
It releases the current ComfyUI model before switching engines.

The historical specification mentioned FLUX.1-schnell. The configured GPU
runtime provides FLUX.2 instead, so this implementation uses FLUX.2 while
retaining the three-engine, three-direction, nine-output design.

The Prompt Service readiness endpoint requires the three contracted Ollama
tags above. It returns HTTP 503 until all three are installed on the GPU host.
