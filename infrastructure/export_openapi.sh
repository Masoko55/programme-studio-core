#!/usr/bin/env bash
set -euo pipefail

mkdir -p infrastructure/openapi
curl --fail --silent --show-error http://127.0.0.1:8001/openapi.json \
  -o infrastructure/openapi/prompt-service.json
curl --fail --silent --show-error http://127.0.0.1:8002/openapi.json \
  -o infrastructure/openapi/image-service.json
printf 'Exported infrastructure/openapi/prompt-service.json\n'
printf 'Exported infrastructure/openapi/image-service.json\n'
