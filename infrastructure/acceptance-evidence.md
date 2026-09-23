# Acceptance evidence

Run `python3 infrastructure/verify_acceptance.py` after starting the three
Podman services. It verifies these completed API-driven samples:

| Requirement | Reference | Evidence |
| --- | --- | --- |
| Funeral with a headshot | `6CA211-996230` | 9 completed final PNGs; headshot recorded in the frozen brief. |
| Celebration with a logo | `4E8EEE-671617` | 9 completed final PNGs; logo recorded in the frozen brief. |
| 15-row programme with both assets | `1B4566-784525` | 9 completed final PNGs; 15 agenda rows, headshot and logo. |

For each reference, the checker calls the Prompt and Image Service status
endpoints, retrieves the repository manifest, downloads all nine artifacts,
and checks every SHA-256 value. This demonstrates durable repository retrieval
after the documented repository restart procedure in `podman-runbook.md`.

The current validation also includes an idempotency rerun of the 15-row
sample: the workflow must reuse the same nine repository artifact IDs and
preserve every SHA-256 value.

## Interruption and recovery evidence

Reference `C2AE04-430496` was deliberately interrupted while its Image Service
job reported 8 of 9 outputs complete and `processing`. The container required
SIGKILL after its normal termination window, then restarted successfully. The
persisted workflow resumed to completion and `verify_resume.py` confirmed nine
repository artifacts without changing an existing artifact ID or SHA-256 value.
