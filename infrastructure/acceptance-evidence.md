# Acceptance evidence

Run `python3 infrastructure/verify_acceptance.py` after starting the three
Podman services. It verifies these completed API-driven samples:

| Requirement | Reference | Evidence |
| --- | --- | --- |
| Funeral with a headshot | `97C1E4-549908` | 9 completed final PNGs; headshot recorded in the frozen brief. |
| Celebration with a logo | `312CE7-445589` | 9 completed final PNGs; logo recorded in the frozen brief. |
| 15-row programme with both assets | `212407-703088` | 9 completed final PNGs; 15 agenda rows, headshot and logo. |

For each reference, the checker calls the Prompt and Image Service status
endpoints, retrieves the repository manifest, downloads all nine artifacts,
and checks every SHA-256 value. This demonstrates durable repository retrieval
after the documented repository restart procedure in `podman-runbook.md`.

The current validation also includes an idempotency rerun of
`212407-703088`: the workflow reused the same nine repository artifact IDs and
preserved every SHA-256 value.
