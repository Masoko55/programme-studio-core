# Unseen renderer holdout

`holdout_cases.py` defines 50 static offline briefs. **HOLDOUT CASES MUST NOT BE USED AS EVENT-SPECIFIC IMPLEMENTATION TARGETS.** The genericity check compiles them across SDXL, SD3.5 and FLUX without rendering images. Add new cases to measure generalisation, not to justify named-event branches or tune thresholds.

Run `scripts/check_genericity.sh` in the image-service environment as the genericity and holdout quality gate.
