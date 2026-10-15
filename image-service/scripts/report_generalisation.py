"""Offline semantic transport health across fixed, unseen, holdout and fuzz briefs."""

from __future__ import annotations

import json

from audit_genericity import CASES, UNKNOWN_CASES, HOLDOUT_CASES, ENGINES, _document, audit_briefs, audit_branches
from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt
from tests.fixtures.fuzz_cases import cases as fuzz_cases


def report() -> dict:
    groups = {"fixed_regression": CASES, "unseen": UNKNOWN_CASES,
              "holdout": HOLDOUT_CASES, "fuzz": fuzz_cases()}
    summary = {}
    for name, cases in groups.items():
        failures = audit_briefs(cases)
        total = len(cases) * len(ENGINES)
        completeness = []
        covered = []
        truncations = 0
        compiled_count = 0
        for index, case in enumerate(cases):
            for engine in ENGINES:
                try:
                    spec = build_candidate_spec(_document(index, case), engine, "A")
                    compiled = compile_candidate_prompt(spec, 1)
                except (ValueError, KeyError, TypeError):
                    continue
                compiled_count += 1
                completeness.append(spec.visual_design["semantic_completeness"]["score"])
                active = compiled.positive if engine == "flux-2" else compiled.clip_g
                covered.append(all(word.casefold() in active.casefold() for word in case[2].split()))
                truncations += bool(compiled.record()["truncated"])
        summary[name] = {"briefs": len(cases), "engine_runs": total,
                         "compile_success_pct": round(100 * compiled_count / total, 2),
                         "semantic_preservation_pct": round(100 * (total - len(failures)) / total, 2),
                         "contract_completeness_pct": round(100 * sum(completeness) / len(completeness), 2) if completeness else None,
                         "engine_coverage_pct": round(100 * sum(covered) / len(covered), 2) if covered else None,
                         "unexpected_truncation_count": truncations,
                         "findings": failures[:10]}
    branches = audit_branches()
    return {"groups": summary, "case_specific_production_branch_count":
            sum(item["classification"] == "CASE_SPECIFIC_BEHAVIOUR" for item in branches)}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
