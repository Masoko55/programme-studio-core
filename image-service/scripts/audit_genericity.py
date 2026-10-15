"""Static, offline audit of unseen briefs and production event branches.

Run from image-service with: ./.venv/bin/python scripts/audit_genericity.py
This script compiles prompts and mocks the vision service; it never renders.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt
from app.services.subject_adherence import assess_subject_adherence
from tests.fixtures.holdout_cases import CASES as HOLDOUT_CASES


CASES = (
    ("baby shower", "clouds and stars", "soft dreamy warm", "family",
     "layered clouds and scattered stars", "cream", "pale blue", "ominous"),
    ("technology conference", "futuristic network", "professional innovative clean", "professionals",
     "connected network nodes and luminous links", "navy", "silver", "chaotic"),
    ("rustic wedding", "countryside natural materials", "romantic warm elegant", "adult guests",
     "woven natural textures and countryside foliage", "sage green", "cream", "sterile"),
    ("children's dinosaur party", "prehistoric jungle", "playful adventurous child-friendly", "children",
     "layered fern silhouettes and dinosaur footprints", "forest green", "orange", "ominous"),
    ("wellness retreat", "calm natural landscape", "peaceful restorative minimal", "adults",
     "layered hills and flowing water", "sage green", "cream", "chaotic"),
    ("music festival", "energetic abstract sound waves", "vibrant rhythmic celebratory", "mixed ages",
     "repeating wave bands and rhythmic arcs", "purple", "gold", "dull"),
)
UNKNOWN_CASES = (
    ("underwater bioluminescent research exhibition", "glowing deep-sea observatory", "quietly luminous meditative weightless", "curious researchers", "floating bioluminescent clusters", "navy", "teal", "harsh"),
    ("vintage railway heritage gathering", "historic rail platforms", "nostalgic intimate earthy", "rail enthusiasts", "layered tracks and arched station ironwork", "brown", "cream", "futuristic"),
    ("lunar horticulture showcase", "moon garden habitat", "serene ethereal restorative", "garden visitors", "lunar planting beds and translucent domes", "silver", "green", "chaotic"),
    ("ceramic craft market", "handmade pottery display", "warm tactile unhurried", "craft lovers", "repeating clay vessel silhouettes", "terracotta", "cream", "sterile"),
    ("desert astronomy evening", "star-filled desert horizon", "vast contemplative still", "stargazers", "layered dunes and distributed constellations", "navy", "gold", "cluttered"),
)

ENGINES = ("sdxl-1-0", "sd-3-5-medium", "flux-2")
EVENT_LITERALS = re.compile(
    r"\b(?:superhero|black[- ]tie|botanic\w*|award\w*|garden|ballroom|"
    r"memorial|corporate|conference|birthday|wedding|bridal|shower|child\w*|kid\w*)\b",
    re.I,
)


def _document(index: int, case: tuple[str, ...]) -> dict:
    event, theme, mood, audience, motif, primary, secondary, incompatible = case
    subject = f"Illustrated {theme} with recognizable spatial depth"
    return {"reference_number": f"{index + 500:06X}-123456", "brief": {
        "event_type": event, "theme": theme, "background_inspiration": subject,
        "creative_description": f"Create a {mood} event background with {theme}.",
        "mood": mood, "audience_context": audience,
        "symbolic_motifs": [motif], "visual_anchors": [theme, motif],
        "primary_colour": primary, "secondary_colour": secondary,
        "palette_mode": "dominant", "palette_relationship": f"{primary} primary, {secondary} secondary",
        "visual_semantics": {"atmosphere_descriptors": mood.split(),
            "audience_descriptors": [audience], "motif_descriptors": [motif],
            "motif_structure_description": motif, "lighting_descriptors": mood.split(),
            "incompatible_treatments": [incompatible], "scene_family": "general",
            "composition_affinities": ["layered"]},
    }, "directions": [{"direction": {
        "direction_id": "A", "role": "Visual Designer",
        "positive_prompt": f"Illustrate {theme} as a layered background",
        "negative_prompt": "people, text", "composition_strategy": "asymmetric layered subject arrangement",
        "focal_strategy": "off-centre subject focal forms",
        "depth_strategy": "foreground middle and distant layers",
        "border_or_perimeter_strategy": "open irregular perimeter",
        "title_safe_region_strategy": "quiet upper region integrated with scenery",
        "programme_safe_region_strategy": "quiet lower region integrated with scenery",
        "layout_guidance": {},
    }}]}


def _vision_contract(design: dict) -> tuple[dict, str]:
    image = Mock()
    image.convert.return_value = image
    image.save.side_effect = lambda buffer, **_: buffer.write(b"mock-jpeg")
    opened = Mock()
    opened.__enter__ = Mock(return_value=image)
    opened.__exit__ = Mock(return_value=False)
    response = Mock(status_code=200, headers={"content-type": "application/json"})
    response.json.return_value = {"message": {"content": json.dumps({
        "subject_score": .9, "atmosphere_score": .9, "motif_score": .9,
        "audience_fit_score": .9, "observed_evidence": ["matching forms"],
        "reason": "visible brief fit",
    })}}
    with patch("app.services.subject_adherence.Image.open", return_value=opened), \
         patch("app.services.subject_adherence.httpx.post", return_value=response) as post:
        result = assess_subject_adherence(Path("/unused.png"), design, model="minicpm-v4.5:latest")
    return result, post.call_args.kwargs["json"]["messages"][0]["content"]


def audit_briefs(cases: tuple = (*CASES, *UNKNOWN_CASES)) -> list[dict]:
    findings = []
    for index, case in enumerate(cases):
        event, theme, mood, audience, motif, primary, secondary, incompatible = case
        document = _document(index, case)
        for engine in ENGINES:
            try:
                spec = build_candidate_spec(document, engine, "A")
                compiled = compile_candidate_prompt(spec, 1)
                result, vision_prompt = _vision_contract(spec.visual_design)
            except (ValueError, KeyError, TypeError) as error:
                findings.append({"event": event, "engine": engine,
                                 "missing": ["compile_success"], "error": str(error)[:180]})
                continue
            active_atmosphere = compiled.positive if engine == "flux-2" else compiled.clip_g
            checks = {
                "subject": spec.visual_design["primary_visual_subject"] in compiled.positive
                           and theme in vision_prompt,
                "atmosphere": all(word in compiled.positive.casefold() for word in mood.split())
                              and all(word in active_atmosphere.casefold()
                                      for word in mood.split()),
                "audience_fit": audience in vision_prompt and result["audience_fit_score"] == .9
                                and "audience_fit_score" in result["thresholds"],
                "motif_structure": motif in (compiled.t5 or compiled.positive) and motif in vision_prompt,
                "palette_constraints": primary in compiled.positive and secondary in compiled.positive
                                       and primary in (compiled.positive if engine == "flux-2" else compiled.clip_g),
                "negative_mood_exclusions": (compiled.negative_semantics_preserved
                                              and incompatible in spec.visual_design["visual_semantics"]["incompatible_treatments"]
                                              and compiled.negative_prompt_channel_active == (engine != "flux-2")
                                              and (engine == "flux-2" or incompatible in compiled.negative.casefold())),
                "scene_family": spec.visual_design["visual_semantics"]["scene_family"] == "general",
            }
            missing = [name for name, passed in checks.items() if not passed]
            if missing:
                findings.append({"event": event, "engine": engine, "missing": missing})
    return findings


def classify_source(relative: str, source: str) -> list[dict]:
    """Separate educational examples from named-event production decisions."""
    tree = ast.parse(source, filename=relative)
    findings = []
    condition_nodes = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp)):
            continue
        condition_nodes.update(ast.walk(node.test))
        expression = ast.unparse(node.test)
        if EVENT_LITERALS.search(expression) or re.search(r"\bevent_type\s*==\s*['\"]", expression):
            taxonomy = relative.endswith("/grill_me_semantic.py")
            findings.append({"file": relative, "line": node.lineno,
                             "literal": expression[:180],
                             "classification": "GENERIC_TAXONOMY" if taxonomy else "CASE_SPECIFIC_BEHAVIOUR",
                             "reason": ("Semantic review classification; does not select rendering treatment"
                                        if taxonomy else "Named event or theme changes production behavior")})
    if relative.endswith("/grill_me_semantic.py"):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node not in condition_nodes and EVENT_LITERALS.search(node.value)):
                findings.append({"file": relative, "line": node.lineno,
                                 "literal": EVENT_LITERALS.search(node.value).group(),
                                 "classification": "EXAMPLE_ONLY",
                                 "reason": "Instructional or data literal, not a rendering branch"})
    return findings


def audit_branches() -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    findings = []
    for service in ("image-service", "prompt-service"):
        for path in (root / service / "app").rglob("*.py"):
            findings.extend(classify_source(str(path.relative_to(root)), path.read_text(encoding="utf-8")))
    return findings


def main() -> int:
    report = {"briefs": len(CASES), "unknown_briefs": len(UNKNOWN_CASES),
              "holdout_briefs": len(HOLDOUT_CASES), "engines": len(ENGINES),
              "brief_findings": audit_briefs(), "holdout_findings": audit_briefs(HOLDOUT_CASES),
              "branch_findings": audit_branches()}
    bad_branches = [item for item in report["branch_findings"] if item["classification"] == "CASE_SPECIFIC_BEHAVIOUR"]
    report["result"] = ("GENERICITY AUDIT FAIL — CASE-SPECIFIC BEHAVIOUR REMAINS"
                        if report["brief_findings"] or report["holdout_findings"] or bad_branches else "GENERICITY AUDIT PASS")
    print(json.dumps(report, indent=2))
    return 1 if report["brief_findings"] or report["holdout_findings"] or bad_branches else 0


if __name__ == "__main__":
    raise SystemExit(main())
