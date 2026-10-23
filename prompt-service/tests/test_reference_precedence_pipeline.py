"""Static prompt-to-engine contract replay; no model or image generation."""

import asyncio
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.schemas.creative_direction import CreativeDirectionOutput
from app.schemas.grill_me import GrillMeAnswers, GrillMeSession, ProgrammeItem
from app.services.creative_direction import build_creative_direction_prompt, generate_creative_direction
from app.services.grill_me_semantic import (_safe_context, _selected_reference_trait,
                                             model_facing_brief, renderable_brief)
from app.services.grill_me_service import apply_clarifications, freeze_for_generation
from app.services.semantic_contract import build_semantic_contract


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
IMAGE_SERVICE = ROOT / "image-service"


def _engine_replay(document: dict) -> dict:
    code = """import json, sys
from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt
document = json.load(sys.stdin)
result = {}
for engine in ('flux-2', 'sdxl-1-0', 'sd-3-5-medium'):
    result[engine] = {}
    for letter in 'ABC':
        spec = build_candidate_spec(document, engine, letter)
        prompt = compile_candidate_prompt(spec, 1)
        result[engine][letter] = {
            'motifs': spec.semantic_contract['motif_descriptors'],
            'structure': spec.semantic_contract['motif_structure_description'],
            'positive': prompt.positive,
            'clip_l': prompt.clip_l,
            'clip_g': prompt.clip_g,
            't5': prompt.t5 or '',
            'negative': prompt.negative,
            'incompatible': spec.semantic_contract['incompatible_treatments'],
        }
print(json.dumps(result))"""
    run = subprocess.run([str(PYTHON), "-c", code], input=json.dumps(document), text=True,
                         capture_output=True, cwd=IMAGE_SERVICE, check=True)
    return json.loads(run.stdout)


def _direction(letter: str, role: str = "brief-led") -> CreativeDirectionOutput:
    return CreativeDirectionOutput.model_validate({
        "direction_id": letter, "role": role,
        "positive_prompt": "Abstract requested-colour fields with calm open space",
        "negative_prompt": "people, characters, text", "title_suggestions": ["Concentric Motifs"],
        "short_design_rationale": "The rejected lattice would have been a different concept",
        "composition_strategy": "asymmetric distribution of broad visual masses",
        "focal_strategy": "off-centre focal colour field",
        "depth_strategy": "shallow overlapping colour planes",
        "border_or_perimeter_strategy": "open irregular perimeter",
        "title_safe_region_strategy": "quiet upper region",
        "programme_safe_region_strategy": "quiet lower region",
        "layout_guidance": {"title_zone": {"x": .1, "y": .08, "width": .8, "height": .12},
                            "programme_zone": {"x": .1, "y": .5, "width": .8, "height": .35}},
    })


class ReferencePrecedencePipelineTests(unittest.TestCase):
    @staticmethod
    def _answers(name: str, treatment: str | None = None) -> GrillMeAnswers:
        primary, secondary = ("blue", "red") if name == "Spider-Man" else ("burgundy", "ivory")
        return GrillMeAnswers(
            event_type="Children birthday party" if name == "Spider-Man" else "family celebration",
            theme=f"inspired by {name}",
            age_group="Children aged 7 to 10" if name == "Spider-Man" else "family",
            primary_colour=primary, secondary_colour=secondary,
            creative_description=("A kids birthday party for a kid who is into superheroes, "
                                  f"most especially {name}." if name == "Spider-Man" else
                                  f"A family celebration inspired by {name}"),
            background_inspiration="Abstract requested-colour fields with calm open space",
            theme_reference_treatment=treatment, title_preference="Welcome",
            event_date="2026-10-15", start_time="10:00",
            programme=[ProgrammeItem(time="10:00", title="Arrival")],
        )

    @staticmethod
    def _review(name: str) -> dict:
        motif = "web patterns" if name == "Spider-Man" else "concentric motifs"
        review = {"reference_detected": True, "reference_name": name,
                "reference_type": "character", "visual_semantics": {
                    "motif_descriptors": [motif],
                    "motif_structure_description": f"{motif} with flowing movement",
                    "composition_affinities": ["flowing movement"],
                    "material_descriptors": ["dark architectural forms"],
                }}
        if name == "Spider-Man":
            review["visual_semantics"].update({
                "atmosphere_descriptors": ["energetic and playful"],
                "audience_descriptors": ["children aged 7 to 10"],
            })
            review["semantic_grounding"] = {
                "atmosphere_descriptors": {"energetic and playful": "Children birthday party"},
                "audience_descriptors": {"children aged 7 to 10": "Children aged 7 to 10"},
            }
        return review

    def _freeze_after_clarification(self, name: str, treatment: str) -> dict:
        answers = self._answers(name)
        review = self._review(name)
        session = GrillMeSession(session_id="static-replay", status="clarification_required",
                                answers=answers, creative_context=_safe_context(review, answers.model_dump()))
        self.assertTrue(session.creative_context["reference_needs_clarification"])
        with (patch("app.services.grill_me_service.load_session", return_value=session),
              patch("app.services.grill_me_service._save", side_effect=lambda value: value),
              patch("app.services.grill_me_service.review_creative_context",
                    side_effect=lambda value: _safe_context(review, value))):
            clarified = apply_clarifications("static-replay", {"theme_reference_treatment": treatment})
        hydrated = GrillMeSession.model_validate_json(clarified.model_dump_json())
        self.assertFalse(hydrated.creative_context["reference_needs_clarification"])
        with (patch("app.services.grill_me_service.load_session", return_value=hydrated),
              patch("app.services.grill_me_service.evaluate_session", return_value=[]),
              patch("app.services.grill_me_service._save")):
            _, brief, _ = freeze_for_generation("static-replay", "ABCDEF-123456")
        return brief

    def _replay(self, name: str, treatment: str, forbidden: tuple[str, ...],
                approved: tuple[str, ...] = (), intrude: bool = False) -> tuple[dict, dict]:
        brief = self._freeze_after_clarification(name, treatment)
        self.assertTrue(brief["grill_me_context"]["reference_detected"])
        self.assertEqual(brief["grill_me_context"]["reference_name"], name)
        positive_semantics = {key: brief["semantic_contract"].get(key) for key in (
            "motif_descriptors", "motif_structure_description", "composition_affinities",
            "material_descriptors", "scene_family")}
        for term in forbidden:
            self.assertNotIn(term.casefold(), json.dumps(positive_semantics).casefold())
        directions = []
        for letter in "ABC":
            prompt = build_creative_direction_prompt(brief, letter, "visual designer")
            self.assertNotIn(name.casefold(), prompt.casefold())
            response = _direction(letter)
            if intrude:
                response.role = "Concentric motifs"
                response.positive_prompt = "Concentric motifs with layered colour fields"
                response.short_design_rationale = "Concentric motifs should lead the composition"
            with patch("app.services.creative_direction.generate_text", new_callable=AsyncMock,
                       return_value={"response": response.model_dump_json()}):
                direction = asyncio.run(generate_creative_direction(letter, brief))
            directions.append({"direction": direction.model_dump()})
        document = {"reference_number": "ABCDEF-123456", "brief": brief, "directions": directions}
        engines = _engine_replay(document)
        for variants in engines.values():
            for channels in variants.values():
                active = " ".join(channels[key] for key in ("positive", "clip_l", "clip_g", "t5"))
                for term in (*forbidden, name):
                    self.assertNotIn(term.casefold(), active.casefold())
                for term in approved:
                    self.assertIn(term.casefold(), active.casefold())
        return brief, engines

    def test_narrow_treatment_bounds_inferred_reference_traits(self):
        brief, engines = self._replay(
            "Crimson Oracle", "Use burgundy and ivory as the visual influence. "
            "Do not use architecture. Do not depict any person or character.",
            ("concentric", "flowing movement", "architecture"), ("burgundy", "ivory"),
            intrude=True)
        self.assertEqual(brief["visual_semantics"]["motif_descriptors"], [])
        self.assertEqual(brief["visual_semantics"]["motif_structure_description"], "")
        self.assertIn("architecture", brief["semantic_contract"]["incompatible_treatments"])
        for variants in engines.values():
            for channels in variants.values():
                self.assertIn("architecture", channels["incompatible"])

    def test_explicit_motif_selection_survives(self):
        brief, _ = self._replay(
            "Crimson Oracle", "Use burgundy and ivory with subtle concentric motifs "
            "and flowing movement. Do not use architecture. Do not depict the character.",
            ("architecture",), ("burgundy", "ivory", "concentric motifs", "flowing movement"))
        self.assertIn("concentric motifs", brief["semantic_contract"]["motif_descriptors"])

    def test_explicit_motif_rejection_wins(self):
        brief, _ = self._replay(
            "Crimson Oracle", "Use burgundy and ivory only. Do not use concentric motifs, "
            "radial patterns or architecture. Do not depict the character.",
            ("concentric", "radial patterns", "architecture"), ("burgundy", "ivory"))
        self.assertIn("concentric motifs", brief["semantic_contract"]["incompatible_treatments"])

    def test_mixed_approval_and_exclusion_in_one_sentence(self):
        brief, _ = self._replay(
            "Crimson Oracle", "Use burgundy and ivory with subtle concentric motifs, "
            "but do not use architecture. Do not depict the character.",
            ("architecture",), ("burgundy", "ivory", "concentric motifs"))
        self.assertIn("concentric motifs", brief["semantic_contract"]["motif_descriptors"])

    def test_live_case_semantics_replay_without_web_authorization(self):
        brief, engines = self._replay(
            "Spider-Man", "The colours must influence the visual traits. Do not add buildings. "
            "Do not depict Spider-Man or any person or character.",
            ("web", "flowing movement", "architecture"), ("blue", "red"))
        self.assertIn("buildings", brief["semantic_contract"]["incompatible_treatments"])
        self.assertIn("energetic and playful", brief["semantic_contract"]["atmosphere_descriptors"])
        self.assertIn("children aged 7 to 10", brief["semantic_contract"]["audience_descriptors"])
        self.assertIn("person", brief["semantic_contract"]["incompatible_treatments"])
        self.assertIn("character", brief["semantic_contract"]["incompatible_treatments"])
        for variants in engines.values():
            for channels in variants.values():
                self.assertNotIn("web", " ".join(channels[key] for key in
                                 ("positive", "clip_l", "clip_g", "t5")).casefold())

    def test_ordinary_brief_keeps_inferred_motif(self):
        answers = {"theme": "Future Gathering"}
        context = _safe_context({"visual_semantics": {"motif_descriptors": ["floating lattice"]}}, answers)
        self.assertEqual(context["visual_semantics"]["motif_descriptors"], ["floating lattice"])

    def test_unresolved_safe_treatment_stays_distinct(self):
        answers = {"theme": "Crimson Oracle-inspired", "theme_reference_treatment":
                   "Use burgundy and ivory with concentric motifs and flowing directional "
                   "movement. Do not depict any person or character."}
        context = _safe_context({}, answers, semantic_review_available=False)
        self.assertFalse(context["reference_detected"])
        self.assertIsNone(context["reference_name"])
        self.assertEqual(context["reference_type"], "none")
        self.assertEqual(context["reference_rendering_mode"], "safe_treatment_only")

    def test_prefix_collision_cannot_authorize_trait(self):
        self.assertFalse(_selected_reference_trait("orangutan silhouettes", "Use orange circles", []))
        self.assertFalse(_selected_reference_trait("red web patterns", "Use red and blue", []))
        self.assertTrue(_selected_reference_trait("curves and patterns", "Use curve pattern", []))

    def test_unrelated_reference_atmosphere_needs_independent_evidence(self):
        answers = self._answers("Azure-Monarch").model_dump()
        answers["theme_reference_treatment"] = "Use burgundy and ivory only"
        review = self._review("Azure-Monarch")
        review["visual_semantics"]["atmosphere_descriptors"] = ["heroic and intense"]
        context = _safe_context(review, answers)
        self.assertEqual(context["visual_semantics"]["atmosphere_descriptors"], [])

    def test_independent_material_light_composition_and_scene_survive(self):
        answers = self._answers("Azure-Monarch").model_dump()
        answers.update(theme_reference_treatment="Use burgundy and ivory only",
                       background_inspiration="Architectural interior with linen texture, soft light and arcing bands")
        review = self._review("Azure-Monarch")
        review["visual_semantics"].update({
            "material_descriptors": ["linen", "metallic"],
            "texture_descriptors": ["linen texture"],
            "lighting_descriptors": ["soft light"],
            "composition_affinities": ["arcing bands", "radial movement"],
            "scene_family": "architectural-interior",
        })
        semantics = _safe_context(review, answers)["visual_semantics"]
        self.assertEqual(semantics["material_descriptors"], ["linen"])
        self.assertEqual(semantics["texture_descriptors"], ["linen texture"])
        self.assertEqual(semantics["lighting_descriptors"], ["soft light"])
        self.assertEqual(semantics["composition_affinities"], ["arcing bands"])
        self.assertEqual(semantics["scene_family"], "architectural-interior")

    def test_explicit_prohibition_overrides_other_user_grounding(self):
        answers = self._answers("Azure-Monarch").model_dump()
        answers.update(theme_reference_treatment="Use burgundy and ivory, but do not use linen",
                       background_inspiration="Linen texture and calm colour fields")
        review = self._review("Azure-Monarch")
        review["visual_semantics"]["material_descriptors"] = ["linen"]
        semantics = _safe_context(review, answers)["visual_semantics"]
        self.assertEqual(semantics["material_descriptors"], [])
        self.assertIn("linen", semantics["incompatible_treatments"])

    def test_separator_variants_are_removed_without_erasing_safe_content(self):
        brief = {"grill_me_context": {"reference_detected": True,
                                      "reference_name": "Azure-Monarch"},
                 "theme_reference_treatment": "Use layered burgundy ribbons inspired by Azure Monarch",
                 "creative_description": "Use layered burgundy ribbons inspired by Azure_Monarch",
                 "background_inspiration": "Burgundy ribbons, Azure-Monarch-inspired",
                 "visual_semantics": {"motif_descriptors": ["Azure Monarch crest"]}}
        renderable = renderable_brief(brief)
        self.assertIn("layered burgundy ribbons", renderable["creative_description"])
        self.assertNotIn("Azure", json.dumps({key: renderable[key] for key in
                                              ("theme_reference_treatment", "creative_description",
                                               "background_inspiration")}))
        sanitized = model_facing_brief(renderable)
        serialized = json.dumps({key: value for key, value in sanitized.items()
                                 if key != "grill_me_context"})
        for spelling in ("Azure-Monarch", "Azure Monarch", "Azure_Monarch"):
            self.assertNotIn(spelling, serialized)
        self.assertIn("layered burgundy ribbons", serialized)

    def test_reference_redaction_keeps_visual_detail_but_drops_empty_shells(self):
        brief = {
            "event_type": "children's celebration", "theme": "heroic adventure",
            "background_inspiration": "A Solar Sentinel-inspired theme for a child's celebration",
            "creative_description": "Use Solar Sentinel-inspired angular geometry, layered shadows and limited web-like linework",
            "theme_reference_treatment": "Use Solar Sentinel-inspired colours and visual traits",
            "grill_me_context": {"reference_detected": True, "reference_name": "Solar Sentinel",
                                 "reference_type": "character"},
        }
        safe = renderable_brief(brief)
        self.assertEqual(safe["background_inspiration"], "")
        self.assertEqual(safe["theme_reference_treatment"], "")
        self.assertIn("angular geometry", safe["creative_description"])
        self.assertIn("layered shadows", safe["creative_description"])
        self.assertIn("limited web-like linework", safe["creative_description"])
        contract = build_semantic_contract(safe)
        self.assertEqual(contract["primary_visual_subject"], "heroic adventure")
        self.assertEqual(contract["visual_anchors"], [])
        self.assertNotIn("Solar Sentinel", json.dumps(model_facing_brief(safe)))

    def test_clarified_superhero_brief_keeps_grounded_constraints_without_identity(self):
        brief = {
            "event_type": "Children birthday party", "theme": "Superheroes",
            "age_group": "Children aged 7 to 10", "primary_colour": "royal blue",
            "secondary_colour": "red",
            "creative_description": "A kids party for a kid who is into superheroes, most especially Spider-Man",
            "background_inspiration": "A Spider-Man-inspired theme for a kid's birthday",
            "theme_reference_treatment": "Use Spider-Man-inspired colours and visual traits; limited web motifs",
            "visual_semantics": {"motif_descriptors": ["limited web motifs"],
                                 "incompatible_treatments": ["buildings", "people", "humanoids", "characters"]},
            "grill_me_context": {"reference_detected": True, "reference_name": "Spider-Man",
                                 "reference_type": "character",
                                 "generation_constraints": ["background artwork only", "no buildings",
                                                            "no people", "no humanoids", "no characters"]},
        }
        safe = renderable_brief(brief)
        model = model_facing_brief(safe)
        contract = build_semantic_contract(safe)
        self.assertEqual(contract["primary_visual_subject"], "Superheroes")
        self.assertEqual(contract["palette"], [{"colour": "royal blue", "role": "primary"},
                                               {"colour": "red", "role": "secondary"}])
        self.assertIn("limited web motifs", contract["motif_descriptors"])
        self.assertIn("buildings", contract["incompatible_treatments"])
        self.assertIn("no buildings", model["grill_me_context"]["generation_constraints"])
        self.assertEqual(model["age_group"], "Children aged 7 to 10")
        self.assertNotIn("Spider-Man", json.dumps(model))
