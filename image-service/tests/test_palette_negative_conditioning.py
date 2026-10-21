"""Palette exclusions must reach native negative encoders across retries."""

import unittest

import numpy as np
from PIL import Image

from app.services.background_policy import NAMED_COLOURS, validate_palette
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import build_workflow
from app.services.prompt_compiler import compile_candidate_prompt, runtime_retry_category
from app.services.sampling_profiles import FailureHistorySummary
from tests.test_candidate_spec_compiler import CASES, document_for


def palette_spec(engine, primary="green", secondary="purple"):
    document = document_for(CASES[0])
    document["brief"].update(primary_colour=primary, secondary_colour=secondary,
                             palette_mode="dominant", palette_relationship_source="default")
    return build_candidate_spec(document, engine, "B")


class PaletteNegativeConditioningTests(unittest.TestCase):
    def test_negative_capable_workflows_receive_palette_exclusions(self):
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            with self.subTest(engine=engine):
                spec = palette_spec(engine)
                compiled = compile_candidate_prompt(spec, 1, concise=True)
                workflow = build_workflow(engine, compiled.positive, compiled.negative,
                                          123, "offline", compiled_prompt=compiled)
                negative = workflow["40"]["inputs"]
                fields = (negative["text"],) if engine == "sdxl-1-0" else (
                    negative["clip_l"], negative["clip_g"], negative["t5xxl"])
                for field in fields:
                    for phrase in ("off-palette", "third-colour", "people", "characters",
                                   "words", "logos"):
                        self.assertIn(phrase, field)
                    for allowed in ("forest green", "mint green", "violet", "lavender", "black", "white",
                                    "grey", "charcoal", "silver"):
                        self.assertNotIn(allowed, field)
                self.assertIn("green", compiled.clip_g)
                self.assertIn("purple", compiled.clip_g)

    def test_palette_retry_strengthens_without_losing_active_semantics(self):
        spec = palette_spec("sdxl-1-0")
        negatives = []
        for attempt in (2, 4, 7, 9):
            compiled = compile_candidate_prompt(spec, attempt, "PALETTE_OFF: third colour",
                                                concise=True)
            negatives.append(compiled.negative)
            for field in (compiled.positive, compiled.clip_g):
                self.assertIn("green", field)
                self.assertIn("purple", field)
                self.assertIn("city skyline", field)
            self.assertEqual(runtime_retry_category("PALETTE_OFF: third colour"), "PALETTE_OFF")
        self.assertEqual(len(set(negatives)), 4)
        self.assertIn("principal chromatic surfaces", compile_candidate_prompt(
            spec, 9, "PALETTE_OFF: third colour", concise=True).positive)

    def test_safe_region_retry_keeps_persistent_palette_on_both_engines(self):
        history = FailureHistorySummary(counts={"PALETTE_OFF": 6},
                                        consecutive_category="DENSE_SAFE_REGION", consecutive_count=1)
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            with self.subTest(engine=engine):
                spec = palette_spec(engine)
                compiled = compile_candidate_prompt(spec, 9, "DENSE_SAFE_REGION: title area",
                                                    concise=True, failure_history=history)
                self.assertIn("Persistent palette recovery", compiled.positive)
                self.assertIn("move detail away from integrated calm regions", compiled.positive.lower())
                self.assertIn("off-palette", compiled.negative)
                self.assertIn("third-colour", compiled.negative)
                without_history = compile_candidate_prompt(
                    spec, 9, "DENSE_SAFE_REGION: title area", concise=True)
                self.assertNotIn("third-colour fields on principal chromatic surfaces",
                                 without_history.negative)
                self.assertIn("third-colour fields on principal chromatic surfaces",
                              compiled.negative)
                if engine == "sd-3-5-medium":
                    for phrase in ("Visual subject", "Motif", "Event atmosphere", "Palette"):
                        self.assertIn(phrase, compiled.t5)
                    self.assertIn("Persistent palette recovery", compiled.t5)

    def test_flux_remains_affirmative_only_and_black_white_is_achromatic(self):
        flux = compile_candidate_prompt(palette_spec("flux-2"), 4, "PALETTE_OFF",
                                        concise=True)
        self.assertFalse(flux.negative_prompt_channel_active)
        self.assertIn("restricted to the requested families", flux.positive)
        workflow = build_workflow("flux-2", flux.positive, flux.negative, 123,
                                  "offline", compiled_prompt=flux)
        self.assertNotIn("40", workflow)
        monochrome = compile_candidate_prompt(palette_spec("sdxl-1-0", "black", "white"),
                                               1, concise=True)
        self.assertIn("chromatic", monochrome.negative)
        self.assertNotIn("black fields", monochrome.negative)

    def test_validator_still_rejects_independent_third_colour(self):
        for third in ("red", "yellow", "cyan"):
            with self.subTest(third=third):
                pixels = np.empty((176, 128, 3), dtype=np.uint8)
                pixels[:, :48] = NAMED_COLOURS["green"]
                pixels[:, 48:96] = NAMED_COLOURS["purple"]
                pixels[:, 96:] = NAMED_COLOURS[third]
                with self.assertRaises(ValueError):
                    validate_palette(Image.fromarray(pixels), "green", "purple", quality_result={})

    def test_validator_accepted_named_tone_is_not_negated(self):
        pixels = np.empty((176, 128, 3), dtype=np.uint8)
        pixels[:, :42] = NAMED_COLOURS["blue"]
        pixels[:, 42:84] = NAMED_COLOURS["red"]
        pixels[:, 84:] = NAMED_COLOURS["pink"]
        result = validate_palette(Image.fromarray(pixels), "blue", "red", quality_result={})
        self.assertEqual(result["off_palette_ratio"], 0.0)
        compiled = compile_candidate_prompt(palette_spec("sdxl-1-0", "blue", "red"),
                                            1, concise=True)
        self.assertNotIn("pink", [item.strip() for item in compiled.negative.split(",")])


if __name__ == "__main__":
    unittest.main()
