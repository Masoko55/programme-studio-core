"""A later retry must retain confirmed figure exclusions in active channels."""

import unittest

from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import build_workflow
from app.services.prompt_compiler import compile_candidate_prompt
from app.services.sampling_profiles import FailureHistorySummary
from tests.test_candidate_spec_compiler import CASES, document_for


class IntrusionRetryContractTests(unittest.TestCase):
    def _spec(self, engine):
        return build_candidate_spec(document_for(CASES[0]), engine, "B")

    def test_prior_human_survives_later_palette_retry(self):
        history = FailureHistorySummary({"HUMAN": 1, "PALETTE_OFF": 1}, "PALETTE_OFF", 1)
        for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
            with self.subTest(engine=engine):
                spec = self._spec(engine)
                compiled = compile_candidate_prompt(spec, 3, "PALETTE_OFF: drift",
                                                    concise=True, failure_history=history)
                self.assertIn("same background scene", compiled.positive)
                self.assertIn("no human, mascot or costumed figures", compiled.positive)
                self.assertIn("no human, mascot or costumed figures", compiled.clip_g)
                workflow = build_workflow(engine, compiled.positive, compiled.negative,
                                          123, "offline", compiled_prompt=compiled)
                if engine == "flux-2":
                    self.assertNotIn("40", workflow)
                else:
                    self.assertIn("human silhouettes", compiled.negative)
                    negative = workflow["40"]["inputs"]
                    fields = (negative["text"],) if engine == "sdxl-1-0" else (
                        negative["clip_l"], negative["clip_g"], negative["t5xxl"])
                    self.assertTrue(all("human silhouettes" in field for field in fields))
                    if engine == "sd-3-5-medium":
                        self.assertIn("no human, mascot or costumed figures", compiled.t5)

    def test_prior_character_survives_later_safe_region_retry(self):
        history = FailureHistorySummary({"CHARACTER": 1, "DENSE_SAFE_REGION": 1},
                                        "DENSE_SAFE_REGION", 1)
        for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
            with self.subTest(engine=engine):
                compiled = compile_candidate_prompt(
                    self._spec(engine), 3, "DENSE_SAFE_REGION: detail", concise=True,
                    failure_history=history)
                self.assertIn("permitted abstract motifs", compiled.positive)
                self.assertIn("move detail away", compiled.positive.lower())
                if engine != "flux-2":
                    self.assertIn("depicted costumed figures", compiled.negative)

    def test_current_figure_failure_and_unrelated_retry_histories(self):
        for category in ("HUMAN", "CHARACTER"):
            with self.subTest(category=category):
                compiled = compile_candidate_prompt(self._spec("sdxl-1-0"), 2,
                                                    f"{category}: confirmed", concise=True)
                self.assertIn("same background scene", compiled.positive)
                self.assertIn("figures", compiled.negative)
        palette_only = FailureHistorySummary({"PALETTE_OFF": 1}, "PALETTE_OFF", 1)
        compiled = compile_candidate_prompt(self._spec("sdxl-1-0"), 3,
                                            "DENSE_SAFE_REGION", concise=True,
                                            failure_history=palette_only)
        self.assertNotIn("same background scene", compiled.positive)
        self.assertNotIn("depicted costumed figures", compiled.negative)

    def test_mixed_failure_sequences_keep_only_persistent_contracts(self):
        sequences = (
            ({"HUMAN": 1, "PALETTE_OFF": 1}, "PALETTE_OFF", "human silhouettes"),
            ({"HUMAN": 1, "PALETTE_OFF": 1, "DENSE_SAFE_REGION": 1},
             "DENSE_SAFE_REGION", "human silhouettes"),
            ({"HUMAN": 1, "TEXT": 1}, "TEXT", "human silhouettes"),
            ({"CHARACTER": 1, "DENSE_SAFE_REGION": 1}, "DENSE_SAFE_REGION",
             "depicted costumed figures"),
            ({"PALETTE_OFF": 1, "HUMAN": 1}, "HUMAN", "human silhouettes"),
            ({"SUBJECT_DRIFT": 1, "PALETTE_OFF": 1}, "PALETTE_OFF", ""),
            ({"PALETTE_OFF": 1, "DENSE_SAFE_REGION": 1}, "DENSE_SAFE_REGION", ""),
        )
        spec = self._spec("sdxl-1-0")
        for counts, latest, exclusion in sequences:
            with self.subTest(counts=counts, latest=latest):
                history = FailureHistorySummary(counts, latest, 1)
                compiled = compile_candidate_prompt(spec, 3, latest, concise=True,
                                                    failure_history=history)
                if exclusion:
                    self.assertIn(exclusion, compiled.negative)
                    self.assertIn("same background scene", compiled.positive)
                else:
                    self.assertNotIn("human silhouettes", compiled.negative)
                    self.assertNotIn("depicted costumed figures", compiled.negative)
                if "PALETTE_OFF" in counts and latest != "PALETTE_OFF":
                    self.assertIn("Persistent palette recovery", compiled.positive)


if __name__ == "__main__":
    unittest.main()
