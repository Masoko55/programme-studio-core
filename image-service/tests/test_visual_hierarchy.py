"""Generic visual hierarchy and structure checks without diffusion."""

import unittest

import cv2
import numpy as np
from PIL import Image

from app.services.background_policy import constrain_to_requested_palette, validate_palette
from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt, failure_category
from app.services.scene_density import assess_scene_density
from app.services.visual_hierarchy import density_level
from tests.test_semantic_foundation import document


COMPLEX = (
    "dense forest festival", "futuristic transport exhibition",
    "underwater reef celebration", "historic city heritage event",
    "industrial machinery showcase", "night market",
    "abstract geometric installation",
)
SIMPLE = (
    "minimal abstract ceremony", "single floral motif",
    "simple monochrome corporate background", "soft cloud background",
)


class VisualHierarchyTests(unittest.TestCase):
    def test_density_levels_and_unseen_briefs(self):
        for index, subject in enumerate((*COMPLEX, *SIMPLE)):
            data = document()
            data["brief"].update(background_inspiration=subject,
                                 density_preference="rich layered" if index < len(COMPLEX) else "sparse minimal",
                                 primary_colour="green", secondary_colour="cream")
            data["brief"]["visual_semantics"].update(
                motif_descriptors=["repeated organic arcs"],
                motif_structure_description="connected arcs with clear repeated spacing")
            for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
                with self.subTest(subject=subject, engine=engine):
                    spec = build_candidate_spec(data, engine, "A")
                    hierarchy = spec.visual_design["visual_hierarchy"]
                    compiled = compile_candidate_prompt(spec, 1)
                    self.assertIn(subject, compiled.positive)
                    self.assertIn(hierarchy["detail_density"], compiled.positive)
                    self.assertIn("breathing room", compiled.positive)
                    self.assertIn("motif topology", compiled.positive)
                    self.assertIn("largest visual masses", compiled.positive)
                    self.assertIn("connected arcs", compiled.positive)
                    self.assertEqual(hierarchy["semantic_emphasis"]["subject"], "PRIMARY")
                    self.assertEqual(hierarchy["detail_density"],
                                     "rich" if index < len(COMPLEX) else "sparse")

    def test_motif_prominence_uses_intent_and_not_event(self):
        data = document()
        data["brief"]["visual_semantics"].update(
            motif_descriptors=["nested rings"], motif_structure_description="concentric rings")
        data["brief"]["theme_reference_treatment"] = "nested rings as the focal structure"
        spec = build_candidate_spec(data, "sdxl-1-0", "A")
        self.assertEqual(spec.visual_design["visual_hierarchy"]["motif_prominence"], "primary")
        data["brief"]["theme_reference_treatment"] = "restrained atmospheric treatment"
        spec = build_candidate_spec(data, "sdxl-1-0", "A")
        self.assertEqual(spec.visual_design["visual_hierarchy"]["motif_prominence"], "supporting")

    def test_density_drift_depends_on_distribution(self):
        rng = np.random.default_rng(7)
        crowded = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
        calm_pixels = np.full((256, 256, 3), 220, dtype=np.uint8)
        calm_pixels[:90, :90] = rng.integers(0, 256, (90, 90, 3), dtype=np.uint8)
        calm = Image.fromarray(calm_pixels)
        self.assertEqual(assess_scene_density(crowded, "sparse")["decision"], "DENSITY_DRIFT")
        self.assertEqual(assess_scene_density(crowded, "dense")["decision"], "PASS")
        self.assertEqual(assess_scene_density(calm, "sparse")["decision"], "PASS")
        self.assertGreater(assess_scene_density(calm, "sparse")["quiet_tile_fraction"], 0)
        self.assertEqual(failure_category("DENSITY_DRIFT: crowded"), "DENSITY_DRIFT")

    def test_palette_recovery_preserves_luminance_edges(self):
        gray = np.tile(np.linspace(50, 220, 128, dtype=np.uint8), (128, 1))
        source = np.stack((gray, (gray * .7).astype(np.uint8), (gray * .5).astype(np.uint8)), axis=2)
        source[:, 64:] = source[:, 64:, ::-1]
        recovered = np.asarray(constrain_to_requested_palette(Image.fromarray(source), "royal blue", "red"))
        before = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        after = cv2.cvtColor(recovered, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        self.assertLess(float(np.mean(np.abs(before - after))), 3)
        self.assertLess(float(np.mean(np.abs(np.diff(before, axis=1) - np.diff(after, axis=1)))), 4)

    def test_incidental_hue_does_not_equal_major_design_mass(self):
        source = np.zeros((176, 128, 3), dtype=np.uint8)
        source[:] = (25, 35, 130)
        source[:, 90:] = (220, 30, 35)
        incidental = source.copy()
        incidental[20:22, 20:22] = (255, 225, 0)
        self.assertLess(validate_palette(Image.fromarray(incidental), "royal blue", "red",
                                         mode="dominant", quality_result={})["off_palette_ratio"], .01)
        major = source.copy()
        major[20:60, 20:60] = (255, 225, 0)
        with self.assertRaisesRegex(ValueError, "colours that were not requested"):
            validate_palette(Image.fromarray(major), "royal blue", "red",
                             mode="dominant", quality_result={})

    def test_botanical_control_remains_layered_and_calm(self):
        data = document()
        data["brief"].update(background_inspiration="organic foliage with a clear garden path",
                             primary_colour="green", secondary_colour="cream",
                             density_preference="rich layered")
        data["brief"]["visual_semantics"]["atmosphere_descriptors"] = ["warm", "calm"]
        prompt = compile_candidate_prompt(build_candidate_spec(data, "sdxl-1-0", "A"), 1)
        for phrase in ("organic foliage", "garden path", "green", "cream", "warm", "calm",
                       "layered", "breathing room"):
            self.assertIn(phrase, prompt.positive)


if __name__ == "__main__":
    unittest.main()
