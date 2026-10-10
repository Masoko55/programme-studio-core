import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from app.config.settings import settings
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import ComfyUIError, build_workflow, validate_workflow
from app.services.template_conditioning import compile_template_conditioning, render_template


def candidate(engine, direction):
    return build_candidate_spec({
        "reference_number": "ABCDEF-123456",
        "brief": {"event_type": "gala", "theme": "formal", "primary_colour": "royal blue",
                  "secondary_colour": "red"},
        "directions": [{"direction": {"direction_id": direction, "role": "editorial",
                                     "positive_prompt": "broad edge decoration",
                                     "layout_guidance": {
                                         "title_zone": {"x": .1, "y": .1, "width": .8, "height": .12},
                                         "programme_zone": {"x": .1, "y": .45, "width": .8, "height": .4},
                                     }}}],
    }, engine, direction)


class TemplateConditioningTests(unittest.TestCase):
    def test_frozen_stable_hash_paths_and_distinct_geometry(self):
        with TemporaryDirectory() as temporary:
            with patch.object(settings, "programme_data_path", Path(temporary)):
                layouts = set()
                for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
                    for direction in "ABC":
                        spec = compile_template_conditioning(candidate(engine, direction))
                        self.assertEqual(spec.conditioning_sha256,
                                         compile_template_conditioning(candidate(engine, direction)).conditioning_sha256)
                        self.assertEqual(Path(spec.template_path).relative_to(temporary).parts,
                                         ("ABCDEF-123456", "conditioning", engine, direction, "template.png"))
                        with self.assertRaises(FrozenInstanceError):
                            spec.direction_id = "C"
                        path = render_template(spec, (160, 256))
                        with Image.open(path) as image:
                            primary = image.getpixel((80, 60))
                            self.assertEqual(image.getpixel((80, 35)), primary)
                            self.assertEqual(image.getpixel((80, 150)), primary)
                            colours = image.getcolors(160 * 256)
                            self.assertEqual(len(colours), 2)
                            self.assertGreater(max(count for count, _ in colours), 160 * 256 * .6)
                            layouts.add((direction, image.tobytes()))
                self.assertEqual(len(layouts), 3)

    def test_retry_adjustments_keep_unrelated_contracts(self):
        with TemporaryDirectory() as temporary:
            with patch.object(settings, "programme_data_path", Path(temporary)):
                initial = compile_template_conditioning(candidate("sd-3-5-medium", "A"))
                dense = compile_template_conditioning(candidate("sd-3-5-medium", "A"), failure_category="DENSE_SAFE_REGION")
                weak = compile_template_conditioning(candidate("sd-3-5-medium", "A"), failure_category="WEAK_OUTER_STRUCTURE")
                palette = compile_template_conditioning(candidate("sd-3-5-medium", "A"), failure_category="PALETTE_OFF")
                primary = compile_template_conditioning(candidate("sd-3-5-medium", "A"), failure_category="PRIMARY_MISSING")
                secondary = compile_template_conditioning(candidate("sd-3-5-medium", "A"), failure_category="SECONDARY_MISSING")
                raster = compile_template_conditioning(candidate("sd-3-5-medium", "A"), failure_category="RASTER")
                human = compile_template_conditioning(candidate("sd-3-5-medium", "A"), failure_category="HUMAN")
                self.assertEqual(dense.secondary_fraction, initial.secondary_fraction)
                self.assertGreater(dense.quiet_padding, initial.quiet_padding)
                self.assertGreater(weak.edge_fraction, initial.edge_fraction)
                self.assertEqual((palette.edge_fraction, palette.quiet_padding),
                                 (initial.edge_fraction, initial.quiet_padding))
                self.assertLess(palette.denoise, initial.denoise)
                self.assertLess(primary.secondary_fraction, initial.secondary_fraction)
                self.assertGreater(secondary.secondary_fraction, initial.secondary_fraction)
                self.assertTrue(raster.simplified)
                self.assertEqual((human.edge_fraction, human.denoise),
                                 (initial.edge_fraction, initial.denoise))
                self.assertNotEqual(dense.conditioning_sha256, initial.conditioning_sha256)

    def test_text_fallback_and_core_img2img_paths_for_all_engines(self):
        for engine, sampler, empty, vae in (
            ("sdxl-1-0", "3", "53", ["4", 2]),
            ("sd-3-5-medium", "3", "53", ["4", 2]),
            ("flux-2", "11", "10", ["3", 0]),
        ):
            with self.subTest(engine=engine):
                text = build_workflow(engine, "art", "", 123, "probe")
                self.assertIn(empty, text)
                self.assertNotIn("90", text)
                image = build_workflow(engine, "art", "", 123, "probe",
                                       template_image="template.png", denoise=.55)
                self.assertNotIn(empty, image)
                self.assertEqual(image["90"]["class_type"], "LoadImage")
                self.assertEqual(image["91"]["inputs"]["vae"], vae)
                self.assertEqual(image[sampler]["inputs"]["latent_image"], ["91", 0])
                if engine == "flux-2":
                    self.assertEqual(image["92"]["class_type"], "SplitSigmasDenoise")
                    self.assertEqual(image["11"]["inputs"]["sigmas"], ["92", 1])
                else:
                    self.assertEqual(image["3"]["inputs"]["denoise"], .55)

    def test_invalid_template_or_missing_node_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, "uploaded ComfyUI input"):
            build_workflow("flux-2", "art", "", 1, "probe", template_image="/tmp/template.png", denoise=.5)
        with self.assertRaisesRegex(ValueError, "denoise"):
            build_workflow("sdxl-1-0", "art", "", 1, "probe", template_image="template.png", denoise=1)
        workflow = build_workflow("sdxl-1-0", "art", "", 1, "probe", template_image="template.png", denoise=.5)
        with self.assertRaises(ComfyUIError):
            validate_workflow(workflow, {})
