import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.config.settings import settings
from app.services.prompt_compiler import failure_category
from app.services.template_adherence import assess_template_adherence
from app.services.template_conditioning import compile_template_conditioning, render_template
from test_template_conditioning import candidate


class TemplateAdherenceTests(unittest.TestCase):
    def test_occupancy_and_protected_maps_and_spatial_rejections(self):
        with TemporaryDirectory() as directory, patch.object(settings, "programme_data_path", Path(directory)):
            spec = compile_template_conditioning(candidate("sdxl-1-0", "A"))
            template = render_template(spec, (832, 1216))
            self.assertTrue(template.with_name("occupancy-map.png").is_file())
            self.assertTrue(template.with_name("protected-mask.png").is_file())
            for mode, expected in (
                ("edge", None),
                ("title", "PROTECTED_REGION_INTRUSION"),
                ("programme", "PROTECTED_REGION_INTRUSION"),
                ("blank", "WEAK_TEMPLATE_STRUCTURE"),
                ("drift", "TEMPLATE_DRIFT"),
            ):
                with self.subTest(mode=mode):
                    image = Image.new("RGB", (832, 1216), "white")
                    draw = ImageDraw.Draw(image)
                    if mode != "blank":
                        for x in (15, 80, 740, 810):
                            for y in range(0, 1216, 25):
                                draw.line((x, y, x + 20, y + 12), fill="black", width=5)
                    if mode in {"title", "programme"}:
                        top, bottom = (130, 260) if mode == "title" else (550, 1000)
                        for y in range(top, bottom, 12):
                            draw.line((130, y, 700, y), fill="black", width=4)
                    if mode == "drift":
                        for y in list(range(0, 105, 15)) + list(range(315, 525, 15)) + list(range(1070, 1216, 15)):
                            draw.line((120, y, 715, y), fill="black", width=5)
                    path = Path(directory) / f"{mode}.png"
                    image.save(path)
                    metrics = assess_template_adherence(path, spec)
                    self.assertEqual(metrics["template_failure_category"], expected)
                    self.assertEqual(metrics["template_adherence_passed"], expected is None)
                    self.assertIn("template_structure_iou", metrics)

    def test_retry_adjustments_and_classification(self):
        with TemporaryDirectory() as directory, patch.object(settings, "programme_data_path", Path(directory)):
            spec = candidate("flux-2", "B")
            base = compile_template_conditioning(spec)
            drift = compile_template_conditioning(spec, failure_category="TEMPLATE_DRIFT", attempt=2)
            intrusion = compile_template_conditioning(spec, failure_category="PROTECTED_REGION_INTRUSION", attempt=3)
            weak = compile_template_conditioning(spec, failure_category="WEAK_TEMPLATE_STRUCTURE", attempt=4)
            decorative = compile_template_conditioning(spec, failure_category="WEAK_DECORATIVE_DESIGN", attempt=4)
            self.assertLess(base.denoise, .95)
            self.assertLess(drift.denoise, base.denoise)
            self.assertLess(intrusion.denoise, base.denoise)
            self.assertGreater(intrusion.quiet_padding, base.quiet_padding)
            self.assertGreater(weak.edge_fraction, base.edge_fraction)
            self.assertGreater(decorative.edge_fraction, base.edge_fraction)
            self.assertNotEqual(drift.template_path, base.template_path)
            for category in ("TEMPLATE_DRIFT", "PROTECTED_REGION_INTRUSION", "WEAK_TEMPLATE_STRUCTURE", "WEAK_DECORATIVE_DESIGN"):
                self.assertEqual(failure_category(category), category)
