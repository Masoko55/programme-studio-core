import hashlib
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from app.config.settings import settings
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import ComfyUIClient, ComfyUIError, build_workflow, validate_workflow
from app.services.template_conditioning import (
    compile_template_conditioning, persisted_conditioning_spec, render_template,
    verify_conditioning_artifacts,
)


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
    def test_client_validates_retry_against_its_saved_attempt(self):
        with TemporaryDirectory() as temporary, patch.object(settings, "programme_data_path", Path(temporary)):
            candidate_spec = candidate("flux-2", "B")
            spec = compile_template_conditioning(candidate_spec,
                                                 failure_category="PROTECTED_REGION_INTRUSION", attempt=2)
            render_template(spec, (160, 256))
            directory = Path(temporary) / "backgrounds"
            directory.mkdir()
            downloaded = directory / "candidate.part.png"
            Image.new("RGB", (160, 256), "white").save(downloaded)
            output = directory / "image-b.png"
            conditioning = spec.record()
            conditioning.update(enabled=True, retry_adjustment="PROTECTED_REGION_INTRUSION",
                                attempt_conditioning_sha256=spec.conditioning_sha256,
                                template_sha256=hashlib.sha256(Path(spec.template_path).read_bytes()).hexdigest())
            record = {"reference_number": spec.reference_number, "engine_id": spec.engine_id,
                      "direction_id": spec.direction_id, "attempt_count": 2,
                      "spec_sha256": candidate_spec.spec_sha256, "retry_stage": "normal",
                      "template_conditioning": conditioning}
            with patch("app.services.comfyui_client.assess_template_adherence",
                       return_value={"template_adherence_passed": True}), \
                 patch("app.services.comfyui_client.validate_background",
                       return_value={"sha256": "validated"}):
                validation, *_ = ComfyUIClient()._validate_downloaded_candidate(
                    record, True, output, downloaded, {}, ("royal blue", "red")
                )
            self.assertEqual(validation["sha256"], "validated")
            self.assertTrue(output.is_file())

    def test_retry_sha_restores_exact_attempt_without_double_adjustment(self):
        with TemporaryDirectory() as temporary, patch.object(settings, "programme_data_path", Path(temporary)):
            candidate_spec = candidate("flux-2", "B")
            first = compile_template_conditioning(candidate_spec)
            second = compile_template_conditioning(
                candidate_spec, failure_category="PROTECTED_REGION_INTRUSION", attempt=2,
            )
            render_template(first, (160, 256))
            render_template(second, (160, 256))
            self.assertNotEqual(first.conditioning_sha256, second.conditioning_sha256)
            self.assertNotEqual(first.template_path, second.template_path)
            record = second.record()
            record["attempt_conditioning_sha256"] = second.conditioning_sha256
            record["template_sha256"] = hashlib.sha256(Path(second.template_path).read_bytes()).hexdigest()
            self.assertEqual(persisted_conditioning_spec(record).conditioning_sha256,
                             second.conditioning_sha256)
            verify_conditioning_artifacts(record)
            with self.assertRaisesRegex(ValueError, "metadata SHA mismatch"):
                persisted_conditioning_spec({**record, "denoise": .5})
            Path(second.template_path).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "template SHA mismatch"):
                verify_conditioning_artifacts(record)

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
                        with Image.open(path.with_name("generation-mask.png")) as mask:
                            self.assertLess(mask.getpixel((80, 35)), mask.getpixel((5, 128)))
                            self.assertLess(mask.getpixel((80, 150)), mask.getpixel((5, 128)))
                            self.assertGreater(mask.getpixel((80, 70)), 0)
                            self.assertLess(mask.getpixel((80, 70)), 255)
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
                masked = build_workflow(engine, "art", "", 123, "probe",
                                        template_image="template.png", denoise=.55,
                                        generation_mask="mask.png")
                self.assertEqual(masked["93"]["class_type"], "LoadImageMask")
                self.assertEqual(masked["94"]["class_type"], "SetLatentNoiseMask")
                self.assertEqual(masked[sampler]["inputs"]["latent_image"], ["94", 0])

    def test_invalid_template_or_missing_node_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, "uploaded ComfyUI input"):
            build_workflow("flux-2", "art", "", 1, "probe", template_image="/tmp/template.png", denoise=.5)
        with self.assertRaisesRegex(ValueError, "denoise"):
            build_workflow("sdxl-1-0", "art", "", 1, "probe", template_image="template.png", denoise=1)
        workflow = build_workflow("sdxl-1-0", "art", "", 1, "probe", template_image="template.png", denoise=.5)
        with self.assertRaises(ComfyUIError):
            validate_workflow(workflow, {})
