import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from app.config.settings import settings
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import build_workflow
from app.services.composition_diversity import direction_similarity_metrics, compare_directions_metrics
from app.services.prompt_compiler import compile_candidate_prompt, failure_category
from app.services.sampling_profiles import (
    FailureHistorySummary, probe_profiles, select_sampling_profile,
    profile_selection_diagnostics, recommended_probe_profiles,
)
from app.services.subject_adherence import assess_subject_adherence, _model_answer, select_vision_endpoint


def document(subject="A layered botanical garden with paper flowers", primary="green", secondary="cream"):
    return {"reference_number": "ABCDEF-123456", "brief": {
        "event_type": "garden celebration", "theme": "botanical garden",
        "background_inspiration": subject, "primary_colour": primary,
        "secondary_colour": secondary, "palette_mode": "dominant",
        "palette_relationship": "green foliage dominates with cream highlights",
        "environmental_motifs": ["layered garden paths", "leaf canopies"],
        "symbolic_motifs": ["paper flowers"], "visual_style": "illustrated dimensional environment",
    }, "directions": [{"direction": {
        "direction_id": "A", "role": "Architectural Master",
        "positive_prompt": "Layered botanical garden with paper flowers and winding paths",
        "negative_prompt": "people, lettering",
        "composition_strategy": "asymmetric botanical pathway through layered plants",
        "focal_strategy": "off-centre flowering arch",
        "depth_strategy": "foreground foliage, middle path and distant garden",
        "border_or_perimeter_strategy": "open irregular leaf perimeter",
        "title_safe_region_strategy": "calm upper sky",
        "programme_safe_region_strategy": "quiet lower path",
        "layout_guidance": {},
    }}]}


class RendererConditioningTests(unittest.TestCase):
    def test_concise_palette_roles_survive_explicit_relationship_in_both_engines(self):
        for primary, secondary in (("red", "gold"), ("green", "purple"),
                                   ("black", "yellow"), ("blue", "red")):
            brief = document("Abstract geometric celebration background", primary, secondary)
            brief["brief"]["palette_relationship"] = (
                f"Use {primary} as the main colour and {secondary} as the secondary colour")
            brief["brief"]["palette_relationship_source"] = "user_explicit"
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                with self.subTest(primary=primary, secondary=secondary, engine=engine):
                    spec = build_candidate_spec(brief, engine, "A")
                    for attempt, reason in ((1, ""), (2, "PALETTE_ROLE_DRIFT: secondary missing")):
                        prompt = compile_candidate_prompt(spec, attempt, reason, concise=True)
                        for channel in ("clip_g", "positive") + (("t5",) if engine == "sd-3-5-medium" else ()):
                            value = getattr(prompt, channel).casefold()
                            self.assertIn(f"primary {primary}", value)
                            self.assertIn(f"secondary {secondary}", value)
                        full_channel = prompt.clip_l if engine == "sdxl-1-0" else prompt.t5
                        for value in (prompt.positive, full_channel):
                            self.assertIn(brief["brief"]["palette_relationship"].casefold(),
                                          value.casefold())

    def test_sd35_figure_recovery_aligns_all_encoders_through_palette_retry(self):
        motif = "layered geometric shapes and circular energy patterns"
        brief = document(f"Comic-style technology superhero background with {motif}", "red", "gold")
        brief["brief"]["grill_me_context"] = {"generation_constraints": ["No people or characters"]}
        brief["brief"]["visual_semantics"] = {
            "motif_descriptors": [motif], "motif_structure_description": motif,
            "atmosphere_descriptors": ["fast and powerful"],
        }
        spec = build_candidate_spec(brief, "sd-3-5-medium", "A")
        initial = compile_candidate_prompt(spec, 1, concise=True)
        self.assertIn("superhero", initial.clip_l.casefold())
        for attempt, reason, history in (
            (2, "HUMAN: figure visible", None),
            (4, "PALETTE_OFF: other colours", {"HUMAN": 1, "PALETTE_OFF": 2}),
            (7, "PALETTE_ROLE_DRIFT: gold missing", {"CHARACTER": 2, "PALETTE_OFF": 3}),
        ):
            with self.subTest(attempt=attempt):
                prompt = compile_candidate_prompt(spec, attempt, reason, concise=True,
                                                  failure_history=history)
                for channel in (prompt.clip_l, prompt.clip_g, prompt.t5):
                    self.assertIn("background", channel.casefold())
                    self.assertNotIn("superhero", channel.casefold())
                for channel in (prompt.clip_g, prompt.t5):
                    self.assertIn(motif, channel.casefold())
                    self.assertIn("distributed", channel.casefold())
                    self.assertIn("primary red", channel.casefold())
                    self.assertIn("secondary gold", channel.casefold())
                self.assertEqual(prompt.spec_sha256, initial.spec_sha256)

    def test_broad_background_subject_does_not_block_figure_recovery(self):
        cases = (("layered angular shapes", "blue", "red"),
                 ("layered geometric shapes and circular energy patterns", "red", "gold"))
        for motif, primary, secondary in cases:
            brief = document("A comic-style technology superhero background for a child's birthday",
                             primary, secondary)
            brief["brief"]["theme_reference_treatment"] = (
                f"Comic-book style with {motif}, energetic and superhero-like, as abstract background artwork")
            brief["brief"]["grill_me_context"] = {"generation_constraints": ["No people or characters"]}
            brief["brief"]["visual_semantics"] = {
                "motif_descriptors": [motif], "motif_structure_description": motif}
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                with self.subTest(motif=motif, engine=engine):
                    spec = build_candidate_spec(brief, engine, "A")
                    self.assertTrue(spec.visual_design["authorized_literal_subject"])
                    retry = compile_candidate_prompt(spec, 4, "PALETTE_OFF: drift", concise=True,
                                                     failure_history={"HUMAN": 1, "PALETTE_OFF": 2})
                    for channel in (retry.clip_l, retry.clip_g) + ((retry.t5,) if retry.t5 else ()):
                        self.assertNotIn("superhero", channel.casefold())
                        self.assertNotIn("web", channel.casefold())
                        self.assertIn("background", channel.casefold())
                    self.assertIn(motif, retry.positive.casefold())
                    self.assertIn(f"primary {primary}", retry.clip_g.casefold())
                    self.assertIn(f"secondary {secondary}", retry.clip_g.casefold())

    def test_explicit_concrete_subject_survives_figure_recovery(self):
        brief = document("A wedding cake", "pink", "white")
        brief["brief"]["grill_me_context"] = {"generation_constraints": ["No people or characters"]}
        brief["brief"]["visual_semantics"] = {
            "motif_descriptors": ["floral scrollwork"],
            "motif_structure_description": "floral scrollwork"}
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            spec = build_candidate_spec(brief, engine, "A")
            self.assertEqual(spec.visual_design["authorized_literal_subject"], "A wedding cake")
            retry = compile_candidate_prompt(spec, 2, "HUMAN: figure visible", concise=True)
            self.assertIn("wedding cake", retry.clip_l.casefold())
            self.assertIn("wedding cake", retry.clip_g.casefold())

    def test_sd35_literal_subject_drift_uses_approved_motif(self):
        brief = document("A comic-style superhero background", "black", "yellow")
        brief["brief"]["grill_me_context"] = {"generation_constraints": ["No people or characters"]}
        brief["brief"]["visual_semantics"] = {
            "motif_descriptors": ["bat shapes"], "motif_structure_description": "bat shapes"}
        spec = build_candidate_spec(brief, "sd-3-5-medium", "A")
        retry = compile_candidate_prompt(spec, 2, "SUBJECT_DRIFT: humanoid figure displaced motif",
                                         concise=True)
        for channel in (retry.clip_l, retry.clip_g, retry.t5):
            self.assertIn("bat shapes", channel.casefold())
            self.assertNotIn("superhero", channel.casefold())

    def test_sd35_figure_recovery_clip_g_stays_within_budget(self):
        motif = "Layered geometric shapes and circular energy patterns distributed throughout the background"
        brief = document("An comic-style technology superhero background for a kid's birthday with no buildings",
                         "red", "gold")
        brief["brief"].update(
            event_type="Children birthday party", audience_context="Children aged 7 to 10",
            theme_reference_treatment=("I want a comic-book style background with clean futuristic tech patterns "
                                       "and glowing energy-like details; Use red as the main colour and gold "
                                       "as the secondary colour, with layered geometric shapes and circular "
                                       "energy patterns distributed throughout the background"),
            palette_relationship=("Use red as the main colour and gold as the secondary colour, "
                                  "with layered geometric shapes and circular energy patterns "
                                  "distributed throughout the background"),
            palette_relationship_source="user_explicit",
            grill_me_context={"generation_constraints": ["No people or characters"]},
            visual_semantics={"motif_descriptors": ["geometric shapes", "circular energy patterns"],
                              "motif_structure_description": motif,
                              "atmosphere_descriptors": ["modern", "fast", "powerful"]})
        spec = build_candidate_spec(brief, "sd-3-5-medium", "A")
        retry = compile_candidate_prompt(spec, 4, "PALETTE_OFF: drift", concise=True,
                                         failure_history={"HUMAN": 1, "PALETTE_OFF": 2})
        self.assertFalse(retry.record()["prompt_budget"]["clip_g"]["budget_risk"])
        self.assertIn("circular energy patterns", retry.clip_g.casefold())
        self.assertIn("secondary gold", retry.clip_g.casefold())
        self.assertIn(brief["brief"]["palette_relationship"], retry.t5)

    def test_sdxl_figure_and_palette_retries_keep_approved_background_motifs(self):
        cases = (("bat shapes", "black", "yellow"),
                 ("cracked shapes and impact bursts", "green", "purple"),
                 ("layered geometric shapes and circular energy patterns", "red", "gold"))
        for motif, primary, secondary in cases:
            with self.subTest(motif=motif):
                brief = document(f"Comic-style superhero background with {motif}", primary, secondary)
                brief["brief"]["grill_me_context"] = {"generation_constraints": ["No people or characters"]}
                brief["brief"]["visual_semantics"] = {"motif_descriptors": [motif],
                                                        "motif_structure_description": motif}
                spec = build_candidate_spec(brief, "sdxl-1-0", "A")
                initial = compile_candidate_prompt(spec, 1, concise=True)
                figure = compile_candidate_prompt(spec, 2, "CHARACTER: costumed figure", concise=True)
                palette = compile_candidate_prompt(
                    spec, 3, "PALETTE_OFF: unrelated colour", concise=True,
                    failure_history={"CHARACTER": 1, "PALETTE_OFF": 1})
                palette_only = compile_candidate_prompt(
                    spec, 2, "PALETTE_OFF: unrelated colour", concise=True)
                self.assertIn("superhero", initial.clip_l.casefold())
                for retry in (figure, palette, palette_only):
                    self.assertIn(motif, retry.positive.casefold())
                    self.assertIn(motif, retry.clip_l.casefold())
                    self.assertNotIn(motif, retry.negative.casefold())
                    self.assertNotIn("superhero", retry.clip_l.casefold())
                    self.assertIn("background", retry.clip_l.casefold())
                    self.assertIn("distributed", retry.clip_g.casefold())
                    self.assertIn("figure", retry.negative.casefold())
                    self.assertLess(retry.clip_l.casefold().index(primary),
                                    retry.clip_l.casefold().index(secondary))
                    self.assertEqual(retry.spec_sha256, initial.spec_sha256)
                self.assertIn("distributed", palette_only.clip_g.casefold())

    def test_sd35_encoders_put_concrete_environment_before_style_and_palette(self):
        spec = build_candidate_spec(document(), settings.engine_3_id, "A")
        compiled = compile_candidate_prompt(spec, 1)
        for field in (compiled.clip_l, compiled.clip_g, compiled.t5):
            self.assertIn(spec.background_inspiration, field)
            self.assertNotIn(spec.direction_role, field)
            self.assertLess(field.index(spec.background_inspiration), field.find("Palette:") if "Palette:" in field else len(field))
        self.assertIn("layered garden paths", compiled.t5)
        self.assertIn("paper flowers", compiled.t5)
        self.assertIn("asymmetric botanical pathway", compiled.t5)
        self.assertIn("cream", compiled.clip_g)
        self.assertLess(len(compiled.clip_l.split()), 40)
        self.assertLess(len(compiled.t5.split()), 230)
        workflow = build_workflow(settings.engine_3_id, compiled.positive, compiled.negative,
                                  12345, "test/sd35", compiled_prompt=compiled)
        self.assertEqual(workflow["16"]["inputs"]["clip_l"], compiled.clip_l)
        self.assertEqual(workflow["16"]["inputs"]["clip_g"], compiled.clip_g)
        self.assertEqual(workflow["16"]["inputs"]["t5xxl"], compiled.t5)

    def test_sdxl_encoders_are_complementary_and_subject_led(self):
        spec = build_candidate_spec(document(), settings.engine_2_id, "A")
        compiled = compile_candidate_prompt(spec, 1)
        self.assertNotEqual(compiled.clip_l, compiled.clip_g)
        for field in (compiled.clip_l, compiled.clip_g):
            self.assertIn(spec.background_inspiration, field)
            self.assertIn("green", field)
            self.assertIn("cream", field)
            self.assertIn("asymmetric botanical pathway", field)
            self.assertNotIn(spec.direction_role, field)
        workflow = build_workflow(settings.engine_2_id, compiled.positive, compiled.negative,
                                  12345, "test/sdxl", compiled_prompt=compiled)
        self.assertEqual(workflow["16"]["inputs"]["text_l"], compiled.clip_l)
        self.assertEqual(workflow["16"]["inputs"]["text_g"], compiled.clip_g)

    def test_duplicate_recovery_escalates_without_changing_spec(self):
        spec = build_candidate_spec(document(), settings.engine_2_id, "A")
        prompts = [compile_candidate_prompt(spec, attempt, "DIRECTION_DUPLICATE") for attempt in (2, 4, 7, 9)]
        self.assertEqual(len({item.spec_sha256 for item in prompts}), 1)
        self.assertEqual(len({item.positive for item in prompts}), 4)
        for item in prompts:
            self.assertIn(spec.background_inspiration, item.positive)
            self.assertIn("green", item.positive)
            self.assertNotIn(spec.direction_role, item.positive)
        self.assertEqual(failure_category("SUBJECT_DRIFT: environment absent"), "SUBJECT_DRIFT")
        rescue = compile_candidate_prompt(spec, 2, "SUBJECT_DRIFT: environment absent")
        self.assertIn("recognizable subject-specific forms", rescue.positive)
        self.assertNotIn("Change focal location", rescue.positive)

    def test_probe_matrix_and_failure_aware_selection(self):
        for engine, names in ((settings.engine_2_id, {"baseline", "cfg_5_5", "cfg_6_5", "dpmpp_normal", "composition_variability"}),
                              (settings.engine_3_id, {"alternate_sampler", "official_baseline", "lower_guidance_trial",
                                                        "higher_guidance_trial", "alternate_schedule_trial", "subject_rescue_trial", "structural_rescue_trial"})):
            self.assertTrue(names <= {item.name for item in probe_profiles(engine)})
            self.assertLessEqual(len(recommended_probe_profiles(engine)), 8)
            self.assertTrue({item.name for item in recommended_probe_profiles(engine)} <=
                            {item.name for item in probe_profiles(engine)})
        for category, expected in (("SUBJECT_DRIFT", "subject_rescue_trial"),
                                   ("FLAT", "structural_rescue_trial")):
            history = FailureHistorySummary({category: 1}, category, 1)
            profile = select_sampling_profile(settings.engine_3_id, 2, history)
            self.assertEqual(profile.name, expected)
            self.assertEqual(profile_selection_diagnostics(profile, history)["failure_history"]["consecutive_category"], category)
        raster_history = FailureHistorySummary({"RASTER": 2}, "RASTER", 2)
        self.assertEqual(select_sampling_profile(settings.engine_3_id, 3, raster_history).name, "raster_rescue_trial")
        history = FailureHistorySummary({"DIRECTION_DUPLICATE": 1}, "DIRECTION_DUPLICATE", 1)
        self.assertEqual(select_sampling_profile(settings.engine_2_id, 2, history).name, "composition_variability")


class SubjectAdherenceTests(unittest.TestCase):
    def test_model_specific_endpoint_selection(self):
        self.assertEqual(select_vision_endpoint("minicpm-v4.5:latest"), "/api/chat")
        self.assertEqual(select_vision_endpoint("MINICPM-V4.5:Q8"), "/api/chat")
        self.assertEqual(select_vision_endpoint("qwen3-vl:30b"), "/api/generate")
        self.assertEqual(select_vision_endpoint("other-vision:latest"), "/api/generate")

    def test_minicpm_chat_response_uses_message_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (16, 16), "white").save(path)
            response = Mock(status_code=200, headers={"content-type": "application/json"})
            response.raise_for_status.return_value = None
            answer = '{"score":0.95,"observed_evidence":["angular architecture"],"reason":"city visible"}'
            for content in (answer, f"```json\n{answer}\n```", f"Result:\n{answer}", "", "{broken}"):
                with self.subTest(content=content):
                    response.json.return_value = {"message": {"content": content, "thinking": "private"}}
                    with patch("app.services.subject_adherence.httpx.post", return_value=response) as send:
                        result = assess_subject_adherence(path, {"primary_visual_subject": "city"}, model="minicpm-v4.5:latest")
                    self.assertTrue(send.call_args.args[0].endswith("/api/chat"))
                    request = send.call_args.kwargs["json"]
                    self.assertIn("messages", request)
                    self.assertNotIn("prompt", request)
                    self.assertEqual(result["vision_debug"]["endpoint"], "/api/chat")
                    if content in ("", "{broken}"):
                        self.assertFalse(result["checked"])
                        self.assertEqual(result["decision"], "unavailable")
                    else:
                        self.assertTrue(result["checked"])
                        self.assertEqual(result["score"], .95)
                        self.assertEqual(result["vision_debug"]["final_source"], "message.content")

    def test_ollama_envelopes_and_wrapped_json(self):
        answer = '{"score":0.82,"observed_evidence":["skyline"]}'
        for endpoint, payload in (("/api/generate", {"response": answer}),
                                  ("/api/chat", {"message": {"content": answer}}),
                                  ("/api/generate", {"response": f"```json\n{answer}\n```"}),
                                  ("/api/chat", {"message": {"content": f"Analysis: visible skyline.\n{answer}"}})):
            with self.subTest(endpoint=endpoint, payload=payload):
                self.assertEqual(_model_answer(payload, endpoint)["score"], .82)

    def test_code_fence_parser_preserves_json_and_handles_long_unfenced_input(self):
        answer = '{"score":0.82,"observed_evidence":["skyline"]}'
        for content in (f"```json\n{answer}\n```", f"```JSON\n{answer}\n```",
                        f"```\n{answer}\n```", answer):
            with self.subTest(content=content[:12]):
                self.assertEqual(_model_answer({"response": content}, "/api/generate")["score"], .82)
        with self.assertRaises(ValueError):
            _model_answer({"response": "x" + " " * 12000 + "y"}, "/api/generate")

    def test_invalid_and_empty_envelopes_fail(self):
        for content in ("", "not JSON", "{broken}"):
            with self.subTest(content=content), self.assertRaises((ValueError, json.JSONDecodeError)):
                _model_answer({"response": content}, "/api/generate")

    def test_malformed_and_http_failure_are_unavailable_with_bounded_debug(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (16, 16), "white").save(path)
            response = Mock(status_code=200, headers={"content-type": "application/json"})
            response.json.return_value = {"response": "not JSON" * 1000}
            response.raise_for_status.return_value = None
            with patch("app.services.subject_adherence.httpx.post", return_value=response):
                result = assess_subject_adherence(path, {"primary_visual_subject": "garden"}, model="qwen3-vl:30b")
            self.assertFalse(result["checked"])
            self.assertEqual(result["decision"], "unavailable")
            self.assertLessEqual(len(result["vision_debug"]["raw_text_preview"]), 400)
            response.raise_for_status.side_effect = __import__("httpx").HTTPStatusError("failed", request=Mock(), response=Mock())
            with patch("app.services.subject_adherence.httpx.post", return_value=response):
                result = assess_subject_adherence(path, {"primary_visual_subject": "garden"}, model="qwen3-vl:30b")
            self.assertFalse(result["checked"])

    def test_thinking_only_response_remains_unchecked_and_think_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (16, 16), "white").save(path)
            response = Mock(status_code=200, headers={"content-type": "application/json"})
            response.json.return_value = {"response": "", "thinking": "private reasoning " * 1000}
            response.raise_for_status.return_value = None
            with patch("app.services.subject_adherence.httpx.post", return_value=response) as send:
                result = assess_subject_adherence(path, {"primary_visual_subject": "garden"}, model="qwen3-vl:30b")
            self.assertIs(send.call_args.kwargs["json"]["think"], False)
            self.assertEqual(result["decision"], "unavailable")
            self.assertTrue(result["vision_debug"]["thinking_present"])
            self.assertEqual(result["vision_debug"]["response_length"], 0)
            self.assertIsNone(result["vision_debug"]["final_source"])
            self.assertNotIn("private reasoning", str(result))

    def assess(self, answer):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (128, 128), "white").save(path)
            response = Mock()
            response.json.return_value = {"response": json.dumps(answer)}
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.raise_for_status.return_value = None
            with patch("app.services.subject_adherence.httpx.post", return_value=response):
                return assess_subject_adherence(path, {"primary_visual_subject": "layered botanical garden",
                                                "visual_anchors": ["flower beds", "garden path"]}, model="vision-test")

    def test_visible_subject_passes_and_abstract_drift_fails(self):
        passed = self.assess({"score": .82, "observed_evidence": ["flowers", "path"], "reason": "garden visible"})
        failed = self.assess({"score": .19, "observed_evidence": ["colour fields"], "reason": "no garden forms"})
        self.assertTrue(passed["checked"])
        self.assertEqual(passed["decision"], "pass")
        self.assertEqual(failed["decision"], "SUBJECT_DRIFT")
        self.assertEqual(failed["anchors"], ["flower beds", "garden path"])

    def test_unconfigured_checker_is_explicitly_unchecked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (16, 16), "white").save(path)
            result = assess_subject_adherence(path, {"primary_visual_subject": "abstract sculptural forms"}, model="")
        self.assertFalse(result["checked"])
        self.assertIsNone(result["score"])


class CompositionAwareSimilarityTests(unittest.TestCase):
    def image(self, left):
        image = Image.new("RGB", (400, 600), "#4168b0")
        draw = ImageDraw.Draw(image)
        for x in range(0, 400, 20):
            draw.line((x, 0, 400 - x, 599), fill="#101a35", width=2)
        if left:
            draw.rectangle((15, 90, 160, 580), fill="#16254b")
        else:
            draw.rectangle((245, 90, 390, 580), fill="#16254b")
        return image

    def test_matching_theme_lines_do_not_override_different_visual_mass(self):
        first = self.image(True)
        different = self.image(False)
        same = self.image(True)
        self.assertLess(direction_similarity_metrics(first, different)["similarity"], .88)
        self.assertGreaterEqual(direction_similarity_metrics(first, same)["similarity"], .88)
        self.assertIn("coarse_mass_iou", direction_similarity_metrics(first, different))
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.png"
            peer = Path(directory) / "peer.png"
            different.save(candidate)
            first.save(peer)
            diagnostics = compare_directions_metrics(candidate, [peer])
        self.assertEqual(diagnostics["closest_peer"]["peer"], str(peer))
        self.assertLess(diagnostics["similarity"], .88)
