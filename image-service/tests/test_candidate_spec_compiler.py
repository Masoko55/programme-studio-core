import unittest

from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import (_background_only, _treatment_covered, compile_candidate_prompt,
                                          failure_category, runtime_retry_category)
from app.services.comfyui_client import build_workflow
from app.services.sampling_profiles import FailureHistorySummary


CASES = (
    (
        "children's birthday", "Spider-Man city adventure",
        "A stylised comic-book city skyline with bold angular architecture",
        "Spiderweb-inspired geometry and dynamic motion shapes; no character",
        "expressive", "royal blue", "red", "city skyline", "spiderweb",
    ),
    (
        "30th birthday", "black-tie celebration",
        "A formal ballroom with chandeliers and black-tie ornament",
        "Architectural ballroom details in crisp monochrome",
        "elegant", "black", "white", "ballroom", "black-tie",
    ),
    (
        "bridal shower", "paper flower garden",
        "Large layered paper flowers around the border",
        "Floral paper craft with visible petals and leaves",
        "contemporary", "pink", "white", "paper flowers", "floral",
    ),
)

REASONS = (
    "",
    "Generated background contains colours that were not requested (off-palette ratio 0.5)",
    "Generated background contains a human silhouette",
    "Generated background contains readable text",
    "Generated background became a smooth gradient colour wash",
    "Outer background area contains too little useful decorative structure",
    "Generated background has repetitive raster lines",
    "Generated background failed due to runtime timeout",
    "Generated background has missing secondary colour",
)

DENSE_SAFE_REGION_REASON = (
    "Generated background failed visual quality validation: "
    "programme safe region is too visually dense "
    "(center structural edge density 0.123, "
    "outer structural edge density 0.119, "
    "center-to-outer ratio 1.03)."
)


def document_for(case, direction_id="B"):
    event, theme, inspiration, treatment, role, primary, secondary, *_ = case
    return {
        "reference_number": "ABCDEF-123456",
        "brief": {
            "event_type": event,
            "theme": theme,
            "creative_description": f"A distinctive {event} background",
            "background_inspiration": inspiration,
            "theme_reference_treatment": treatment,
            "primary_colour": primary,
            "secondary_colour": secondary,
        },
        "directions": [{
            "direction": {
                "direction_id": direction_id,
                "role": role,
                "positive_prompt": f"Render {inspiration} with {treatment}",
                "negative_prompt": "people, readable text",
                "composition_strategy": "asymmetric layered composition",
                "focal_strategy": "off-centre subject focus",
                "depth_strategy": "foreground to background depth",
                "border_or_perimeter_strategy": "open irregular perimeter",
                "title_safe_region_strategy": "quiet upper reading area",
                "programme_safe_region_strategy": "quiet lower reading area",
                "layout_guidance": {
                    "title_zone": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.12},
                    "programme_zone": {"x": 0.1, "y": 0.45, "width": 0.8, "height": 0.4},
                },
            }
        }],
    }


class CandidateSpecCompilerTests(unittest.TestCase):
    def test_final_workflow_keeps_text_exclusions_for_each_engine(self):
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(document_for(CASES[0]), engine, "B")
            compiled = compile_candidate_prompt(spec, 1, concise=True)
            workflow = build_workflow(engine, compiled.positive, compiled.negative,
                                      123, "offline", compiled_prompt=compiled)
            if engine == "flux-2":
                active = workflow["4"]["inputs"]["text"].lower()
            elif engine == "sdxl-1-0":
                active = workflow["40"]["inputs"]["text"].lower()
            else:
                active = workflow["40"]["inputs"]["t5xxl"].lower()
            for prohibited in ("words", "letters", "numbers", "logos", "signage", "watermark"):
                self.assertIn(prohibited, active)

    def test_background_only_preserves_negative_meaning(self):
        spec = build_candidate_spec(document_for(CASES[0]), "sdxl-1-0", "B")
        for phrase in ("no typography or glyph-like marks", "without text", "no logos", "without logos"):
            cleaned = _background_only(f"Decorative background with {phrase}.", spec).lower()
            self.assertNotIn("no visual calm", cleaned)
            self.assertNotIn("without visual calm", cleaned)
            self.assertIn("continuous image surfaces", cleaned)
        planned = _background_only("Quiet upper area for title and programme", spec).lower()
        self.assertIn("quiet upper area", planned)
        self.assertNotIn("programme", planned)

    def test_concise_production_budget_for_three_generic_briefs(self):
        unseen = document_for(CASES[0])
        unseen["brief"].update(event_type="community lantern walk", theme="river reflections",
                               background_inspiration=("Layered riverside lights and reflections, floating "
                                                       "sculptural fabric, garden forms and a clear passage "
                                                       "leading through several receding spaces"),
                               theme_reference_treatment=("Use a first-person viewpoint and forward movement "
                                                          "through layered riverside forms with curved flow of light "
                                                          "across the foreground and receding depth into the scene"),
                               palette_relationship=("Primary royal blue occupies broad visual mass while "
                                                     "secondary red appears in repeated luminous accents with "
                                                     "neutrals supporting both colour families"),
                               palette_relationship_source="user_explicit")
        for document in (document_for(CASES[0]), document_for(CASES[1]), unseen):
            document["brief"]["audience_context"] = "invited guests"
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                spec = build_candidate_spec(document, engine, "B")
                for attempt in (1, 4):
                    compiled = compile_candidate_prompt(spec, attempt,
                                                        "PALETTE_ROLE_DRIFT" if attempt > 1 else "", concise=True)
                    budget = compiled.record()["prompt_budget"]
                    self.assertFalse(budget["clip_l"]["budget_risk"])
                    self.assertFalse(budget["clip_g"]["budget_risk"])
                    if engine == "sd-3-5-medium":
                        self.assertFalse(budget["t5"]["budget_risk"])
                        for label in ("Event context:", "Audience:", "Event atmosphere:", "Palette:"):
                            self.assertIn(label, compiled.clip_g)
                    self.assertIn(spec.primary_colour, compiled.clip_g)
                    self.assertIn(spec.secondary_colour, compiled.clip_g)

    def test_structural_treatment_coverage_requires_each_feature(self):
        source = "first-person viewpoint with curved flow into receding depth"
        self.assertTrue(_treatment_covered(source, "Treatment: first-person view, curved flow, receding depth"))
        self.assertFalse(_treatment_covered(source, "Treatment: first-person view, receding depth"))
        self.assertFalse(_treatment_covered(source, "Treatment: aerial viewpoint, curved flow, shallow depth"))

    def test_unseen_abstract_brief_keeps_subject_motif_and_palette(self):
        document = document_for(CASES[0])
        document["brief"].update(event_type="community kite exhibition",
                                  theme="interlaced ribbon contours",
                                  background_inspiration="abstract interlaced ribbon contours",
                                  primary_colour="purple", secondary_colour="sand",
                                  visual_semantics={"scene_family": "general",
                                                    "motif_descriptors": ["repeating ribbon contours"]})
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(document, engine, "B")
            prompt = compile_candidate_prompt(spec, 1, concise=True)
            self.assertIn("interlaced ribbon contours", prompt.positive)
            self.assertIn("purple", prompt.positive)
            self.assertIn("sand", prompt.positive)
            self.assertEqual(spec.visual_design["visual_semantics"]["scene_family"], "general")

    def test_sdxl_active_encoders_keep_generic_palette_through_retry_stages(self):
        cases = (("green", "cream", "dominant", ""),
                 ("pink", "gold", "dominant", ""),
                 ("black", "white", "achromatic", ""),
                 ("white", "navy", "dominant", "white is the main field with navy accents"),
                 ("orange", "teal", "dominant", "orange dominant with teal supporting"))
        for primary, secondary, mode, relationship in cases:
            document = document_for(CASES[0])
            document["brief"].update(primary_colour=primary, secondary_colour=secondary,
                                      palette_mode=mode, palette_relationship=relationship,
                                      palette_relationship_source="user_explicit" if relationship else "default")
            spec = build_candidate_spec(document, "sdxl-1-0", "B")
            for attempt in (1, 3, 4, 5, 7, 9):
                reason = "PALETTE_ROLE_DRIFT: roles missing" if attempt > 1 else ""
                compiled = compile_candidate_prompt(spec, attempt, reason, concise=True)
                workflow = build_workflow("sdxl-1-0", compiled.positive, compiled.negative,
                                          12345, "offline-test", compiled_prompt=compiled)
                active = workflow["16"]["inputs"]
                self.assertEqual(workflow["3"]["inputs"]["positive"], ["16", 0])
                self.assertEqual(active["text_l"], compiled.clip_l)
                self.assertEqual(active["text_g"], compiled.clip_g)
                for channel in (active["text_l"], active["text_g"]):
                    self.assertIn(primary, channel)
                    self.assertIn(secondary, channel)
                if relationship:
                    self.assertIn(relationship, active["text_g"])
                if attempt > 1:
                    self.assertIn("palette", active["text_g"].lower())
                self.assertLess(len(active["text_g"]), 1800)

    def test_palette_history_and_latest_failure_both_reach_active_sdxl_encoder(self):
        document = document_for(CASES[0])
        document["brief"].update(primary_colour="orange", secondary_colour="teal",
                                  palette_relationship_source="default")
        spec = build_candidate_spec(document, "sdxl-1-0", "B")
        history = FailureHistorySummary(counts={"PALETTE_ROLE_DRIFT": 4},
                                        consecutive_category="PALETTE_ROLE_DRIFT", consecutive_count=4)
        for reason, cue in (("SUBJECT_DRIFT: wrong subject", "visual anchors"),
                            ("HUMAN: figure", "without figures"),
                            ("AUDIENCE_DRIFT: mood", "audience-appropriate"),
                            ("MOTIF_DRIFT: topology", "motif topology"),
                            ("TREATMENT_DRIFT: viewpoint", "viewpoint and movement")):
            compiled = compile_candidate_prompt(spec, 5, reason, concise=True,
                                                 failure_history=history)
            active = build_workflow("sdxl-1-0", compiled.positive, compiled.negative,
                                    12345, "offline-test", compiled_prompt=compiled)["16"]["inputs"]["text_g"]
            self.assertIn("Persistent palette recovery", active)
            self.assertIn("orange", active)
            self.assertIn("teal", active)
            self.assertIn(cue, active)

    def test_inferred_motifs_remain_patterns_across_unrelated_families(self):
        for motif in ("layered feather-like strokes", "repeating scale texture",
                      "interconnected geometric network"):
            document = document_for(CASES[0])
            document["brief"].update(
                background_inspiration="abstract layered marks",
                visual_semantics={"motif_descriptors": [motif],
                                  "motif_structure_description": motif},
                grill_me_context={"semantic_provenance": {"motif_descriptors": "SEMANTIC_INFERENCE",
                                                          "motif_structure_description": "SEMANTIC_INFERENCE"}},
            )
            spec = build_candidate_spec(document, "sdxl-1-0", "B")
            self.assertEqual(spec.visual_design["motif_literalization_policy"], "motif_only")
            self.assertEqual(spec.visual_design["authorized_literal_subject"], "")
            compiled = compile_candidate_prompt(spec, 1, concise=True)
            self.assertIn("abstract pattern structure", compiled.clip_g)
            for unrelated in ("bird", "snake", "bee", "dog"):
                self.assertNotIn(unrelated, compiled.clip_g.casefold())

    def test_explicit_literal_subject_is_kept_separate_from_inferred_motif(self):
        document = document_for(CASES[0])
        document["brief"].update(
            background_inspiration="A bird perched on a branch",
            visual_semantics={"motif_descriptors": ["layered feather pattern"]},
            grill_me_context={"semantic_provenance": {"motif_descriptors": "SEMANTIC_INFERENCE"}},
        )
        spec = build_candidate_spec(document, "sdxl-1-0", "B")
        self.assertEqual(spec.visual_design["motif_literalization_policy"], "motif_only")
        self.assertIn("bird", spec.visual_design["authorized_literal_subject"])

    def test_explicit_spatial_treatment_survives_reference_sanitization_and_retries(self):
        document = document_for(CASES[0])
        raw = "Solar Sentinel is moving through the city in first person view"
        document["brief"].update(
            background_inspiration="A city seen through Solar Sentinel's viewpoint",
            theme_reference_treatment=raw,
            grill_me_context={"reference_detected": True, "reference_name": "Solar Sentinel",
                              "semantic_provenance": {"theme_reference_treatment": "USER_EXPLICIT"}},
            visual_semantics={"motif_descriptors": ["radial urban geometry"],
                              "motif_structure_description": "radial urban geometry",
                              "composition_affinities": ["first-person viewpoint", "movement through the city"]},
        )
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(document, engine, "B")
            self.assertEqual(spec.theme_reference_treatment, raw)
            self.assertNotIn("Solar Sentinel", spec.visual_design["theme_reference_treatment"])
            for attempt in (1, 2, 4, 7, 9):
                compiled = compile_candidate_prompt(spec, attempt, "TREATMENT_DRIFT: viewpoint missing" if attempt > 1 else "", concise=attempt > 1)
                conditioning = " ".join(str(value or "") for value in (compiled.positive, compiled.clip_g, compiled.t5))
                self.assertIn("first-person", conditioning.lower())
                self.assertIn("movement", conditioning.lower())
                self.assertNotIn("Solar Sentinel", conditioning)

    def test_treatment_drift_is_a_distinct_retry_category(self):
        self.assertEqual(runtime_retry_category("TREATMENT_DRIFT: missing viewpoint"), "TREATMENT_DRIFT")

    def test_sdxl_concise_palette_roles_survive_all_retry_categories(self):
        spec = build_candidate_spec(document_for(CASES[0]), "sdxl-1-0", "B")
        for attempt, reason in ((1, ""), (2, "PALETTE_ROLE_DRIFT: neutrals dominate"),
                                (4, "HUMAN: figure"), (7, "TEXT: glyphs"),
                                (9, "TREATMENT_DRIFT: viewpoint")):
            compiled = compile_candidate_prompt(spec, attempt, reason, concise=True)
            effective = compiled.clip_g.lower()
            self.assertIn("royal blue", effective)
            self.assertIn("red", effective)
            self.assertIn("visual mass", effective)
            self.assertIn("neutrals support", effective)


    def test_abstract_web_motif_does_not_expand_to_a_creature(self):
        document = document_for(CASES[0])
        document["brief"].update(
            theme="interconnected geometry",
            background_inspiration="Interconnected strands around a geometric skyline",
            theme_reference_treatment="Web-like lattice integrated into the perimeter",
            visual_semantics={
                "motif_descriptors": ["web-like lattice"],
                "motif_structure_description": "Interconnected strands with cross-links",
            },
        )
        document["directions"][0]["direction"]["positive_prompt"] = "Geometric skyline with interlinked strands"
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(document, engine, "B")
            for attempt, reason in ((1, ""), (2, "MOTIF_DRIFT: missing lattice")):
                with self.subTest(engine=engine, attempt=attempt):
                    compiled = compile_candidate_prompt(spec, attempt, reason)
                    conditioning = " ".join(value or "" for value in (
                        compiled.positive, compiled.clip_l, compiled.clip_g, compiled.t5))
                    self.assertIn("strands", conditioning.lower())
                    for creature in ("spider", "insect", "arachnid", "animal", "creature"):
                        self.assertNotIn(creature, conditioning.lower())

    def test_named_reference_uses_semantic_render_fields_and_preserves_raw_provenance(self):
        document = document_for(CASES[0])
        brief = document["brief"]
        brief.update(
            background_inspiration="A Solar Sentinel inspired celebration",
            theme_reference_treatment=(
                "Use Solar Sentinel-inspired traits through radial geometry. "
                "Do not depict Solar Sentinel, masks, people or text."
            ),
            grill_me_context={"reference_detected": True, "reference_name": "Solar-Sentinel"},
            visual_semantics={
                "motif_descriptors": ["interconnected radial geometry", "layered urban forms"],
                "motif_structure_description": "Radial strands connect through curved cross-links across layered urban forms",
                "atmosphere_descriptors": ["energetic and playful"],
                "audience_descriptors": ["children aged 7 to 10"],
                "incompatible_treatments": ["masks", "costumes"],
                "scene_family": "architectural-interior",
            },
        )
        document["directions"][0]["direction"]["positive_prompt"] = (
            "Solar Sentinel-inspired radial geometry in a layered urban scene"
        )
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            with self.subTest(engine=engine):
                spec = build_candidate_spec(document, engine, "B")
                self.assertIn("Solar Sentinel", spec.background_inspiration)
                self.assertIn("Solar Sentinel", spec.theme_reference_treatment)
                self.assertIn("Solar Sentinel", spec.semantic_contract["primary_visual_subject"])
                self.assertNotIn("Solar Sentinel", spec.visual_design["primary_visual_subject"])
                self.assertIn("interconnected radial geometry", spec.visual_design["primary_visual_subject"])
                for attempt, reason in ((1, ""), (2, "MOTIF_DRIFT: missing topology"),
                                        (3, "MOTIF_DRIFT: missing topology")):
                    compiled = compile_candidate_prompt(spec, attempt, reason, concise=attempt > 1)
                    for prompt in (compiled.positive, compiled.clip_l, compiled.clip_g, compiled.t5 or ""):
                        self.assertNotIn("Solar Sentinel", prompt)
                    self.assertIn("radial", compiled.positive.lower())
                    self.assertIn("layered urban forms", compiled.positive)
                    self.assertIn("royal blue", compiled.positive)
                    self.assertIn("red", compiled.positive)
                    self.assertIn("energetic and playful", compiled.positive)
                    self.assertIn("children aged 7 to 10", compiled.positive)
                    self.assertIn("asymmetric layered composition" if attempt == 1 else "layered depth",
                                  compiled.positive)
                    self.assertEqual(compiled.spec_sha256, spec.spec_sha256)
                    self.assertIn("people", compiled.negative)
                    self.assertIn("text", compiled.negative)
                    self.assertIn("characters", compiled.negative)
                    if engine != "flux-2":
                        self.assertIn("masks", compiled.negative)
                        self.assertIn("costumes", compiled.negative)
                    if attempt > 1:
                        self.assertEqual(compiled.failure_category, "MOTIF_DRIFT")
                        self.assertEqual(compiled.positive.count("Make the supplied motif topology visible"), 1)

    def test_named_character_reference_keeps_explicit_visual_traits_without_figure_pressure(self):
        examples = (
            ("Solar Sentinel", "blue and red", "restrained web motifs and energetic comic-style movement"),
            ("Night Guardian", "black and yellow", "angular wing-like motifs and nocturnal comic movement"),
            ("Emerald Titan", "green and purple", "powerful fractured abstract shapes and impact-like movement"),
        )
        for name, colours, traits in examples:
            document = document_for(CASES[0])
            document["brief"].update(
                background_inspiration=(f"A superhero-inspired background using {colours} with {traits}. "
                                        f"Do not depict {name} or any character."),
                theme_reference_treatment=f"Use {name}-inspired colours.",
                grill_me_context={"reference_detected": True, "reference_name": name,
                                  "generation_constraints": ["No people or characters"]},
                visual_semantics={"motif_descriptors": ["abstract geometric motifs"],
                                  "motif_structure_description": "interconnected abstract motifs"},
            )
            for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
                with self.subTest(name=name, engine=engine):
                    spec = build_candidate_spec(document, engine, "B")
                    self.assertIn(traits, spec.visual_design["primary_visual_subject"])
                    self.assertIn(traits, spec.visual_design["theme_reference_treatment"])
                    compiled = compile_candidate_prompt(spec, 1, concise=True)
                    active = " ".join((compiled.positive, compiled.clip_g, compiled.t5 or ""))
                    self.assertIn(traits, active)
                    self.assertNotIn(name, active)
                    self.assertNotIn("superhero-inspired", active)
                    self.assertIn("no human or costumed figures", active)
                    workflow = build_workflow(engine, compiled.positive, compiled.negative,
                                              123, "reference-test", compiled_prompt=compiled)
                    if engine == "flux-2":
                        self.assertFalse(compiled.negative_prompt_channel_active)
                        self.assertNotIn("40", workflow)
                        self.assertIn("no human or costumed figures", workflow["4"]["inputs"]["text"])
                    else:
                        self.assertIn("masked or costumed figures", compiled.negative)
                        negative = workflow["40"]["inputs"]
                        fields = (negative["text"],) if engine == "sdxl-1-0" else (
                            negative["clip_l"], negative["clip_g"], negative["t5xxl"])
                        self.assertTrue(all("masked or costumed figures" in field for field in fields))

    def test_motif_drift_routes_to_motif_correction(self):
        self.assertEqual(runtime_retry_category("MOTIF_DRIFT: missing motif"), "MOTIF_DRIFT")

    def test_named_reference_without_renderable_semantics_fails_closed(self):
        document = document_for(CASES[0])
        document["brief"]["background_inspiration"] = "A Solar Sentinel inspired celebration"
        document["brief"]["grill_me_context"] = {
            "reference_detected": True, "reference_name": "Solar Sentinel",
        }
        with self.assertRaisesRegex(ValueError, "renderable semantic motif"):
            build_candidate_spec(document, "sdxl-1-0", "B")

    def test_six_retry_categories_compile_from_same_spec_without_correction_accumulation(self):
        from app.services.prompt_compiler import _CORRECTIONS
        reasons = {
            "TEXT": "generated readable text", "HUMAN": "human silhouette",
            "PALETTE_OFF": "off-palette colours", "FLAT": "flat background",
            "RASTER": "raster scanlines", "TITLE_PANEL": "TITLE_PANEL bordered card",
        }
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(document_for(CASES[0]), engine, "B")
            initial = compile_candidate_prompt(spec, 1)
            for category, reason in reasons.items():
                with self.subTest(engine=engine, category=category):
                    retry = compile_candidate_prompt(spec, 2, reason)
                    self.assertEqual(retry.failure_category, category)
                    self.assertEqual(retry.spec_sha256, initial.spec_sha256)
                    self.assertIn(spec.background_inspiration, retry.positive)
                    self.assertIn(spec.visual_design["composition_strategy"], retry.positive)
                    self.assertIn(spec.visual_design["palette_relationship"], retry.positive)
                    self.assertIn(_CORRECTIONS[category], retry.positive)
                    for other, wording in _CORRECTIONS.items():
                        if other != category:
                            self.assertNotIn(wording, retry.positive)
                    for field in (retry.positive, retry.clip_l, retry.clip_g, retry.t5 or ""):
                        self.assertNotIn(spec.direction_role, field)
                    self.assertEqual(spec.spec_sha256, initial.spec_sha256)

    def test_model_conditioning_has_no_role_or_panel_language(self):
        labels = ("Architectural Master", "Comic Book Cityscape Illustrator",
                  "Creative Direction for A4 Portrait Event Background", "visual character")
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            document = document_for(CASES[0])
            document["directions"][0]["direction"]["role"] = labels[0]
            document["directions"][0]["direction"]["positive_prompt"] += " " + " ".join(labels)
            spec = build_candidate_spec(document, engine, "B")
            compiled = compile_candidate_prompt(spec, 1)
            conditioning = " ".join(value or "" for value in (compiled.positive, compiled.negative,
                                     compiled.clip_l, compiled.clip_g, compiled.t5))
            for label in labels:
                self.assertNotIn(label.lower(), conditioning.lower())
            self.assertIn("panel", conditioning.lower())

    def test_event_label_is_planning_metadata_not_rendered_text(self):
        document = document_for(CASES[2])
        document["brief"]["title_preference"] = "THE BRIDAL SHOWER"
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(document, engine, "B")
            compiled = compile_candidate_prompt(spec, 2, "Generated background contains OCR text")
            for prompt in (compiled.positive, compiled.clip_l, compiled.clip_g, compiled.t5 or ""):
                if engine != "sd-3-5-medium":
                    self.assertNotIn("bridal shower", prompt.casefold())
                self.assertNotIn("THE BRIDAL SHOWER", prompt)
            if engine == "sd-3-5-medium":
                self.assertIn("Event context: bridal shower.", compiled.clip_g)
            self.assertIn("uninterrupted visual surfaces", compiled.positive)

    def test_sd35_event_and_audience_are_grounded_across_active_encoders(self):
        cases = (
            ("children's birthday party", "children aged 7 to 10", "playful, energetic"),
            ("formal civic ceremony", "adult guests", "dignified, calm"),
            ("unseen lantern symposium", "curious observers", "reflective, warm"),
        )
        for event, audience, mood in cases:
            with self.subTest(event=event):
                document = document_for(CASES[2])
                document["brief"].update(
                    event_context=event, audience_context=audience,
                    visual_semantics={"audience_descriptors": [audience],
                                      "atmosphere_descriptors": mood.split(", ")},
                )
                spec = build_candidate_spec(document, "sd-3-5-medium", "B")
                for concise in (False, True):
                    compiled = compile_candidate_prompt(spec, 1, concise=concise)
                    for channel in (compiled.clip_g, compiled.t5):
                        self.assertIn(event, channel.casefold())
                        self.assertIn(audience, channel.casefold())
                        self.assertIn(mood.split(", ")[0], channel.casefold())
                        for label in ("Event context:", "Audience:", "Event atmosphere:", "Palette:"):
                            self.assertIn(label, channel)
                    self.assertIn("clip_g", compiled.conditioning_coverage["audience"])
                    self.assertIn("t5", compiled.conditioning_coverage["audience"])
                    self.assertIn("clip_g", compiled.conditioning_coverage["event_context"])
                    self.assertIn("t5", compiled.conditioning_coverage["event_context"])
                    self.assertFalse(compiled.record()["prompt_budget"]["clip_g"]["budget_risk"])
                    self.assertFalse(compiled.record()["prompt_budget"]["t5"]["budget_risk"])

    def test_sd35_audience_fallbacks_do_not_invent_or_repeat_context(self):
        for fields, expected in (
            ({}, ""),
            ({"age_group": "ages 7 to 10"}, "ages 7 to 10"),
            ({"audience_context": "invited researchers"}, "invited researchers"),
            ({"audience_context": "adult guests", "visual_semantics":
              {"audience_descriptors": ["adult guests"]}}, "adult guests"),
        ):
            with self.subTest(fields=fields):
                document = document_for(CASES[2])
                document["brief"].update(fields)
                spec = build_candidate_spec(document, "sd-3-5-medium", "B")
                spec.visual_design["visual_semantics"]["audience_descriptors"] = (
                    fields.get("visual_semantics", {}).get("audience_descriptors", []))
                compiled = compile_candidate_prompt(spec, 1, concise=True)
                if expected:
                    self.assertIn(expected, compiled.clip_g.casefold())
                    self.assertIn(expected, compiled.t5.casefold())
                    self.assertEqual(compiled.clip_g.casefold().count(expected), 1)
                else:
                    self.assertNotIn("Audience:", compiled.clip_g)
                    self.assertEqual(compiled.conditioning_coverage["audience"], [])

    def test_incomplete_direction_strategy_is_rejected(self):
        for name in (
            "composition_strategy", "focal_strategy", "depth_strategy",
            "border_or_perimeter_strategy", "title_safe_region_strategy",
            "programme_safe_region_strategy",
        ):
            with self.subTest(name=name):
                document = document_for(CASES[0])
                document["directions"][0]["direction"].pop(name)
                with self.assertRaisesRegex(ValueError, name):
                    build_candidate_spec(document, "sdxl-1-0", "B")

    def test_all_engines_receive_structured_strategies(self):
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            with self.subTest(engine=engine):
                spec = build_candidate_spec(document_for(CASES[0]), engine, "B")
                compiled = compile_candidate_prompt(spec, 1)
                fields = (compiled.positive, compiled.clip_l, compiled.clip_g)
                if compiled.t5:
                    fields += (compiled.t5,)
                for prompt in fields:
                    if engine == "sd-3-5-medium" and prompt == compiled.clip_l:
                        self.assertIn("city skyline", prompt)
                        self.assertLess(len(prompt.split()), 40)
                    else:
                        self.assertIn("asymmetric layered composition", prompt)
                        self.assertIn("off-centre subject focus", prompt)
                        self.assertIn("foreground to background depth", prompt)

    def test_spec_is_structured_and_deeply_immutable(self):
        document = document_for(CASES[0])
        spec = build_candidate_spec(document, "sdxl-1-0", "B")
        digest = spec.spec_sha256
        self.assertEqual(spec.background_inspiration, CASES[0][2])
        self.assertEqual(spec.theme_reference_treatment, CASES[0][3])
        self.assertEqual(spec.direction_role, "expressive")
        self.assertEqual((spec.primary_colour, spec.secondary_colour), ("royal blue", "red"))
        self.assertIn("programme_zone", spec.layout_guidance)
        spec.layout_guidance["title_zone"]["x"] = 0.9
        self.assertEqual(spec.layout_guidance["title_zone"]["x"], 0.1)
        self.assertEqual(spec.spec_sha256, digest)
        document["brief"]["background_inspiration"] = "a different scene"
        self.assertEqual(spec.spec_sha256, digest)

    def test_all_native_attempts_retain_each_brief_and_direction(self):
        for case in CASES:
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                spec = build_candidate_spec(document_for(case), engine, "B")
                for attempt in range(1, 10):
                    reason = REASONS[(attempt - 1) % len(REASONS)] if attempt > 1 else ""
                    compiled = compile_candidate_prompt(spec, attempt, reason)
                    with self.subTest(case=case[1], engine=engine, attempt=attempt):
                        text = compiled.positive.lower()
                        for anchor in (case[7], case[8], case[5], case[6]):
                            self.assertIn(anchor.lower(), text)
                        self.assertIn("intentional negative space", text)
                        self.assertIn("breathing room", text)
                        self.assertIn("spatial balance", text)
                        self.assertNotIn("template boxes", text)
                        self.assertNotIn("stock poster frames", text)
                        self.assertEqual(compiled.spec_sha256, spec.spec_sha256)
                        self.assertNotIn("previous attempt", text)

    def test_recovery_categories_keep_subject_and_one_targeted_correction(self):
        spec = build_candidate_spec(document_for(CASES[0]), "sdxl-1-0", "B")
        expected = {
            "PALETTE_OFF": "requested colour relationship",
            "HUMAN": "Remove people",
            "TEXT": "Render uninterrupted visual surfaces",
            "GRADIENT": "tonal variation and visible depth",
            "WEAK_OUTER_STRUCTURE": "areas permitted by this direction",
            "RASTER": "clean, varied edge structure",
            "RUNTIME": "fewer, larger forms",
            "SECONDARY_MISSING": "secondary colour",
        }
        for reason in REASONS[1:]:
            compiled = compile_candidate_prompt(spec, 9, reason)
            self.assertIn(CASES[0][2], compiled.positive)
            self.assertIn(CASES[0][3], compiled.positive)
            self.assertIn(expected[compiled.failure_category], compiled.positive)
            self.assertEqual(compiled.retry_stage, "rescue")
            self.assertNotIn("Replace fine artefacts", compiled.positive)

    def test_dense_safe_region_is_not_misclassified_as_weak_outer_structure(self):
        self.assertEqual(
            failure_category(DENSE_SAFE_REGION_REASON),
            "DENSE_SAFE_REGION",
        )

        spec = build_candidate_spec(
            document_for(CASES[0]),
            "sd-3-5-medium",
            "B",
        )

        compiled = compile_candidate_prompt(
            spec,
            2,
            DENSE_SAFE_REGION_REASON,
        )

        self.assertEqual(
            compiled.failure_category,
            "DENSE_SAFE_REGION",
        )
        self.assertIn(
            "Relocate detail away from integrated calm regions",
            compiled.positive,
        )
        self.assertNotIn(
            "Increase meaningful requested design structure",
            compiled.positive,
        )

    def test_sd35_conditioning_preserves_subject_direction_and_layout(self):
        spec = build_candidate_spec(document_for(CASES[0]), "sd-3-5-medium", "B")
        compiled = compile_candidate_prompt(spec, 7, REASONS[4])
        self.assertIn("city skyline", compiled.clip_l.lower())
        self.assertNotIn("expressive", compiled.clip_g.lower())
        self.assertIn("city skyline", compiled.clip_g.lower())
        self.assertIn("spiderweb", compiled.clip_g.lower())
        self.assertIn("royal blue", compiled.clip_g.lower())
        self.assertIn("intentional negative space", compiled.t5.lower())
        self.assertIn("breathing room", compiled.t5.lower())
        for anchor in ("city skyline", "spiderweb", "royal blue", "red"):
            self.assertIn(anchor.lower(), compiled.t5.lower())
        self.assertNotIn("spider-man", compiled.t5.lower())
        self.assertLess(len(compiled.negative), 200)

    def test_missing_inspiration_uses_specific_creative_scene_first(self):
        document = document_for(CASES[0], "A")
        document["brief"]["background_inspiration"] = None
        document["brief"]["creative_description"] = (
            "Kid's party for a child into superheroes. "
            "Create a comic-book city skyline background with blue and red "
            "colour blocking, web geometry around the corners and side edges, "
            "and a calm central area for programme text."
        )
        document["directions"][0]["direction"]["positive_prompt"] = (
            "Use a balanced border-led composition. " * 20
            + "Make the visual subject an elegant abstract event background."
        )
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            spec = build_candidate_spec(document, engine, "A")
            compiled = compile_candidate_prompt(spec, 1)
            self.assertEqual(compiled.spec_sha256, spec.spec_sha256)
            if engine == "sdxl-1-0":
                self.assertTrue(compiled.positive.startswith("Background inspiration:"))
            else:
                self.assertTrue(compiled.positive.startswith("Background inspiration:"))
            self.assertIn("Background inspiration: Create a comic-book city skyline", compiled.positive[:450])
            if engine == "sd-3-5-medium":
                self.assertIn("Visual subject: Create a comic-book city skyline", compiled.clip_l)
                self.assertIn("Visual motif:", compiled.clip_g)
                self.assertIn("city skyline", compiled.t5)

    def test_missing_inspiration_fallback_is_theme_and_palette_independent(self):
        examples = (
            ("paper flower garden", "pink", "cream", "Illustrate layered paper flowers along the border"),
            ("black-tie celebration", "navy", "gold", "Depict ballroom pillars and geometric fan motifs"),
            ("coastal wedding", "teal", "sand", "Render dune grasses and shoreline curves"),
        )
        for theme, primary, secondary, scene in examples:
            document = document_for(CASES[0], "A")
            document["brief"].update({
                "theme": theme,
                "primary_colour": primary,
                "secondary_colour": secondary,
                "background_inspiration": None,
                "creative_description": f"An invitation background. {scene} with a calm central text area.",
            })
            document["directions"][0]["direction"]["positive_prompt"] = (
                "An elegant abstract event background with edge decoration."
            )
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                spec = build_candidate_spec(document, engine, "A")
                compiled = compile_candidate_prompt(spec, 1)
                self.assertIn(f"Background inspiration: {scene}", compiled.positive[:450])
                self.assertIn(primary, compiled.positive)
                self.assertIn(secondary, compiled.positive)
                if engine == "sd-3-5-medium":
                    self.assertIn(f"Visual subject: {scene}", compiled.clip_l)

    def test_sdxl_colour_contract_covers_rendered_materials_for_any_palette(self):
        for primary, secondary in (("royal blue", "red"), ("black", "white"), ("pink", "white")):
            document = document_for(CASES[0], "B")
            document["brief"]["primary_colour"] = primary
            document["brief"]["secondary_colour"] = secondary
            compiled = compile_candidate_prompt(
                build_candidate_spec(document, "sdxl-1-0", "B"), 1
            )
            for field in (compiled.clip_g, compiled.clip_l):
                self.assertTrue(field.startswith("Background artwork"))
                self.assertIn(primary, field)
                self.assertIn(secondary, field)
                self.assertIn("requested palette", field.lower())
                self.assertIn("spiderweb", field.lower())
                self.assertLess(field.lower().index("spiderweb"), field.lower().index("requested palette"))
                self.assertNotIn("flat page", field.lower())
                self.assertNotIn("matte forms", field.lower())

    def test_sdxl_retries_keep_source_inspiration_in_both_encoders(self):
        document = document_for(CASES[0], "B")
        spec = build_candidate_spec(document, "sdxl-1-0", "B")
        for attempt in range(1, 10):
            reason = REASONS[(attempt - 1) % len(REASONS)] if attempt > 1 else ""
            compiled = compile_candidate_prompt(spec, attempt, reason)
            for field in (compiled.clip_g, compiled.clip_l):
                self.assertIn(spec.background_inspiration, field)
                self.assertIn(spec.theme_reference_treatment, compiled.positive)
                self.assertIn("spiderweb", field.lower())
                if attempt > 1:
                    self.assertIn("Correction:", field)
            self.assertEqual(compiled.spec_sha256, spec.spec_sha256)

    def test_architectural_frames_remain_background_artwork(self):
        document = document_for(CASES[1], "A")
        document["brief"]["background_inspiration"] = (
            "Black architectural frames and white geometric panels around the edges"
        )
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            spec = build_candidate_spec(document, engine, "A")
            compiled = compile_candidate_prompt(spec, 9, REASONS[6])
            for field in (compiled.clip_l, compiled.clip_g):
                self.assertIn("background", field.lower())
                self.assertIn("architectural frames", field.lower())
                self.assertNotIn("flat page", field.lower())
            self.assertNotIn("four to six", compiled.positive)
            if engine == "sdxl-1-0":
                self.assertIn("background artwork", compiled.clip_l.lower())
                self.assertIn("quiet upper reading area", compiled.clip_g.lower())
                self.assertNotIn("No lettering or logos", compiled.clip_g)
                self.assertIn("poster mockup", compiled.negative)

    def test_named_character_in_backstory_does_not_become_image_subject(self):
        for name in ("Spider-Man", "Captain Comet"):
            document = document_for(CASES[0], "A")
            document["brief"].update({
                "theme": f"{name} city adventure",
                "background_inspiration": None,
                "theme_reference_treatment": None,
                "creative_description": (
                    f"A child loves {name}. "
                    "Create a comic-book city skyline with radial web geometry "
                    "at the border and an open centre for programme text."
                ),
            })
            document["directions"][0]["direction"]["positive_prompt"] = (
                "Radial web geometry around the edges, angular skyline silhouettes, "
                "bright geometric colour blocks."
            )
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                spec = build_candidate_spec(document, engine, "A")
                compiled = compile_candidate_prompt(spec, 1)
                self.assertIn(name, spec.creative_description)
                self.assertNotIn(name.lower(), compiled.positive.lower())
                self.assertIn("city skyline", compiled.positive.lower())
                self.assertIn("radial web geometry", compiled.positive.lower())
                if engine == "sd-3-5-medium":
                    for field in (compiled.clip_l, compiled.clip_g, compiled.t5):
                        self.assertNotIn(name.lower(), field.lower())

    def test_backstory_without_visual_instruction_uses_direction_design(self):
        document = document_for(CASES[0], "A")
        document["brief"].update({
            "theme": "Captain Comet celebration",
            "background_inspiration": None,
            "creative_description": "The guest of honour loves Captain Comet.",
        })
        document["directions"][0]["direction"]["positive_prompt"] = (
            "Angular skyline silhouettes, radial linework at the outer border, "
            "clear central text field."
        )
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            compiled = compile_candidate_prompt(build_candidate_spec(document, engine, "A"), 1)
            self.assertNotIn("Captain Comet", compiled.positive)
            self.assertNotIn("Angular skyline silhouettes", compiled.positive)

    def test_retry_limit_and_failure_classification(self):
        self.assertEqual(failure_category(REASONS[1]), "PALETTE_OFF")
        self.assertEqual(
            failure_category(DENSE_SAFE_REGION_REASON),
            "DENSE_SAFE_REGION",
        )
        with self.assertRaises(ValueError):
            compile_candidate_prompt(build_candidate_spec(document_for(CASES[0]), "sdxl-1-0", "B"), 10, "bad")


if __name__ == "__main__":
    unittest.main()
