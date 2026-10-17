"""Offline regressions for the final live validation and recovery gaps."""

import tempfile
import unittest
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from PIL import Image, ImageDraw

from app.services.background_policy import (
    _palette_role_metrics,
    detect_human_signals,
    validate_palette,
)
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import (
    ComfyUIClient,
    ComfyUIError,
    write_json,
    _palette_only_recovery,
)
from app.services.prompt_compiler import (
    compile_candidate_prompt,
    runtime_retry_category,
)
from app.services.image_execution import (
    _is_palette_failure,
    _retry_candidate_after_error,
)
from app.services.subject_adherence import (
    assess_subject_adherence,
    hard_fidelity_failure,
)
from tests.test_semantic_foundation import document


class LiveValidationRetestTests(unittest.TestCase):
    def _assess(self, design, **scores):
        response = Mock(
            status_code=200,
            headers={},
        )

        response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "subject_score": .9,
                        "atmosphere_score": .85,
                        "motif_score": .7,
                        "audience_fit_score": .9,
                        "observed_evidence": [
                            "visible forms"
                        ],
                        **scores,
                    }
                )
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "candidate.png"
            )

            Image.new(
                "RGB",
                (
                    32,
                    32,
                ),
                "white",
            ).save(
                path
            )

            with patch(
                "app.services.subject_adherence.httpx.post",
                return_value=response,
            ):
                return assess_subject_adherence(
                    path,
                    design,
                    model="minicpm-v4.5:latest",
                )

    def test_default_presence_and_explicit_dominance_are_separate(self):
        default = _palette_role_metrics(
            "navy",
            "white",
            {
                "navy": 200,
                "white": 550,
            },
            1000,
            "",
            "dominant",
            None,
            "default",
        )

        self.assertEqual(
            default[
                "primary_ratio"
            ],
            .2,
        )

        self.assertEqual(
            default[
                "palette_role_warning"
            ],
            "PALETTE_ROLE_WARNING",
        )

        with self.assertRaisesRegex(
            ValueError,
            "primary colour is not meaningfully present",
        ):
            _palette_role_metrics(
                "navy",
                "white",
                {
                    "navy": 1,
                    "white": 800,
                },
                1000,
                "",
                "dominant",
                None,
                "default",
            )

        valid = _palette_role_metrics(
            "green",
            "cream",
            {
                "green": 550,
                "cream": 200,
            },
            1000,
            "green dominant with cream supporting",
            "dominant",
            None,
            "user_explicit",
        )

        self.assertTrue(
            valid[
                "dominance_relationship_passed"
            ]
        )

        with self.assertRaisesRegex(ValueError, "PALETTE_ROLE_DRIFT"):
            _palette_role_metrics(
                "green", "cream", {"green": 450, "cream": 470}, 1000,
                "green dominant with cream supporting", "dominant", None,
                "user_explicit",
            )

    def test_large_unrelated_chromatic_regions_still_fail(self):
        image = Image.new(
            "RGB",
            (
                400,
                200,
            ),
            "blue",
        )

        draw = ImageDraw.Draw(
            image
        )

        draw.rectangle(
            (
                100,
                0,
                199,
                199,
            ),
            fill="red",
        )

        draw.rectangle(
            (
                200,
                0,
                299,
                199,
            ),
            fill="cyan",
        )

        draw.rectangle(
            (
                300,
                0,
                399,
                199,
            ),
            fill="teal",
        )

        with self.assertRaisesRegex(
            ValueError,
            "not requested",
        ):
            validate_palette(
                image,
                "royal blue",
                "red",
                relationship_source="default",
                quality_result={},
            )

    def test_user_explicit_palette_source_enforces_dominance(self):
        with self.assertRaisesRegex(
            ValueError,
            "PALETTE_ROLE_DRIFT",
        ):
            _palette_role_metrics(
                "green",
                "cream",
                {
                    "green": 200,
                    "cream": 550,
                },
                1000,
                "green dominant with cream supporting",
                "dominant",
                None,
                "user_explicit",
            )

    def test_default_compiler_does_not_invent_area_ordering(self):
        data = document()

        data[
            "brief"
        ].update(
            primary_colour="navy",
            secondary_colour="white",
            palette_relationship_source="default",
            palette_relationship="",
        )

        for engine in (
            "sdxl-1-0",
            "sd-3-5-medium",
            "flux-2",
        ):
            spec = build_candidate_spec(
                data,
                engine,
                "A",
            )

            for concise in (
                False,
                True,
            ):
                with self.subTest(
                    engine=engine,
                    concise=concise,
                ):
                    compiled = (
                        compile_candidate_prompt(
                            spec,
                            1,
                            concise=concise,
                        )
                    )

                    text = " ".join(
                        (
                            compiled.positive,
                            compiled.clip_g,
                            compiled.t5
                            or "",
                        )
                    )

                    self.assertNotIn(
                        "largest masses",
                        text,
                    )

                    self.assertNotIn(
                        "largest visual masses",
                        text,
                    )

                    self.assertNotIn(
                        "primary leading",
                        text,
                    )

                    self.assertIn(
                        "navy",
                        text,
                    )

                    self.assertIn(
                        "white",
                        text,
                    )

    def test_flux_has_affirmative_family_restriction_without_negative_channel(self):
        for concise in (
            False,
            True,
        ):
            compiled = (
                compile_candidate_prompt(
                    build_candidate_spec(
                        document(),
                        "flux-2",
                        "A",
                    ),
                    1,
                    concise=concise,
                )
            )

            self.assertIn(
                "chromatic design is restricted to",
                compiled.positive,
            )

            self.assertFalse(
                compiled.negative_prompt_channel_active
            )

            self.assertTrue(
                compiled.negative
            )

    def test_explicit_palette_relationship_reaches_all_engines(self):
        data = document()

        relation = (
            "green dominant with cream supporting"
        )

        data[
            "brief"
        ].update(
            primary_colour="green",
            secondary_colour="cream",
            palette_relationship=relation,
            palette_relationship_source="user_explicit",
        )

        for engine in (
            "flux-2",
            "sdxl-1-0",
            "sd-3-5-medium",
        ):
            for concise in (
                False,
                True,
            ):
                prompt = (
                    compile_candidate_prompt(
                        build_candidate_spec(
                            data,
                            engine,
                            "A",
                        ),
                        1,
                        concise=concise,
                    )
                )

                self.assertIn(
                    relation,
                    prompt.positive,
                )

    def test_audience_is_hard_only_when_explicit_and_atmosphere_requires_hard_flag(
        self,
    ):
        design = {
            "primary_visual_subject": "city",
            "visual_semantics": {
                "audience_descriptors": [
                    "young children"
                ],
                "atmosphere_descriptors": [
                    "playful"
                ],
            },
            "semantic_provenance": {
                "audience_descriptors": (
                    "USER_EXPLICIT"
                )
            },
        }

        result = self._assess(
            design,
            audience_fit_score=.4,
        )

        self.assertEqual(
            result[
                "failure_category"
            ],
            "AUDIENCE_DRIFT",
        )

        self.assertTrue(
            hard_fidelity_failure(
                result
            )
        )

        design[
            "semantic_provenance"
        ][
            "audience_descriptors"
        ] = "SEMANTIC_INFERENCE"

        self.assertFalse(
            hard_fidelity_failure(
                self._assess(
                    design,
                    audience_fit_score=.4,
                )
            )
        )

        self.assertFalse(
            hard_fidelity_failure(
                self._assess(
                    design,
                    atmosphere_score=.2,
                )
            )
        )

        design[
            "hard_dimensions"
        ] = [
            "atmosphere_score"
        ]

        self.assertTrue(
            hard_fidelity_failure(
                self._assess(
                    design,
                    atmosphere_score=.2,
                )
            )
        )

        self.assertTrue(
            hard_fidelity_failure(
                self._assess(
                    design,
                    subject_score=.2,
                )
            )
        )

    def test_vision_human_survives_unrelated_missing_score(self):
        result = self._assess(
            {
                "primary_visual_subject": (
                    "venue"
                )
            },
            subject_score=None,
            human_visible=True,
            human_description=(
                "small upright person beside doorway"
            ),
        )

        self.assertEqual(
            result[
                "decision"
            ],
            "unavailable",
        )

        self.assertTrue(
            result[
                "vision_human_signal"
            ]
        )

        self.assertIn(
            "upright person",
            result[
                "human_description"
            ],
        )

    def test_palette_recovery_uses_fresh_hard_fidelity_and_human_decision(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = (
                Path(directory)
                / "source.png"
            )

            target = (
                Path(directory)
                / "final.png"
            )

            Image.new(
                "RGB",
                (
                    64,
                    64,
                ),
                "blue",
            ).save(
                source
            )

            for assessment, rejection in (
                (
                    {
                        "decision": (
                            "BRIEF_FIDELITY_DRIFT"
                        ),
                        "hard_failed_dimensions": [
                            "audience_fit_score"
                        ],
                        "failure_category": (
                            "AUDIENCE_DRIFT"
                        ),
                    },
                    "AUDIENCE_DRIFT",
                ),
                (
                    {
                        "decision": "pass",
                        "vision_human_signal": True,
                    },
                    "HUMAN",
                ),
                (
                    {
                        "decision": (
                            "BRIEF_FIDELITY_DRIFT"
                        ),
                        "hard_failed_dimensions": [],
                    },
                    None,
                ),
            ):
                with (
                    self.subTest(
                        rejection=rejection
                    ),
                    patch(
                        "app.services.comfyui_client.validate_background",
                        return_value={
                            "validation": {
                                "decision": (
                                    "accepted"
                                )
                            }
                        },
                    ),
                    patch(
                        "app.services.comfyui_client._edge_geometry_similarity",
                        return_value=1,
                    ),
                    patch(
                        "app.services.comfyui_client._preserves_structural_detail",
                        return_value=True,
                    ),
                    patch(
                        "app.services.comfyui_client.assess_subject_adherence",
                        return_value=assessment,
                    ) as check,
                ):
                    if rejection:
                        with self.assertRaisesRegex(
                            ValueError,
                            rejection,
                        ):
                            _palette_only_recovery(
                                source,
                                target,
                                "ABCDEF-123456",
                                "blue",
                                "red",
                                {
                                    "primary_visual_subject": (
                                        "city"
                                    )
                                },
                            )

                    else:
                        _, final = (
                            _palette_only_recovery(
                                source,
                                target,
                                "ABCDEF-123456",
                                "blue",
                                "red",
                                {
                                    "primary_visual_subject": (
                                        "city"
                                    )
                                },
                            )
                        )

                        self.assertEqual(
                            final[
                                "subject_adherence"
                            ],
                            assessment,
                        )

                    self.assertEqual(
                        check.call_args.args[
                            0
                        ],
                        target,
                    )

    def test_small_person_requires_independent_corroboration(self):
        hog = Mock()

        hog.detectMultiScale.return_value = (
            [
                (
                    100,
                    100,
                    20,
                    60,
                )
            ],
            [
                .9
            ],
        )

        for bodies, expected in (
            (
                [
                    (
                        102,
                        102,
                        16,
                        28,
                    )
                ],
                True,
            ),
            (
                [],
                False,
            ),
        ):
            with (
                self.subTest(
                    corroborated=expected
                ),
                patch(
                    "app.services.background_policy.cv2.HOGDescriptor",
                    return_value=hog,
                ),
                patch(
                    "app.services.background_policy._load_cascade",
                    return_value=Mock(),
                ),
                patch(
                    "app.services.background_policy._cascade_detect",
                    side_effect=lambda detector, gray, size, neighbors: (
                        bodies
                        if size
                        == (
                            70,
                            70,
                        )
                        else []
                    ),
                ),
                patch(
                    "app.services.background_policy._confirm_cascade_boxes",
                    return_value=[],
                ),
            ):
                result = (
                    detect_human_signals(
                        Image.new(
                            "RGB",
                            (
                                896,
                                1280,
                            ),
                            "white",
                        )
                    )
                )

            self.assertEqual(
                result[
                    "detected"
                ],
                expected,
            )

    def test_small_face_and_body_corroborate_without_hog(self):
        hog = Mock()

        hog.detectMultiScale.return_value = (
            [],
            [],
        )

        face = (
            102,
            102,
            22,
            22,
        )

        body = (
            95,
            98,
            40,
            60,
        )

        def detections(
            detector,
            gray,
            size,
            neighbors,
        ):
            if size == (
                21,
                21,
            ):
                return [
                    face
                ]

            return (
                [
                    body
                ]
                if size
                == (
                    35,
                    35,
                )
                else []
            )

        with (
            patch(
                "app.services.background_policy.cv2.HOGDescriptor",
                return_value=hog,
            ),
            patch(
                "app.services.background_policy._load_cascade",
                return_value=Mock(),
            ),
            patch(
                "app.services.background_policy._cascade_detect",
                side_effect=detections,
            ),
            patch(
                "app.services.background_policy._confirm_cascade_boxes",
                side_effect=lambda detector, gray, boxes, size: boxes,
            ),
        ):
            result = (
                detect_human_signals(
                    Image.new(
                        "RGB",
                        (
                            896,
                            1280,
                        ),
                        "white",
                    )
                )
            )

        self.assertTrue(
            result[
                "detected"
            ]
        )

        self.assertIn(
            "independent upper-body",
            result[
                "human_detection_reason"
            ],
        )

    def test_palette_retry_classification_matches_recovery_diagnostics(
        self,
    ):
        for reason in (
            "PALETTE_ROLE_DRIFT",
            "PRIMARY_MISSING",
            "SECONDARY_MISSING",
        ):
            self.assertEqual(
                runtime_retry_category(
                    reason
                ),
                reason,
            )

            self.assertTrue(
                _is_palette_failure(
                    reason
                )
            )

            spec = build_candidate_spec(
                document(),
                "sdxl-1-0",
                "A",
            )

            with (
                patch(
                    "app.services.image_execution._rejection_reason",
                    return_value=reason,
                ),
                patch(
                    "app.services.image_execution._candidate_was_rejected",
                    return_value=True,
                ),
                patch(
                    "app.services.image_execution.asyncio.sleep",
                    new_callable=AsyncMock,
                ),
                patch(
                    "app.services.image_execution.logger"
                ) as logger,
            ):
                asyncio.run(
                    _retry_candidate_after_error(
                        Mock(),
                        Mock(
                            engine_id="sdxl-1-0",
                            direction_id="A",
                        ),
                        "ABCDEF-123456",
                        spec,
                        0,
                        ComfyUIError(
                            reason
                        ),
                    )
                )

            events = [
                call.args[
                    0
                ]
                % call.args[
                    1:
                ]
                for call
                in logger.info.call_args_list
            ]

            self.assertTrue(
                any(
                    (
                        "event=prompt_strengthened"
                        in event
                        and "palette_failure=True"
                        in event
                    )
                    for event
                    in events
                )
            )

    def test_failed_audience_retry_has_fresh_final_assessment_and_keeps_history(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            image_path = (
                Path(directory)
                / "image-a.png"
            )

            record_path = (
                image_path
                .with_suffix(
                    ".json"
                )
            )

            client = (
                ComfyUIClient()
            )

            first = {
                "reference_number": (
                    "ABCDEF-123456"
                ),
                "engine_id": (
                    "sdxl-1-0"
                ),
                "direction_id": "A",
                "attempt_count": 1,
            }

            failure = {
                "decision": (
                    "BRIEF_FIDELITY_DRIFT"
                ),
                "failure_category": (
                    "AUDIENCE_DRIFT"
                ),
                "failed_dimensions": [
                    "audience_fit_score"
                ],
                "hard_failed_dimensions": [
                    "audience_fit_score"
                ],
                "reason": (
                    "explicit audience contradicted"
                ),
            }

            passed = {
                "decision": "pass",
                "hard_failed_dimensions": [],
                "failed_dimensions": [],
            }

            with (
                patch(
                    "app.services.comfyui_client.load_prompts_document",
                    return_value=document(),
                ),
                patch(
                    "app.services.comfyui_client.compare_directions_metrics",
                    return_value={
                        "similarity": 0
                    },
                ),
                patch(
                    "app.services.comfyui_client.assess_subject_adherence",
                    side_effect=[
                        failure,
                        passed,
                    ],
                ) as check,
                patch(
                    "app.services.comfyui_client.validate_background",
                    return_value={
                        "validation": {
                            "decision": (
                                "accepted"
                            )
                        }
                    },
                ),
            ):
                candidate = (
                    Path(directory)
                    / "first.png"
                )

                Image.new(
                    "RGB",
                    (
                        64,
                        64,
                    ),
                    "white",
                ).save(
                    candidate
                )

                with self.assertRaises(
                    ComfyUIError
                ):
                    client._validate_downloaded_candidate(
                        first,
                        False,
                        image_path,
                        candidate,
                        {},
                        (
                            "navy",
                            "white",
                        ),
                    )

                rejected, failed = (
                    client._prior_attempts(
                        first,
                        True,
                        False,
                    )
                )

                second = {
                    "reference_number": (
                        "ABCDEF-123456"
                    ),
                    "engine_id": (
                        "sdxl-1-0"
                    ),
                    "direction_id": "A",
                    "attempt_count": 2,
                    "rejected_attempts": (
                        rejected
                    ),
                    "failed_attempts": (
                        failed
                    ),
                    "failure_category": (
                        "AUDIENCE_DRIFT"
                    ),
                }

                candidate = (
                    Path(directory)
                    / "second.png"
                )

                Image.new(
                    "RGB",
                    (
                        64,
                        64,
                    ),
                    "blue",
                ).save(
                    candidate
                )

                (
                    validation,
                    _,
                    _,
                    _,
                ) = (
                    client._validate_downloaded_candidate(
                        second,
                        True,
                        image_path,
                        candidate,
                        {},
                        (
                            "navy",
                            "white",
                        ),
                    )
                )

                second.update(
                    status="complete",
                    validation=validation,
                )

                write_json(
                    record_path,
                    second,
                )

            final = json.loads(
                record_path.read_text()
            )

            self.assertEqual(
                check.call_count,
                2,
            )

            self.assertEqual(
                final[
                    "status"
                ],
                "complete",
            )

            self.assertEqual(
                final[
                    "final_validation"
                ][
                    "decision"
                ],
                "accepted",
            )

            self.assertIsNone(
                final[
                    "failure_category"
                ]
            )

            self.assertEqual(
                final[
                    "brief_fidelity"
                ][
                    "decision"
                ],
                "pass",
            )

            self.assertEqual(
                final[
                    "attempt_history"
                ][
                    0
                ][
                    "primary_failure_category"
                ],
                "AUDIENCE_DRIFT",
            )

    def test_final_hard_audience_failure_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = (
                Path(directory)
                / "image-a.png"
            )

            candidate = (
                Path(directory)
                / "candidate.png"
            )

            Image.new(
                "RGB",
                (
                    64,
                    64,
                ),
                "navy",
            ).save(
                candidate
            )

            record = {
                "reference_number": (
                    "ABCDEF-123456"
                ),
                "engine_id": (
                    "sdxl-1-0"
                ),
                "direction_id": (
                    "A"
                ),
                "attempt_count": 1,
            }

            result = {
                "decision": (
                    "BRIEF_FIDELITY_DRIFT"
                ),
                "failure_category": (
                    "AUDIENCE_DRIFT"
                ),
                "failed_dimensions": [
                    "audience_fit_score"
                ],
                "hard_failed_dimensions": [
                    "audience_fit_score"
                ],
                "reason": (
                    "explicit audience contradicted"
                ),
            }

            with (
                patch(
                    "app.services.comfyui_client.load_prompts_document",
                    return_value=document(),
                ),
                patch(
                    "app.services.comfyui_client.compare_directions_metrics",
                    return_value={
                        "similarity": 0
                    },
                ),
                patch(
                    "app.services.comfyui_client.assess_subject_adherence",
                    return_value=result,
                ),
                patch(
                    "app.services.comfyui_client.validate_background",
                    return_value={
                        "validation": {
                            "decision": (
                                "accepted"
                            )
                        }
                    },
                ),
                patch(
                    "app.services.comfyui_client.write_json"
                ),
            ):
                with self.assertRaisesRegex(
                    ComfyUIError,
                    "AUDIENCE_DRIFT",
                ):
                    (
                        ComfyUIClient()
                        ._validate_downloaded_candidate(
                            record,
                            False,
                            image_path,
                            candidate,
                            {},
                            (
                                "navy",
                                "white",
                            ),
                        )
                    )

    def test_success_record_distinguishes_retry_reason_from_final_decision(
        self,
    ):
        history = [
            {
                "attempt_count": 1,
                "rejection_reason": (
                    "DENSITY_DRIFT"
                ),
                "primary_failure_category": (
                    "DENSITY_DRIFT"
                ),
            }
        ]

        record = {
            "status": "complete",
            "attempt_count": 2,
            "failure_category": (
                "DENSITY_DRIFT"
            ),
            "primary_failure_category": (
                "DENSITY_DRIFT"
            ),
            "rejected_attempts": (
                history
            ),
            "validation": {
                "validation": {
                    "decision": (
                        "accepted"
                    ),
                    "failure_category": (
                        None
                    ),
                }
            },
        }

        with patch(
            "app.services.comfyui_client.atomic_write_json"
        ) as persist:
            write_json(
                Path(
                    "unused.json"
                ),
                record,
            )

        saved = (
            persist
            .call_args
            .args[
                1
            ]
        )

        self.assertIsNone(
            saved[
                "failure_category"
            ]
        )

        self.assertIsNone(
            saved[
                "primary_failure_category"
            ]
        )

        self.assertEqual(
            saved[
                "final_validation"
            ][
                "decision"
            ],
            "accepted",
        )

        self.assertEqual(
            saved[
                "attempt_history"
            ][
                0
            ][
                "primary_failure_category"
            ],
            "DENSITY_DRIFT",
        )

    def test_persistence_cannot_label_hard_failed_semantics_complete(
        self,
    ):
        record = {
            "status": "complete",
            "subject_adherence": {
                "decision": (
                    "BRIEF_FIDELITY_DRIFT"
                ),
                "hard_failed_dimensions": [
                    "audience_fit_score"
                ],
            },
        }

        with patch(
            "app.services.comfyui_client.atomic_write_json"
        ) as persist:
            with self.assertRaisesRegex(
                ComfyUIError,
                "hard brief-fidelity failure",
            ):
                write_json(
                    Path(
                        "unused.json"
                    ),
                    record,
                )

        persist.assert_not_called()

    def test_cached_candidate_cannot_bypass_hard_audience_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            image_path = (
                Path(directory)
                / "image-a.png"
            )

            record_path = (
                image_path
                .with_suffix(
                    ".json"
                )
            )

            Image.new(
                "RGB",
                (
                    64,
                    64,
                ),
                "white",
            ).save(
                image_path
            )

            record_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "sha256": "test",
                        "attempt_count": 1,
                        "validation": {
                            "decision": (
                                "accepted"
                            )
                        },
                    }
                )
            )

            failure = {
                "decision": (
                    "BRIEF_FIDELITY_DRIFT"
                ),
                "hard_failed_dimensions": [
                    "audience_fit_score"
                ],
                "failure_category": (
                    "AUDIENCE_DRIFT"
                ),
                "reason": (
                    "explicit audience contradicted"
                ),
            }

            with (
                patch(
                    "app.services.comfyui_client.load_prompts_document",
                    return_value=document(),
                ),
                patch(
                    "app.services.comfyui_client.assess_subject_adherence",
                    return_value=failure,
                ),
                patch(
                    "app.services.comfyui_client.validate_background",
                    return_value={
                        "validation": {
                            "decision": (
                                "accepted"
                            )
                        }
                    },
                ),
            ):
                previous, reused = (
                    ComfyUIClient()
                    ._load_existing_candidate(
                        record_path,
                        image_path,
                        None,
                        "ABCDEF-123456",
                        "sdxl-1-0",
                        "A",
                    )
                )

            self.assertIsNone(
                reused
            )

            self.assertEqual(
                previous[
                    "status"
                ],
                "rejected",
            )

            saved = json.loads(
                record_path.read_text()
            )

            self.assertEqual(
                saved[
                    "final_validation"
                ][
                    "failure_category"
                ],
                "AUDIENCE_DRIFT",
            )

            self.assertEqual(
                saved[
                    "brief_fidelity"
                ][
                    "hard_failed_dimensions"
                ],
                [
                    "audience_fit_score"
                ],
            )

    def test_recovered_candidate_records_normalized_semantics_as_final(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            image_path = (
                Path(directory)
                / "image-a.png"
            )

            candidate = (
                Path(directory)
                / "candidate.png"
            )

            Image.new(
                "RGB",
                (
                    64,
                    64,
                ),
                "white",
            ).save(
                candidate
            )

            record = {
                "reference_number": (
                    "ABCDEF-123456"
                ),
                "engine_id": (
                    "sdxl-1-0"
                ),
                "direction_id": (
                    "A"
                ),
                "attempt_count": 1,
            }

            raw_semantics = {
                "decision": (
                    "pass"
                ),
                "failed_dimensions": [],
                "hard_failed_dimensions": [],
                "motif_score": .75,
            }

            final_semantics = {
                "decision": "pass",
                "failed_dimensions": [],
                "hard_failed_dimensions": [],
                "motif_score": .9,
            }

            error = ValueError(
                "Generated background contains colours that were not requested "
                "(off-palette ratio 0.10)"
            )
            error.diagnostics = {
                "primary_ratio": .45, "secondary_ratio": .20,
                "requested_chromatic_ratio": .65,
                "structural_neutral_ratio": .10,
                "largest_unrequested_region_ratio": .01,
                "validation": {"diagnostic_findings": []},
            }

            def recovered(
                source,
                target,
                *args,
            ):
                Image.new(
                    "RGB",
                    (
                        64,
                        64,
                    ),
                    "blue",
                ).save(
                    target
                )

                return (
                    {},
                    {
                        "validation": {
                            "decision": (
                                "accepted"
                            )
                        },
                        "subject_adherence": (
                            final_semantics
                        ),
                    },
                )

            with (
                patch(
                    "app.services.comfyui_client.load_prompts_document",
                    return_value=document(),
                ),
                patch(
                    "app.services.comfyui_client.compare_directions_metrics",
                    return_value={
                        "similarity": 0
                    },
                ),
                patch(
                    "app.services.comfyui_client.assess_subject_adherence",
                    return_value=raw_semantics,
                ),
                patch(
                    "app.services.comfyui_client.validate_background",
                    side_effect=error,
                ),
                patch(
                    "app.services.comfyui_client._palette_only_recovery",
                    side_effect=recovered,
                ),
                patch(
                    "app.services.comfyui_client._edge_geometry_similarity",
                    return_value=1,
                ),
                patch(
                    "app.services.comfyui_client._preserves_structural_detail",
                    return_value=True,
                ),
            ):
                (
                    validation,
                    _,
                    normalized,
                    _,
                ) = (
                    ComfyUIClient()
                    ._validate_downloaded_candidate(
                        record,
                        False,
                        image_path,
                        candidate,
                        {},
                        (
                            "navy",
                            "gold",
                        ),
                    )
                )

                record.update(
                    status="complete",
                    validation=validation,
                )

                write_json(
                    image_path.with_suffix(
                        ".json"
                    ),
                    record,
                )

            self.assertTrue(
                normalized
            )

            self.assertEqual(
                record[
                    "subject_adherence"
                ],
                final_semantics,
            )

            self.assertEqual(
                record[
                    "raw_subject_adherence"
                ],
                raw_semantics,
            )

            self.assertEqual(
                record[
                    "final_validation"
                ][
                    "brief_fidelity"
                ][
                    "decision"
                ],
                "pass",
            )

    def test_hard_explicit_motif_failure_remains_motif_drift(
        self,
    ):
        result = {
            "decision": (
                "BRIEF_FIDELITY_DRIFT"
            ),
            "failure_category": (
                "MOTIF_DRIFT"
            ),
            "subject_score": 0.90,
            "motif_score": 0.30,
            "failed_dimensions": [
                "motif_score",
            ],
            "hard_dimensions": [
                "subject_score",
                "motif_score",
            ],
            "hard_failed_dimensions": [
                "motif_score",
            ],
            "compound_fidelity_failure": (
                True
            ),
        }

        self.assertTrue(
            hard_fidelity_failure(
                result
            )
        )

        self.assertEqual(
            result[
                "failure_category"
            ],
            "MOTIF_DRIFT",
        )

        self.assertNotEqual(
            result[
                "failure_category"
            ],
            "SUBJECT_DRIFT",
        )

    def test_final_record_keeps_latest_prior_retry_category(
        self,
    ):
        record = {
            "status": (
                "complete"
            ),
            "attempt_count": 4,
            "failure_category": (
                "MOTIF_DRIFT"
            ),
            "primary_failure_category": (
                "MOTIF_DRIFT"
            ),
            "rejected_attempts": [
                {
                    "attempt_count": 1,
                    "failure_category": (
                        "PALETTE"
                    ),
                    "primary_failure_category": (
                        "PALETTE"
                    ),
                },
                {
                    "attempt_count": 2,
                    "failure_category": (
                        "HUMAN"
                    ),
                    "primary_failure_category": (
                        "HUMAN"
                    ),
                },
                {
                    "attempt_count": 3,
                    "failure_category": (
                        "MOTIF_DRIFT"
                    ),
                    "primary_failure_category": (
                        "MOTIF_DRIFT"
                    ),
                },
            ],
            "validation": {
                "validation": {
                    "decision": (
                        "accepted"
                    ),
                    "failure_category": (
                        None
                    ),
                }
            },
        }

        with patch(
            "app.services.comfyui_client.atomic_write_json"
        ) as persist:
            write_json(
                Path(
                    "unused.json"
                ),
                record,
            )

        saved = (
            persist
            .call_args
            .args[
                1
            ]
        )

        self.assertIsNone(
            saved[
                "failure_category"
            ]
        )

        self.assertIsNone(
            saved[
                "primary_failure_category"
            ]
        )

        self.assertEqual(
            saved[
                "retry_failure_category"
            ],
            "MOTIF_DRIFT",
        )

        self.assertEqual(
            saved[
                "final_validation"
            ][
                "decision"
            ],
            "accepted",
        )

        self.assertEqual(
            [
                item.get(
                    "primary_failure_category"
                )
                for item
                in saved[
                    "attempt_history"
                ][
                    :3
                ]
            ],
            [
                "PALETTE",
                "HUMAN",
                "MOTIF_DRIFT",
            ],
        )


    def test_retry_stage_propagates_through_concise_compilation(self):
        spec = build_candidate_spec(document(), "sdxl-1-0", "A")
        expected = {
            1: "normal",
            2: "normal",
            3: "normal",
            4: "targeted_recovery",
            5: "targeted_recovery",
            6: "targeted_recovery",
            7: "strict_recovery",
            8: "strict_recovery",
            9: "rescue",
        }

        for concise in (False, True):
            for attempt, stage in expected.items():
                with self.subTest(concise=concise, attempt=attempt):
                    reason = "" if attempt == 1 else "PALETTE_OFF"
                    compiled = compile_candidate_prompt(
                        spec,
                        attempt,
                        reason,
                        concise=concise,
                    )
                    self.assertEqual(compiled.retry_stage, stage)



if __name__ == "__main__":
    unittest.main()
