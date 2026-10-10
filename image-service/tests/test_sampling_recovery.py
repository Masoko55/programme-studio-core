import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from app.config.settings import settings

from app.services.comfyui_client import (
    MAX_SD35_RECOVERABLE_OFF_PALETTE_RATIO,
    MAX_SDXL_RECOVERABLE_OFF_PALETTE_RATIO,
    _palette_only_recovery,
    _preserves_structural_detail,
    _recoverable_palette_error,
    build_workflow,
)

from app.services.prompt_compiler import (
    retry_stage,
)

from app.services.sampling_profiles import (
    SD35_PRODUCTION_BASELINE,
    SD35_RESCUE,
    select_sampling_profile,
    summarize_failures,
)


def rejected(
    category: str,
    attempt: int,
) -> dict:
    reasons = {
        "PALETTE_OFF": (
            "off-palette ratio 0.77"
        ),
        "RASTER": (
            "fine repetitive vertical "
            "raster-line structure"
        ),
        "HUMAN": (
            "human figure or face"
        ),
    }

    return {
        "attempt_count": attempt,
        "rejection_reason": (
            reasons[
                category
            ]
        ),
        "failure_category": "INITIAL",
    }


def failure_record(
    category: str,
    count: int,
) -> dict | None:
    if (
        count <= 0
    ):
        return None

    return {
        "status": "rejected",
        "attempt_count": count,
        "rejected_attempts": [
            rejected(
                category,
                attempt,
            )
            for attempt in range(
                1,
                count,
            )
        ],
        **rejected(
            category,
            count,
        ),
    }


class SamplingRecoveryTests(
    unittest.TestCase
):
    def test_palette_recovery_preserves_structural_detail(self):
        raw = {"structural_edge_density": 0.22}
        self.assertFalse(_preserves_structural_detail(raw, {"structural_edge_density": 0.15}))
        self.assertTrue(_preserves_structural_detail(raw, {"structural_edge_density": 0.18}))
        self.assertTrue(_preserves_structural_detail(
            {"structural_edge_density": 0.04},
            {"structural_edge_density": 0.03},
        ))

    # ========================================================
    # Palette recovery limits
    # ========================================================

    def test_engine_specific_palette_recovery_limits(
        self,
    ):
        self.assertEqual(
            MAX_SDXL_RECOVERABLE_OFF_PALETTE_RATIO,
            0.35,
        )

        self.assertEqual(
            MAX_SD35_RECOVERABLE_OFF_PALETTE_RATIO,
            0.15,
        )

    def test_sdxl_palette_recovery_range_is_preserved(
        self,
    ):
        self.assertTrue(
            _recoverable_palette_error(
                "off-palette ratio 0.20",
                settings.engine_2_id,
            )
        )

        self.assertTrue(
            _recoverable_palette_error(
                "off-palette ratio 0.35",
                settings.engine_2_id,
            )
        )

        self.assertFalse(
            _recoverable_palette_error(
                "off-palette ratio 0.36",
                settings.engine_2_id,
            )
        )

    def test_sd35_palette_recovery_is_intentionally_narrow(
        self,
    ):
        self.assertTrue(
            _recoverable_palette_error(
                "off-palette ratio 0.09",
                settings.engine_3_id,
            )
        )

        self.assertTrue(
            _recoverable_palette_error(
                "off-palette ratio 0.15",
                settings.engine_3_id,
            )
        )

        self.assertFalse(
            _recoverable_palette_error(
                "off-palette ratio 0.16",
                settings.engine_3_id,
            )
        )

        self.assertFalse(
            _recoverable_palette_error(
                "off-palette ratio 0.68",
                settings.engine_3_id,
            )
        )

    def test_non_palette_error_is_never_recoloured(
        self,
    ):
        self.assertFalse(
            _recoverable_palette_error(
                "fine repetitive vertical raster-line structure",
                settings.engine_3_id,
            )
        )

        self.assertFalse(
            _recoverable_palette_error(
                "human figure or face",
                settings.engine_3_id,
            )
        )

    def test_flux_is_not_palette_normalization_eligible(
        self,
    ):
        self.assertFalse(
            _recoverable_palette_error(
                "off-palette ratio 0.01",
                settings.engine_1_id,
            )
        )

    # ========================================================
    # Failure history
    # ========================================================

    def test_history_uses_outcome_not_prompt_correction(
        self,
    ):
        record = {
            "status": "rejected",
            "attempt_count": 4,
            "rejected_attempts": [
                rejected(
                    "RASTER",
                    1,
                ),
                rejected(
                    "RASTER",
                    2,
                ),
                rejected(
                    "HUMAN",
                    3,
                ),
            ],
            **rejected(
                "RASTER",
                4,
            ),
        }

        summary = (
            summarize_failures(
                record
            )
        )

        self.assertEqual(
            summary.counts[
                "RASTER"
            ],
            3,
        )

        self.assertEqual(
            summary.consecutive_category,
            "RASTER",
        )

        self.assertEqual(
            summary.consecutive_count,
            1,
        )

    # ========================================================
    # Retry stages
    # ========================================================

    def test_retry_stage_contract_is_preserved(
        self,
    ):
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

        for (
            attempt,
            stage,
        ) in expected.items():
            with self.subTest(
                attempt=attempt
            ):
                self.assertEqual(
                    retry_stage(
                        attempt
                    ),
                    stage,
                )

    # ========================================================
    # SD3.5 profile selection
    # ========================================================

    def test_sd35_normal_stage_uses_empirical_baseline(
        self,
    ):
        for attempt in (
            1,
            2,
            3,
        ):
            with self.subTest(
                attempt=attempt
            ):
                profile = (
                    select_sampling_profile(
                        settings.engine_3_id,
                        attempt,
                        summarize_failures(
                            None
                        ),
                    )
                )

                self.assertEqual(
                    profile,
                    SD35_PRODUCTION_BASELINE,
                )

                self.assertEqual(
                    profile.name,
                    "alternate_sampler",
                )

    def test_sd35_palette_history_keeps_empirical_baseline(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                4,
                summarize_failures(
                    failure_record(
                        "PALETTE_OFF",
                        3,
                    )
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_PRODUCTION_BASELINE,
        )

    def test_sd35_repeated_raster_switches_to_rescue(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                3,
                summarize_failures(
                    failure_record(
                        "RASTER",
                        2,
                    )
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_RESCUE,
        )

    def test_sd35_strict_stage_uses_rescue(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                7,
                summarize_failures(
                    failure_record(
                        "RASTER",
                        6,
                    )
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_RESCUE,
        )

    def test_sd35_attempt_nine_is_rescue_even_without_history(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                9,
                summarize_failures(
                    None
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_RESCUE,
        )

    # ========================================================
    # Engine isolation
    # ========================================================

    def test_flux_keeps_existing_sampling_path(
        self,
    ):
        self.assertIsNone(
            select_sampling_profile(
                settings.engine_1_id,
                4,
                summarize_failures(
                    None
                ),
            )
        )

    # ========================================================
    # Dynamic workflow values
    # ========================================================

    def test_workflows_receive_dynamic_sampling_values_and_preserve_encoders(
        self,
    ):
        for engine in (
            settings.engine_2_id,
            settings.engine_3_id,
        ):
            with self.subTest(
                engine=engine
            ):
                profile = (
                    select_sampling_profile(
                        engine,
                        9,
                        summarize_failures(
                            None
                        ),
                    )
                )

                workflow = (
                    build_workflow(
                        engine,
                        "subject",
                        "",
                        123,
                        "test",
                        sampling_profile=(
                            profile
                        ),
                    )
                )

                sampler = (
                    workflow[
                        "3"
                    ][
                        "inputs"
                    ]
                )

                self.assertEqual(
                    sampler[
                        "steps"
                    ],
                    profile.steps,
                )

                self.assertEqual(
                    sampler[
                        "cfg"
                    ],
                    profile.cfg,
                )

                self.assertEqual(
                    sampler[
                        "sampler_name"
                    ],
                    profile.sampler_name,
                )

                self.assertEqual(
                    sampler[
                        "scheduler"
                    ],
                    profile.scheduler,
                )

                if (
                    engine
                    == settings.engine_3_id
                ):
                    self.assertEqual(
                        workflow[
                            "55"
                        ][
                            "inputs"
                        ][
                            "shift"
                        ],
                        profile.shift,
                    )

                    self.assertEqual(
                        workflow[
                            "16"
                        ][
                            "class_type"
                        ],
                        "CLIPTextEncodeSD3",
                    )

                    self.assertEqual(
                        workflow[
                            "54"
                        ][
                            "class_type"
                        ],
                        "TripleCLIPLoader",
                    )

    def test_sd35_production_baseline_uses_checkpoint_sampling_without_shift_node(
        self,
    ):
        profile = (
            SD35_PRODUCTION_BASELINE
        )

        workflow = (
            build_workflow(
                settings.engine_3_id,
                "subject",
                "",
                123,
                "test",
                sampling_profile=(
                    profile
                ),
            )
        )

        self.assertNotIn(
            "55",
            workflow,
        )

        self.assertEqual(
            workflow[
                "3"
            ][
                "inputs"
            ][
                "model"
            ],
            [
                "4",
                0,
            ],
        )

        self.assertEqual(
            workflow[
                "3"
            ][
                "inputs"
            ][
                "sampler_name"
            ],
            "dpmpp_2m",
        )

        self.assertEqual(
            workflow[
                "3"
            ][
                "inputs"
            ][
                "scheduler"
            ],
            "sgm_uniform",
        )

        self.assertEqual(
            workflow[
                "3"
            ][
                "inputs"
            ][
                "steps"
            ],
            28,
        )

    # ========================================================
    # Deterministic palette-only recovery
    # ========================================================

    def test_recolour_requires_raw_structure_and_revalidates_result(
        self,
    ):
        with TemporaryDirectory() as directory:
            source = (
                Path(
                    directory
                )
                / "raw.png"
            )

            target = (
                Path(
                    directory
                )
                / "normalized.png"
            )

            Image.new(
                "RGB",
                (
                    32,
                    32,
                ),
                (
                    100,
                    120,
                    130,
                ),
            ).save(
                source
            )

            calls = []

            def validate(
                path,
                **kwargs,
            ):
                calls.append(
                    (
                        path,
                        kwargs.get(
                            "enforce_palette",
                            True,
                        ),
                    )
                )

                return {
                    "structural_edge_density": 0.2,
                    "sha256": "test",
                }

            with (
                patch(
                    "app.services.comfyui_client."
                    "validate_background",
                    side_effect=validate,
                ),
                patch(
                    "app.services.comfyui_client."
                    "_edge_geometry_similarity",
                    return_value=0.9,
                ),
            ):
                _palette_only_recovery(
                    source,
                    target,
                    "ABCDEF-123456",
                    "royal blue",
                    "red",
                )

            self.assertEqual(
                calls,
                [
                    (
                        source,
                        False,
                    ),
                    (
                        target,
                        True,
                    ),
                ],
            )

            self.assertTrue(
                target.exists()
            )

            target.unlink()

            #
            # If the raw candidate fails structural validation, deterministic
            # colour remapping must never run.
            #
            with (
                patch(
                    "app.services.comfyui_client."
                    "validate_background",
                    side_effect=ValueError(
                        "raster structure"
                    ),
                ),
                patch(
                    "app.services.comfyui_client."
                    "constrain_to_requested_palette"
                ) as colour,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "raster",
                ):
                    _palette_only_recovery(
                        source,
                        target,
                        "ABCDEF-123456",
                        "royal blue",
                        "red",
                    )

                colour.assert_not_called()

            self.assertFalse(
                target.exists()
            )


if __name__ == "__main__":
    unittest.main()
