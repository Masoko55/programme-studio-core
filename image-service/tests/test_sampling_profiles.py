import unittest

from app.config.settings import settings

from app.services.prompt_compiler import (
    retry_stage,
)

from app.services.sampling_profiles import (
    FailureHistorySummary,
    SD35_PRODUCTION_BASELINE,
    SD35_RESCUE,
    SDXL_BASELINE,
    SDXL_PALETTE_ALTERNATE,
    SDXL_PALETTE_GUIDANCE,
    SDXL_STRICT,
    select_sampling_profile,
)


class SamplingProfileTests(
    unittest.TestCase
):
    def test_retry_stage_contract(
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

    def test_invalid_attempt_rejected(
        self,
    ):
        history = FailureHistorySummary(
            {},
            "",
            0,
        )

        with self.assertRaises(
            ValueError
        ):
            select_sampling_profile(
                settings.engine_2_id,
                10,
                history,
            )

    # ========================================================
    # SDXL
    # ========================================================

    def test_sdxl_initial_attempt_uses_baseline(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_2_id,
                1,
                FailureHistorySummary(
                    {},
                    "",
                    0,
                ),
            )
        )

        self.assertEqual(
            profile,
            SDXL_BASELINE,
        )

    def test_sdxl_one_palette_failure_stays_baseline(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_2_id,
                2,
                FailureHistorySummary(
                    {
                        "PALETTE_OFF": 1,
                    },
                    "PALETTE_OFF",
                    1,
                ),
            )
        )

        self.assertEqual(
            profile,
            SDXL_BASELINE,
        )

    def test_sdxl_repeated_palette_failure_changes_profile(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_2_id,
                3,
                FailureHistorySummary(
                    {
                        "PALETTE_OFF": 2,
                    },
                    "PALETTE_OFF",
                    2,
                ),
            )
        )

        self.assertEqual(
            profile,
            SDXL_PALETTE_GUIDANCE,
        )

    def test_sdxl_targeted_recovery_uses_alternate_profile(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_2_id,
                4,
                FailureHistorySummary(
                    {
                        "PALETTE_OFF": 3,
                    },
                    "PALETTE_OFF",
                    3,
                ),
            )
        )

        self.assertEqual(
            profile,
            SDXL_PALETTE_ALTERNATE,
        )

    def test_sdxl_strict_stage_uses_strict_profile(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_2_id,
                7,
                FailureHistorySummary(
                    {
                        "PALETTE_OFF": 6,
                    },
                    "PALETTE_OFF",
                    6,
                ),
            )
        )

        self.assertEqual(
            profile,
            SDXL_STRICT,
        )

    def test_sdxl_rescue_uses_strict_profile(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_2_id,
                9,
                FailureHistorySummary(
                    {
                        "PALETTE_OFF": 8,
                    },
                    "PALETTE_OFF",
                    8,
                ),
            )
        )

        self.assertEqual(
            profile,
            SDXL_STRICT,
        )

    # ========================================================
    # SD3.5
    # ========================================================

    def test_sd35_initial_attempt_uses_empirical_production_baseline(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                1,
                FailureHistorySummary(
                    {},
                    "",
                    0,
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

        self.assertEqual(
            profile.sampler_name,
            "dpmpp_2m",
        )

        self.assertEqual(
            profile.scheduler,
            "sgm_uniform",
        )

        self.assertEqual(
            profile.steps,
            28,
        )

        self.assertEqual(
            profile.cfg,
            4.0,
        )

        self.assertIsNone(
            profile.shift
        )

    def test_sd35_second_attempt_keeps_production_baseline(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                2,
                FailureHistorySummary(
                    {
                        "PALETTE_OFF": 1,
                    },
                    "PALETTE_OFF",
                    1,
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_PRODUCTION_BASELINE,
        )

    def test_sd35_palette_failures_do_not_move_away_from_best_profile(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                4,
                FailureHistorySummary(
                    {
                        "PALETTE_OFF": 3,
                    },
                    "PALETTE_OFF",
                    3,
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_PRODUCTION_BASELINE,
        )

    def test_sd35_single_raster_failure_does_not_immediately_abandon_baseline(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                2,
                FailureHistorySummary(
                    {
                        "RASTER": 1,
                    },
                    "RASTER",
                    1,
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
                FailureHistorySummary(
                    {
                        "RASTER": 2,
                    },
                    "RASTER",
                    2,
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_RESCUE,
        )

    def test_sd35_targeted_raster_uses_rescue(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                4,
                FailureHistorySummary(
                    {
                        "RASTER": 3,
                    },
                    "RASTER",
                    3,
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
                FailureHistorySummary(
                    {
                        "RASTER": 6,
                    },
                    "RASTER",
                    6,
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_RESCUE,
        )

    def test_sd35_attempt_nine_is_rescue(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_3_id,
                9,
                FailureHistorySummary(
                    {},
                    "",
                    0,
                ),
            )
        )

        self.assertEqual(
            profile,
            SD35_RESCUE,
        )

    # ========================================================
    # FLUX
    # ========================================================

    def test_flux_keeps_existing_sampling_path(
        self,
    ):
        profile = (
            select_sampling_profile(
                settings.engine_1_id,
                1,
                FailureHistorySummary(
                    {},
                    "",
                    0,
                ),
            )
        )

        self.assertIsNone(
            profile
        )


if __name__ == "__main__":
    unittest.main()