import unittest

from app.config.settings import settings

from app.services.prompt_compiler import (
    retry_stage,
)

from app.services.sampling_profiles import (
    FailureHistorySummary,
    SD35_ALTERNATE_SAMPLER,
    SD35_OFFICIAL_BASELINE,
    SD35_RESCUE,
    SD35_REVISED_SCHEDULE,
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
        self.assertEqual(
            retry_stage(1),
            "normal",
        )

        self.assertEqual(
            retry_stage(2),
            "normal",
        )

        self.assertEqual(
            retry_stage(3),
            "normal",
        )

        self.assertEqual(
            retry_stage(4),
            "targeted_recovery",
        )

        self.assertEqual(
            retry_stage(5),
            "targeted_recovery",
        )

        self.assertEqual(
            retry_stage(6),
            "targeted_recovery",
        )

        self.assertEqual(
            retry_stage(7),
            "strict_recovery",
        )

        self.assertEqual(
            retry_stage(8),
            "strict_recovery",
        )

        self.assertEqual(
            retry_stage(9),
            "rescue",
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

    def test_sdxl_initial_attempt_uses_baseline(
        self,
    ):
        history = FailureHistorySummary(
            {},
            "",
            0,
        )

        profile = select_sampling_profile(
            settings.engine_2_id,
            1,
            history,
        )

        self.assertEqual(
            profile,
            SDXL_BASELINE,
        )

    def test_sdxl_repeated_palette_failure_changes_profile(
        self,
    ):
        history = FailureHistorySummary(
            {
                "PALETTE_OFF": 2,
            },
            "PALETTE_OFF",
            2,
        )

        profile = select_sampling_profile(
            settings.engine_2_id,
            3,
            history,
        )

        self.assertEqual(
            profile,
            SDXL_PALETTE_GUIDANCE,
        )

    def test_sdxl_targeted_recovery_uses_alternate_profile(
        self,
    ):
        history = FailureHistorySummary(
            {
                "PALETTE_OFF": 3,
            },
            "PALETTE_OFF",
            3,
        )

        profile = select_sampling_profile(
            settings.engine_2_id,
            4,
            history,
        )

        self.assertEqual(
            profile,
            SDXL_PALETTE_ALTERNATE,
        )

    def test_sdxl_strict_stage_uses_strict_profile(
        self,
    ):
        history = FailureHistorySummary(
            {
                "PALETTE_OFF": 6,
            },
            "PALETTE_OFF",
            6,
        )

        profile = select_sampling_profile(
            settings.engine_2_id,
            7,
            history,
        )

        self.assertEqual(
            profile,
            SDXL_STRICT,
        )

    def test_sdxl_rescue_uses_strict_profile(
        self,
    ):
        history = FailureHistorySummary(
            {
                "PALETTE_OFF": 8,
            },
            "PALETTE_OFF",
            8,
        )

        profile = select_sampling_profile(
            settings.engine_2_id,
            9,
            history,
        )

        self.assertEqual(
            profile,
            SDXL_STRICT,
        )

    def test_sd35_initial_attempt_uses_official_baseline(
        self,
    ):
        history = FailureHistorySummary(
            {},
            "",
            0,
        )

        profile = select_sampling_profile(
            settings.engine_3_id,
            1,
            history,
        )

        self.assertEqual(
            profile,
            SD35_OFFICIAL_BASELINE,
        )

    def test_sd35_repeated_raster_switches_sampler(
        self,
    ):
        history = FailureHistorySummary(
            {
                "RASTER": 2,
            },
            "RASTER",
            2,
        )

        profile = select_sampling_profile(
            settings.engine_3_id,
            3,
            history,
        )

        self.assertEqual(
            profile,
            SD35_ALTERNATE_SAMPLER,
        )

    def test_sd35_targeted_raster_uses_revised_schedule(
        self,
    ):
        history = FailureHistorySummary(
            {
                "RASTER": 3,
            },
            "RASTER",
            3,
        )

        profile = select_sampling_profile(
            settings.engine_3_id,
            4,
            history,
        )

        self.assertEqual(
            profile,
            SD35_REVISED_SCHEDULE,
        )

    def test_sd35_strict_stage_uses_rescue(
        self,
    ):
        history = FailureHistorySummary(
            {
                "RASTER": 6,
            },
            "RASTER",
            6,
        )

        profile = select_sampling_profile(
            settings.engine_3_id,
            7,
            history,
        )

        self.assertEqual(
            profile,
            SD35_RESCUE,
        )

    def test_sd35_attempt_nine_is_rescue(
        self,
    ):
        history = FailureHistorySummary(
            {
                "RASTER": 8,
            },
            "RASTER",
            8,
        )

        profile = select_sampling_profile(
            settings.engine_3_id,
            9,
            history,
        )

        self.assertEqual(
            profile,
            SD35_RESCUE,
        )

    def test_flux_keeps_existing_sampling_path(
        self,
    ):
        history = FailureHistorySummary(
            {},
            "",
            0,
        )

        profile = select_sampling_profile(
            settings.engine_1_id,
            1,
            history,
        )

        self.assertIsNone(
            profile
        )


if __name__ == "__main__":
    unittest.main()