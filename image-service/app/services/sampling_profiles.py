"""Engine-specific sampling profiles and deterministic retry escalation.

Retry stage and sampling profile are separate concepts.

Retry stage describes recovery intensity:

    attempts 1-3 -> normal
    attempts 4-6 -> targeted_recovery
    attempts 7-8 -> strict_recovery
    attempt 9    -> rescue

Sampling profile describes the actual native model configuration used within
that stage.

The SD3.5 production baseline is intentionally based on empirical programme
studio testing rather than the original ComfyUI example baseline.

Controlled probes using the same CandidateSpec and seed showed:

    Euler / SGM / 20                         -> 0.18 off-palette
    Euler / SGM / 20 / explicit shift 3      -> 0.18
    Euler / SGM / 20 / explicit shift 2      -> 0.18
    Euler / normal / 20                      -> 0.18
    DPM++ 2M / SGM / 20                      -> 0.14
    DPM++ 2M / SGM / 28                      -> 0.09
    Euler / simple / 30 / shift 2            -> 0.17
    DPM++ 2M / simple / 30 / shift 2         -> 0.14 in the
                                                preserved diagnostic run

The DPM++ 2M / SGM Uniform / 28-step profile also passed the non-palette
structural QA without raster corruption.

Therefore that profile is now the SD3.5 production baseline.

Every named sampler and scheduler is still checked against the live ComfyUI
/object_info schema before workflow submission.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)

from app.config.settings import settings
from app.services.prompt_compiler import (
    failure_category,
    retry_stage,
)


# ============================================================
# Models
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class SamplingProfile:
    name: str
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    shift: float | None = None

    def record(
        self,
    ) -> dict:
        return {
            **asdict(
                self
            ),
            "shift_mode": (
                "checkpoint_default"
                if self.shift is None
                else "explicit"
            ),
        }


@dataclass(
    frozen=True,
    slots=True,
)
class FailureHistorySummary:
    counts: dict[str, int]
    consecutive_category: str
    consecutive_count: int

    def count(
        self,
        category: str,
    ) -> int:
        return int(
            self.counts.get(
                category,
                0,
            )
        )


# ============================================================
# SDXL production profiles
# ============================================================


SDXL_BASELINE = SamplingProfile(
    name="baseline",
    steps=28,
    cfg=6.0,
    sampler_name="dpmpp_2m",
    scheduler="karras",
)

SDXL_PALETTE_GUIDANCE = SamplingProfile(
    name="palette_guidance",
    steps=32,
    cfg=6.5,
    sampler_name="dpmpp_2m",
    scheduler="karras",
)

SDXL_PALETTE_ALTERNATE = SamplingProfile(
    name="palette_alternate",
    steps=32,
    cfg=5.5,
    sampler_name="euler",
    scheduler="normal",
)

SDXL_STRICT = SamplingProfile(
    name="palette_strict",
    steps=36,
    cfg=7.0,
    sampler_name="dpmpp_2m",
    scheduler="normal",
)


# ============================================================
# SD3.5 profiles
# ============================================================


#
# Retained for controlled diagnostics.
#
# This is no longer the production baseline because our controlled
# programme-studio test produced materially worse palette adherence than
# DPM++ 2M / SGM Uniform / 28 steps.
#
SD35_OFFICIAL_BASELINE = SamplingProfile(
    name="official_baseline",
    steps=20,
    cfg=4.0,
    sampler_name="euler",
    scheduler="sgm_uniform",
    shift=None,
)


#
# Empirically best SD3.5 production baseline so far.
#
SD35_ALTERNATE_SAMPLER = SamplingProfile(
    name="alternate_sampler",
    steps=28,
    cfg=4.0,
    sampler_name="dpmpp_2m",
    scheduler="sgm_uniform",
    shift=None,
)


#
# Give the production role an explicit semantic name without changing the
# persisted profile name used by diagnostics and historical records.
#
SD35_PRODUCTION_BASELINE = (
    SD35_ALTERNATE_SAMPLER
)


#
# Retained as a controlled alternate.
#
# It is not preferred for palette recovery because empirical testing produced
# more neutral drift than the production baseline.
#
SD35_REVISED_SCHEDULE = SamplingProfile(
    name="revised_schedule",
    steps=30,
    cfg=4.0,
    sampler_name="euler",
    scheduler="simple",
    shift=2.0,
)


#
# Strict/rescue SD3.5 profile.
#
# It produced structurally clean output in the preserved diagnostic run and
# remains useful when we are specifically escaping repeated raster failures.
#
SD35_RESCUE = SamplingProfile(
    name="raster_rescue_trial",
    steps=30,
    cfg=3.5,
    sampler_name="dpmpp_2m",
    scheduler="simple",
    shift=2.0,
)


# ============================================================
# Controlled diagnostic profiles
# ============================================================


_SDXL_PROBE = (
    SDXL_BASELINE,
    SamplingProfile(
        name="cfg_5",
        steps=28,
        cfg=5.0,
        sampler_name="dpmpp_2m",
        scheduler="karras",
    ),
    SamplingProfile(
        name="euler_karras",
        steps=28,
        cfg=6.0,
        sampler_name="euler",
        scheduler="karras",
    ),
    SamplingProfile(
        name="dpmpp_normal",
        steps=28,
        cfg=6.0,
        sampler_name="dpmpp_2m",
        scheduler="normal",
    ),
    SDXL_PALETTE_GUIDANCE,
    SDXL_PALETTE_ALTERNATE,
    SDXL_STRICT,
)


_SD35_PROBE = (
    SD35_OFFICIAL_BASELINE,

    SamplingProfile(
        name="explicit_shift_3",
        steps=20,
        cfg=4.0,
        sampler_name="euler",
        scheduler="sgm_uniform",
        shift=3.0,
    ),

    SamplingProfile(
        name="explicit_shift_2",
        steps=20,
        cfg=4.0,
        sampler_name="euler",
        scheduler="sgm_uniform",
        shift=2.0,
    ),

    SamplingProfile(
        name="euler_normal",
        steps=20,
        cfg=4.0,
        sampler_name="euler",
        scheduler="normal",
        shift=None,
    ),

    SamplingProfile(
        name="dpmpp_2m_sgm",
        steps=20,
        cfg=4.0,
        sampler_name="dpmpp_2m",
        scheduler="sgm_uniform",
        shift=None,
    ),

    SD35_ALTERNATE_SAMPLER,
    SD35_REVISED_SCHEDULE,
    SD35_RESCUE,
)


# ============================================================
# Failure-history analysis
# ============================================================


def summarize_failures(
    record: dict | None,
) -> FailureHistorySummary:
    if not record:
        return FailureHistorySummary(
            counts={},
            consecutive_category="",
            consecutive_count=0,
        )

    attempts = list(
        record.get(
            "rejected_attempts"
        )
        or []
    )

    attempts += list(
        record.get(
            "failed_attempts"
        )
        or []
    )

    if (
        record.get(
            "status"
        )
        in {
            "rejected",
            "failed",
            "runtime_failed",
        }
    ):
        attempts.append(
            record
        )

    attempts.sort(
        key=lambda item: int(
            item.get(
                "attempt_count"
            )
            or 0
        )
    )

    counts: dict[str, int] = {}

    categories: list[str] = []

    for item in attempts:
        category = failure_category(
            item.get(
                "rejection_reason"
            )
            or item.get(
                "error"
            )
            or ""
        )

        categories.append(
            category
        )

        counts[
            category
        ] = (
            counts.get(
                category,
                0,
            )
            + 1
        )

    last = (
        categories[-1]
        if categories
        else ""
    )

    consecutive = 0

    for category in reversed(
        categories
    ):
        if (
            category
            != last
        ):
            break

        consecutive += 1

    return FailureHistorySummary(
        counts=counts,
        consecutive_category=(
            last
            or ""
        ),
        consecutive_count=consecutive,
    )


# ============================================================
# Probe lookup
# ============================================================


def probe_profiles(
    engine_id: str,
) -> tuple[
    SamplingProfile,
    ...,
]:
    if (
        engine_id
        == settings.engine_2_id
    ):
        return (
            _SDXL_PROBE
        )

    if (
        engine_id
        == settings.engine_3_id
    ):
        return (
            _SD35_PROBE
        )

    raise ValueError(
        "No native profile matrix for "
        f"{engine_id}"
    )


def profile_by_name(
    engine_id: str,
    name: str,
) -> SamplingProfile:
    for profile in probe_profiles(
        engine_id
    ):
        if (
            profile.name
            == name
        ):
            return (
                profile
            )

    raise ValueError(
        f"Unknown {engine_id} sampling "
        f"profile: {name}"
    )


# ============================================================
# SDXL production selection
# ============================================================


def _select_sdxl_profile(
    stage: str,
    history: FailureHistorySummary,
) -> SamplingProfile:
    palette_count = (
        history.count(
            "PALETTE_OFF"
        )
        + history.count(
            "PRIMARY_MISSING"
        )
        + history.count(
            "SECONDARY_MISSING"
        )
    )

    repeated_palette = bool(
        history.consecutive_category
        in {
            "PALETTE_OFF",
            "PRIMARY_MISSING",
            "SECONDARY_MISSING",
        }
        and history.consecutive_count
        >= 2
    )

    # --------------------------------------------------------
    # Attempts 1-3: normal
    # --------------------------------------------------------

    if (
        stage
        == "normal"
    ):
        if (
            repeated_palette
            or palette_count
            >= 2
        ):
            return (
                SDXL_PALETTE_GUIDANCE
            )

        return (
            SDXL_BASELINE
        )

    # --------------------------------------------------------
    # Attempts 4-6: targeted recovery
    # --------------------------------------------------------

    if (
        stage
        == "targeted_recovery"
    ):
        if (
            repeated_palette
            or palette_count
            >= 3
        ):
            return (
                SDXL_PALETTE_ALTERNATE
            )

        return (
            SDXL_PALETTE_GUIDANCE
        )

    # --------------------------------------------------------
    # Attempts 7-8: strict recovery
    # --------------------------------------------------------

    if (
        stage
        == "strict_recovery"
    ):
        return (
            SDXL_STRICT
        )

    # --------------------------------------------------------
    # Attempt 9: rescue
    # --------------------------------------------------------

    if (
        stage
        == "rescue"
    ):
        return (
            SDXL_STRICT
        )

    raise ValueError(
        "Unsupported SDXL retry stage: "
        f"{stage}"
    )


# ============================================================
# SD3.5 production selection
# ============================================================


def _select_sd35_profile(
    stage: str,
    history: FailureHistorySummary,
) -> SamplingProfile:
    raster_count = (
        history.count(
            "RASTER"
        )
    )

    repeated_raster = bool(
        history.consecutive_category
        == "RASTER"
        and history.consecutive_count
        >= 2
    )

    #
    # PALETTE_OFF is deliberately NOT used to move SD3.5 away from the
    # production baseline.
    #
    # The production baseline is our best empirically measured palette
    # performer, and mild palette errors now have a deterministic correction
    # path in ComfyUIClient.
    #

    # --------------------------------------------------------
    # Attempts 1-3: normal
    # --------------------------------------------------------

    if (
        stage
        == "normal"
    ):
        #
        # Stay on the known-good DPM++/SGM profile unless repeated raster
        # evidence specifically says that this profile is failing
        # structurally.
        #
        if (
            repeated_raster
            or raster_count
            >= 2
        ):
            return (
                SD35_RESCUE
            )

        return (
            SD35_PRODUCTION_BASELINE
        )

    # --------------------------------------------------------
    # Attempts 4-6: targeted recovery
    # --------------------------------------------------------

    if (
        stage
        == "targeted_recovery"
    ):
        if (
            history.consecutive_category
            == "RASTER"
            or raster_count
            >= 2
        ):
            return (
                SD35_RESCUE
            )

        #
        # For palette, human, text, flatness or other non-raster failures,
        # keep the empirically best native profile. Prompt recovery is handled
        # separately by CandidateSpec/CompiledPrompt.
        #
        return (
            SD35_PRODUCTION_BASELINE
        )

    # --------------------------------------------------------
    # Attempts 7-8: strict recovery
    # --------------------------------------------------------

    if (
        stage
        == "strict_recovery"
    ):
        return (
            SD35_RESCUE
        )

    # --------------------------------------------------------
    # Attempt 9: rescue
    # --------------------------------------------------------

    if (
        stage
        == "rescue"
    ):
        return (
            SD35_RESCUE
        )

    raise ValueError(
        "Unsupported SD3.5 retry stage: "
        f"{stage}"
    )


# ============================================================
# Public production selector
# ============================================================


def select_sampling_profile(
    engine_id: str,
    attempt: int,
    history: FailureHistorySummary,
) -> SamplingProfile | None:
    if not (
        1
        <= attempt
        <= 9
    ):
        raise ValueError(
            "One candidate permits attempts "
            "1 through 9 only"
        )

    stage = (
        retry_stage(
            attempt
        )
    )

    if (
        engine_id
        == settings.engine_2_id
    ):
        return (
            _select_sdxl_profile(
                stage,
                history,
            )
        )

    if (
        engine_id
        == settings.engine_3_id
    ):
        return (
            _select_sd35_profile(
                stage,
                history,
            )
        )

    #
    # FLUX keeps its existing generation path.
    #
    return None
