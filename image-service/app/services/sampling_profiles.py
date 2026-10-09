"""Engine-specific sampling choices selected from persisted failure history.

Every named sampler and scheduler below is checked against the live ComfyUI
``/object_info`` schema before a workflow is submitted.
"""

from dataclasses import asdict, dataclass

from app.config.settings import settings
from app.services.prompt_compiler import failure_category


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    name: str
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    shift: float | None = None

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureHistorySummary:
    counts: dict[str, int]
    consecutive_category: str
    consecutive_count: int


def summarize_failures(record: dict | None) -> FailureHistorySummary:
    if not record:
        return FailureHistorySummary({}, "", 0)
    attempts = list(record.get("rejected_attempts") or [])
    attempts += list(record.get("failed_attempts") or [])
    if record.get("status") in {"rejected", "failed", "runtime_failed"}:
        attempts.append(record)
    attempts.sort(key=lambda item: int(item.get("attempt_count") or 0))
    counts: dict[str, int] = {}
    categories = []
    for item in attempts:
        category = failure_category(item.get("rejection_reason") or item.get("error") or "")
        categories.append(category)
        counts[category] = counts.get(category, 0) + 1
    last = categories[-1] if categories else ""
    consecutive = 0
    for category in reversed(categories):
        if category != last:
            break
        consecutive += 1
    return FailureHistorySummary(counts, last or "", consecutive)


_SDXL = (
    SamplingProfile("baseline", 28, 6.0, "dpmpp_2m", "karras"),
    SamplingProfile("palette_guidance", 32, 6.5, "dpmpp_2m", "karras"),
    SamplingProfile("palette_alternate", 32, 5.5, "euler", "normal"),
    SamplingProfile("palette_strict", 36, 7.0, "dpmpp_2m", "normal"),
)

_SD35 = (
    # Baseline matches the official ComfyUI SD3.5 example: checkpoint model
    # and VAE, Euler/SGM uniform, 20 steps, CFG near 4, no shift override.
    SamplingProfile("official_baseline", 20, 4.0, "euler", "sgm_uniform"),
    SamplingProfile("alternate_sampler", 28, 4.0, "dpmpp_2m", "sgm_uniform"),
    SamplingProfile("revised_schedule", 30, 4.0, "euler", "simple", 2.0),
    SamplingProfile("raster_rescue_trial", 30, 3.5, "dpmpp_2m", "simple", 2.0),
)


def select_sampling_profile(engine_id: str, attempt: int, history: FailureHistorySummary) -> SamplingProfile | None:
    if not 1 <= attempt <= 9:
        raise ValueError("One candidate permits attempts 1 through 9 only")
    if engine_id == settings.engine_2_id:
        repeats = history.counts.get("PALETTE_OFF", 0)
        if repeats >= 4 or attempt == 9:
            return _SDXL[3]
        return _SDXL[min(repeats, 2)]
    if engine_id == settings.engine_3_id:
        repeats = history.counts.get("RASTER", 0)
        if repeats >= 4 or attempt == 9:
            return _SD35[3]
        if repeats >= 3:
            return _SD35[2]
        if repeats >= 2:
            return _SD35[1]
        return _SD35[0]
    return None
