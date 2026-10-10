from __future__ import annotations

import cv2
import numpy as np

from PIL import Image

from app.config.settings import settings


# ============================================================
# Sampling
# ============================================================

QUALITY_SAMPLE_WIDTH = settings.quality_sample_width
QUALITY_SAMPLE_HEIGHT = settings.quality_sample_height


# ============================================================
# General edge detection
# ============================================================

QUALITY_EDGE_THRESHOLD = 30.0

QUALITY_DIRECTIONAL_MULTIPLIER = 2.75

MAX_DIRECTIONAL_EDGE_DOMINANCE = 0.82

MIN_DIRECTIONAL_EDGE_DENSITY = 0.16

MIN_FULL_SPAN_LINE_RATIO = 0.18


# ============================================================
# Fine raster / scanline detection
# ============================================================

FINE_LINE_PIXEL_DELTA = 10.0

FINE_LINE_FULL_SPAN_COVERAGE = 0.60

MAX_FINE_LINE_PAIR_RATIO = 0.24

MIN_FINE_LINE_MEAN_DELTA = 4.5


# ============================================================
# Flat / degenerate output detection
# ============================================================

MAX_SINGLE_QUANTISED_COLOUR_RATIO = 0.965

MIN_FLAT_IMAGE_EDGE_DENSITY = 0.010

MIN_FLAT_IMAGE_ENTROPY = 2.10


# ============================================================
# Smooth-gradient / texture-only detection
# ============================================================

STRUCTURAL_BLUR_SIGMA = 1.25

MIN_STRUCTURAL_EDGE_DENSITY = 0.014

MIN_OUTER_STRUCTURAL_EDGE_DENSITY = 0.035

LOW_STRUCTURE_MAX_LAPLACIAN_VARIANCE = 30.0

LOW_STRUCTURE_MAX_HIGHPASS_MEAN = 3.5


# ============================================================
# Noise / incoherent texture detection
# ============================================================

CANNY_LOW_THRESHOLD = 60

CANNY_HIGH_THRESHOLD = 140

MIN_COHERENT_COMPONENT_SIZE = 18

MIN_COHERENT_EDGE_RATIO = 0.16

NOISE_CHECK_MIN_RAW_EDGE_DENSITY = 0.10

NOISE_CHECK_MAX_STRUCTURAL_EDGE_DENSITY = 0.035


# ============================================================
# Programme/poster composition
# ============================================================

CENTER_X_START = settings.quality_center_x_start
CENTER_X_END = settings.quality_center_x_end

CENTER_Y_START = settings.quality_center_y_start
CENTER_Y_END = settings.quality_center_y_end

# The programme title and schedule occupy the centre.  Decorative detail is
# welcome at the edge, but a densely illustrated centre makes the candidate
# unusable even when it passes technical image checks.
MAX_CENTER_STRUCTURAL_EDGE_DENSITY = settings.quality_center_max_edge_density

MAX_CENTER_LOCAL_CONTRAST = settings.quality_center_max_local_contrast


def _image_entropy(
    gray: np.ndarray,
) -> float:
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256],
    ).reshape(
        -1
    )

    total = float(
        np.sum(
            histogram
        )
    )

    if total <= 0:
        return 0.0

    probabilities = (
        histogram[
            histogram > 0
        ]
        / total
    )

    return float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities
            )
        )
    )


def _largest_quantised_colour_ratio(
    rgb: np.ndarray,
) -> float:
    quantised = (
        rgb
        // 32
    ).reshape(
        -1,
        3,
    )

    _, counts = np.unique(
        quantised,
        axis=0,
        return_counts=True,
    )

    if len(
        counts
    ) == 0:
        return 1.0

    return float(
        np.max(
            counts
        )
        / np.sum(
            counts
        )
    )


def _sobel_metrics(
    gray_float: np.ndarray,
    *,
    threshold: float,
) -> dict:
    sobel_x = np.abs(
        cv2.Sobel(
            gray_float,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
    )

    sobel_y = np.abs(
        cv2.Sobel(
            gray_float,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )
    )

    magnitude = np.hypot(
        sobel_x,
        sobel_y,
    )

    edge_mask = (
        magnitude
        >= threshold
    )

    edge_density = float(
        np.mean(
            edge_mask
        )
    )

    horizontal_mask = (
        (
            sobel_y
            >= threshold
        )
        & (
            sobel_y
            > (
                sobel_x
                * QUALITY_DIRECTIONAL_MULTIPLIER
            )
        )
    )

    vertical_mask = (
        (
            sobel_x
            >= threshold
        )
        & (
            sobel_x
            > (
                sobel_y
                * QUALITY_DIRECTIONAL_MULTIPLIER
            )
        )
    )

    return {
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "magnitude": magnitude,
        "edge_mask": edge_mask,
        "edge_density": edge_density,
        "horizontal_mask": horizontal_mask,
        "vertical_mask": vertical_mask,
        "horizontal_density": float(
            np.mean(
                horizontal_mask
            )
        ),
        "vertical_density": float(
            np.mean(
                vertical_mask
            )
        ),
    }


def _directional_metrics(
    metrics: dict,
) -> dict:
    edge_density = (
        metrics[
            "edge_density"
        ]
    )

    horizontal_density = (
        metrics[
            "horizontal_density"
        ]
    )

    vertical_density = (
        metrics[
            "vertical_density"
        ]
    )

    strongest = max(
        horizontal_density,
        vertical_density,
    )

    dominance = (
        strongest
        / max(
            edge_density,
            1e-6,
        )
    )

    horizontal_row_coverage = (
        np.mean(
            metrics[
                "horizontal_mask"
            ],
            axis=1,
        )
    )

    vertical_column_coverage = (
        np.mean(
            metrics[
                "vertical_mask"
            ],
            axis=0,
        )
    )

    horizontal_full_span_ratio = float(
        np.mean(
            horizontal_row_coverage
            >= 0.55
        )
    )

    vertical_full_span_ratio = float(
        np.mean(
            vertical_column_coverage
            >= 0.55
        )
    )

    full_span = max(
        horizontal_full_span_ratio,
        vertical_full_span_ratio,
    )

    axis = (
        "horizontal"
        if (
            horizontal_density
            >= vertical_density
        )
        else "vertical"
    )

    return {
        "directional_edge_dominance": (
            dominance
        ),
        "horizontal_full_span_ratio": (
            horizontal_full_span_ratio
        ),
        "vertical_full_span_ratio": (
            vertical_full_span_ratio
        ),
        "full_span_line_ratio": (
            full_span
        ),
        "dominant_directional_axis": (
            axis
        ),
    }


def _fine_line_metrics(
    gray_float: np.ndarray,
) -> dict:
    horizontal_delta = np.abs(
        np.diff(
            gray_float,
            axis=0,
        )
    )

    vertical_delta = np.abs(
        np.diff(
            gray_float,
            axis=1,
        )
    )

    horizontal_pair_coverage = (
        np.mean(
            horizontal_delta
            >= FINE_LINE_PIXEL_DELTA,
            axis=1,
        )
    )

    vertical_pair_coverage = (
        np.mean(
            vertical_delta
            >= FINE_LINE_PIXEL_DELTA,
            axis=0,
        )
    )

    horizontal_pair_ratio = float(
        np.mean(
            horizontal_pair_coverage
            >= FINE_LINE_FULL_SPAN_COVERAGE
        )
    )

    vertical_pair_ratio = float(
        np.mean(
            vertical_pair_coverage
            >= FINE_LINE_FULL_SPAN_COVERAGE
        )
    )

    horizontal_mean_delta = float(
        np.mean(
            horizontal_delta
        )
    )

    vertical_mean_delta = float(
        np.mean(
            vertical_delta
        )
    )

    strongest_ratio = max(
        horizontal_pair_ratio,
        vertical_pair_ratio,
    )

    strongest_mean_delta = (
        horizontal_mean_delta
        if (
            horizontal_pair_ratio
            >= vertical_pair_ratio
        )
        else vertical_mean_delta
    )

    axis = (
        "horizontal"
        if (
            horizontal_pair_ratio
            >= vertical_pair_ratio
        )
        else "vertical"
    )

    return {
        "fine_horizontal_pair_ratio": (
            horizontal_pair_ratio
        ),
        "fine_vertical_pair_ratio": (
            vertical_pair_ratio
        ),
        "fine_line_pair_ratio": (
            strongest_ratio
        ),
        "fine_horizontal_mean_delta": (
            horizontal_mean_delta
        ),
        "fine_vertical_mean_delta": (
            vertical_mean_delta
        ),
        "fine_line_mean_delta": (
            strongest_mean_delta
        ),
        "fine_line_axis": (
            axis
        ),
    }


def _coherent_edge_ratio(
    gray: np.ndarray,
) -> float:
    canny = cv2.Canny(
        gray,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
    )

    edge_pixels = int(
        np.count_nonzero(
            canny
        )
    )

    if edge_pixels == 0:
        return 0.0

    (
        count,
        _,
        stats,
        _,
    ) = (
        cv2.connectedComponentsWithStats(
            (
                canny > 0
            ).astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    coherent_pixels = 0

    for label in range(
        1,
        count,
    ):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if (
            area
            >= MIN_COHERENT_COMPONENT_SIZE
        ):
            coherent_pixels += area

    return float(
        coherent_pixels
        / edge_pixels
    )


def _region_metrics(
    structural_edge_mask: np.ndarray,
    gray_float: np.ndarray,
    *,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
) -> dict:
    height, width = (
        gray_float.shape
    )

    x0 = max(
        0,
        min(
            width,
            int(
                width
                * x_start
            ),
        ),
    )

    x1 = max(
        x0,
        min(
            width,
            int(
                width
                * x_end
            ),
        ),
    )

    y0 = max(
        0,
        min(
            height,
            int(
                height
                * y_start
            ),
        ),
    )

    y1 = max(
        y0,
        min(
            height,
            int(
                height
                * y_end
            ),
        ),
    )

    region_edges = (
        structural_edge_mask[
            y0:y1,
            x0:x1,
        ]
    )

    region_gray = (
        gray_float[
            y0:y1,
            x0:x1,
        ]
    )

    return {
        "edge_density": (
            float(
                np.mean(
                    region_edges
                )
            )
            if region_edges.size
            else 0.0
        ),
        "local_contrast": (
            float(
                np.std(
                    region_gray
                )
            )
            if region_gray.size
            else 0.0
        ),
    }


def _safe_region_metrics(
    structural_edge_mask: np.ndarray,
    gray_float: np.ndarray,
) -> dict:
    height, width = (
        gray_float.shape
    )

    center = _region_metrics(
        structural_edge_mask,
        gray_float,
        x_start=CENTER_X_START,
        x_end=CENTER_X_END,
        y_start=CENTER_Y_START,
        y_end=CENTER_Y_END,
    )

    #
    # Measure the inner text cores of the current title/programme overlays,
    # not their full bounding boxes. Decorative framing is intentionally
    # allowed near the box edges, while the text-bearing interior must remain
    # calm.
    #
    title = _region_metrics(
        structural_edge_mask,
        gray_float,
        x_start=0.18,
        x_end=0.82,
        y_start=0.09,
        y_end=0.19,
    )

    programme = _region_metrics(
        structural_edge_mask,
        gray_float,
        x_start=0.18,
        x_end=0.82,
        y_start=0.51,
        y_end=0.83,
    )

    x0 = int(
        width
        * CENTER_X_START
    )

    x1 = int(
        width
        * CENTER_X_END
    )

    y0 = int(
        height
        * CENTER_Y_START
    )

    y1 = int(
        height
        * CENTER_Y_END
    )

    center_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=bool,
    )

    center_mask[
        y0:y1,
        x0:x1,
    ] = True

    outer_mask = (
        ~center_mask
    )

    outer_edges = (
        structural_edge_mask[
            outer_mask
        ]
    )

    outer_edge_density = float(
        np.mean(
            outer_edges
        )
    ) if (
        outer_edges.size
    ) else 0.0

    return {
        "center_structural_edge_density": (
            center[
                "edge_density"
            ]
        ),
        "outer_structural_edge_density": (
            outer_edge_density
        ),
        "center_local_contrast": (
            center[
                "local_contrast"
            ]
        ),
        "title_structural_edge_density": (
            title[
                "edge_density"
            ]
        ),
        "title_local_contrast": (
            title[
                "local_contrast"
            ]
        ),
        "programme_structural_edge_density": (
            programme[
                "edge_density"
            ]
        ),
        "programme_local_contrast": (
            programme[
                "local_contrast"
            ]
        ),
    }


def _reject_visual_defects(metrics: dict, directional: dict, fine_lines: dict, safe_region: dict) -> None:
    edge_density = metrics["edge_density"]
    directional_edge_dominance = metrics["directional_edge_dominance"]
    full_span_line_ratio = metrics["full_span_line_ratio"]
    largest_colour_ratio = metrics["largest_colour_ratio"]
    entropy = metrics["entropy"]
    structural_edge_density = metrics["structural_edge_density"]
    laplacian_variance = metrics["laplacian_variance"]
    highpass_mean = metrics["highpass_mean"]
    coherent_edge_ratio = metrics["coherent_edge_ratio"]

    # ========================================================
    # Strong directional scanlines
    # ========================================================

    if (
        edge_density
        >= MIN_DIRECTIONAL_EDGE_DENSITY
        and directional_edge_dominance
        >= MAX_DIRECTIONAL_EDGE_DOMINANCE
        and full_span_line_ratio
        >= MIN_FULL_SPAN_LINE_RATIO
    ):
        axis = (
            directional[
                "dominant_directional_axis"
            ]
        )

        raise ValueError(
            "Generated background failed visual quality validation: "
            f"excessive repetitive {axis} scanline or raster-band "
            "structure was detected "
            f"(edge density {edge_density:.2f}, "
            f"directional dominance "
            f"{directional_edge_dominance:.2f}, "
            f"full-span ratio "
            f"{full_span_line_ratio:.2f})."
        )

    # ========================================================
    # Fine repetitive raster structure
    # ========================================================

    if (
        fine_lines[
            "fine_line_pair_ratio"
        ]
        >= MAX_FINE_LINE_PAIR_RATIO
        and fine_lines[
            "fine_line_mean_delta"
        ]
        >= MIN_FINE_LINE_MEAN_DELTA
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "fine repetitive "
            f"{fine_lines['fine_line_axis']} raster-line structure "
            "covers too much of the page "
            f"(full-span adjacent-line ratio "
            f"{fine_lines['fine_line_pair_ratio']:.2f}, "
            f"mean adjacent-line delta "
            f"{fine_lines['fine_line_mean_delta']:.2f})."
        )

    # ========================================================
    # Blank / flat output
    # ========================================================

    if (
        largest_colour_ratio
        >= MAX_SINGLE_QUANTISED_COLOUR_RATIO
        and edge_density
        < MIN_FLAT_IMAGE_EDGE_DENSITY
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "the image is effectively a flat or blank colour field "
            f"(dominant quantised colour ratio "
            f"{largest_colour_ratio:.2f}, "
            f"edge density "
            f"{edge_density:.3f})."
        )

    if (
        entropy
        < MIN_FLAT_IMAGE_ENTROPY
        and edge_density
        < MIN_FLAT_IMAGE_EDGE_DENSITY
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "the image contains insufficient visual structure "
            f"(entropy "
            f"{entropy:.2f}, "
            f"edge density "
            f"{edge_density:.3f})."
        )

    # ========================================================
    # Smooth gradient / colour wash
    # ========================================================

    if (
        structural_edge_density
        < MIN_STRUCTURAL_EDGE_DENSITY
        and laplacian_variance
        < LOW_STRUCTURE_MAX_LAPLACIAN_VARIANCE
        and highpass_mean
        < LOW_STRUCTURE_MAX_HIGHPASS_MEAN
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "the image is primarily a smooth gradient or colour wash "
            "with insufficient designed structure "
            f"(structural edge density "
            f"{structural_edge_density:.3f}, "
            f"laplacian variance "
            f"{laplacian_variance:.2f}, "
            f"high-pass mean "
            f"{highpass_mean:.2f})."
        )

    # A subtle pixel texture can raise Laplacian variance while the intended
    # subject remains barely visible. Require useful visible structure when
    # both the sampled edges and local contrast are very low.
    if (
        structural_edge_density < 0.045
        and edge_density < 0.12
        and highpass_mean < 2.0
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "the subject has insufficient visible structure beneath fine texture "
            f"(structural edge density {structural_edge_density:.3f}, "
            f"edge density {edge_density:.3f}, high-pass mean {highpass_mean:.2f})."
        )

    # ========================================================
    # Noise / grain without coherent forms
    # ========================================================

    if (
        edge_density
        >= NOISE_CHECK_MIN_RAW_EDGE_DENSITY
        and structural_edge_density
        <= NOISE_CHECK_MAX_STRUCTURAL_EDGE_DENSITY
        and coherent_edge_ratio
        < MIN_COHERENT_EDGE_RATIO
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "the image contains high-frequency texture or noise "
            "without enough coherent designed structure "
            f"(raw edge density "
            f"{edge_density:.2f}, "
            f"structural edge density "
            f"{structural_edge_density:.3f}, "
            f"coherent edge ratio "
            f"{coherent_edge_ratio:.2f})."
        )

    # ========================================================
    # Useful outer decorative structure
    # ========================================================

    if (
        safe_region[
            "outer_structural_edge_density"
        ]
        < MIN_OUTER_STRUCTURAL_EDGE_DENSITY
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "the outer background area contains too little useful "
            "decorative structure for a programme background "
            f"(outer structural edge density "
            f"{safe_region['outer_structural_edge_density']:.3f})."
        )



def validate_visual_quality(
    image: Image.Image,
    *,
    enforce_overlay_region: bool = False,
) -> dict:
    sample = (
        image.convert(
            "RGB"
        )
        .resize(
            (
                QUALITY_SAMPLE_WIDTH,
                QUALITY_SAMPLE_HEIGHT,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    rgb = np.array(
        sample,
        dtype=np.uint8,
    )

    # Preserve original pixels for fine-line detection: resizing can blur
    # one-pixel scanline corruption below the rejection threshold.
    source_rgb = np.array(
        image.convert(
            "RGB"
        ),
        dtype=np.uint8,
    )

    source_gray_float = cv2.cvtColor(
        source_rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(
        np.float32
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    gray_float = gray.astype(
        np.float32
    )

    raw_metrics = (
        _sobel_metrics(
            gray_float,
            threshold=(
                QUALITY_EDGE_THRESHOLD
            ),
        )
    )

    directional = (
        _directional_metrics(
            raw_metrics
        )
    )

    fine_lines = (
        _fine_line_metrics(
            source_gray_float
        )
    )

    blurred = cv2.GaussianBlur(
        gray_float,
        (
            0,
            0,
        ),
        sigmaX=(
            STRUCTURAL_BLUR_SIGMA
        ),
        sigmaY=(
            STRUCTURAL_BLUR_SIGMA
        ),
    )

    structural_metrics = (
        _sobel_metrics(
            blurred,
            threshold=(
                QUALITY_EDGE_THRESHOLD
            ),
        )
    )

    structural_edge_density = float(
        structural_metrics[
            "edge_density"
        ]
    )

    safe_region = (
        _safe_region_metrics(
            structural_metrics[
                "edge_mask"
            ],
            gray_float,
        )
    )

    entropy = (
        _image_entropy(
            gray
        )
    )

    largest_colour_ratio = (
        _largest_quantised_colour_ratio(
            rgb
        )
    )

    laplacian = cv2.Laplacian(
        gray_float,
        cv2.CV_32F,
    )

    laplacian_variance = float(
        np.var(
            laplacian
        )
    )

    highpass = np.abs(
        gray_float
        - blurred
    )

    highpass_mean = float(
        np.mean(
            highpass
        )
    )

    coherent_edge_ratio = (
        _coherent_edge_ratio(
            gray
        )
    )

    edge_density = float(
        raw_metrics[
            "edge_density"
        ]
    )

    horizontal_density = float(
        raw_metrics[
            "horizontal_density"
        ]
    )

    vertical_density = float(
        raw_metrics[
            "vertical_density"
        ]
    )

    directional_edge_dominance = float(
        directional[
            "directional_edge_dominance"
        ]
    )

    full_span_line_ratio = float(
        directional[
            "full_span_line_ratio"
        ]
    )

    _reject_visual_defects(
        {
            "edge_density": edge_density,
            "directional_edge_dominance": directional_edge_dominance,
            "full_span_line_ratio": full_span_line_ratio,
            "largest_colour_ratio": largest_colour_ratio,
            "entropy": entropy,
            "structural_edge_density": structural_edge_density,
            "laplacian_variance": laplacian_variance,
            "highpass_mean": highpass_mean,
            "coherent_edge_ratio": coherent_edge_ratio,
        },
        directional,
        fine_lines,
        safe_region,
    )

    # ========================================================
    # Programme/title safe-region quality
    # ========================================================
    #
    # Programme Studio backgrounds are not merely decorative images.
    #
    # The title and programme content must be composited over the central
    # portion of the artwork. A technically valid illustration can therefore
    # still be unusable when architectural detail, line work or other visual
    # structure fills the centre.
    #
    # We do not require a blank centre. Instead, the centre must be
    # meaningfully calmer than the decorative outer region.
    #

    center_edge_density = float(
        safe_region[
            "center_structural_edge_density"
        ]
    )

    outer_edge_density = float(
        safe_region[
            "outer_structural_edge_density"
        ]
    )

    center_local_contrast = float(
        safe_region[
            "center_local_contrast"
        ]
    )

    #
    # Generic central-zone limits.
    #
    # A centre with meaningful structure must still be clearly calmer than the
    # outer decoration. This catches candidates whose absolute density is only
    # moderate but whose centre is almost as busy as the edges.
    #
    MAX_USABLE_CENTER_EDGE_DENSITY = 0.10
    MIN_MEANINGFUL_CENTER_EDGE_DENSITY = 0.06
    MAX_CENTER_TO_OUTER_EDGE_RATIO = 0.80

    #
    # Text-core limits for the title and programme overlays.
    #
    # These are intentionally applied to inset text cores rather than the full
    # overlay boxes, so edge framing can touch the outside of a title/programme
    # panel without making the actual text-bearing area unusable.
    #
    MAX_TITLE_ZONE_EDGE_DENSITY = 0.10
    MAX_PROGRAMME_ZONE_EDGE_DENSITY = 0.10

    center_to_outer_ratio = (
        center_edge_density
        / max(
            outer_edge_density,
            1e-6,
        )
    )

    dense_center = bool(
        center_edge_density
        > MAX_USABLE_CENTER_EDGE_DENSITY
    )

    insufficient_edge_bias = bool(
        center_edge_density
        > MIN_MEANINGFUL_CENTER_EDGE_DENSITY
        and center_to_outer_ratio
        > MAX_CENTER_TO_OUTER_EDGE_RATIO
    )

    excessive_center_contrast = bool(
        center_local_contrast
        > MAX_CENTER_LOCAL_CONTRAST
    )

    title_edge_density = float(
        safe_region[
            "title_structural_edge_density"
        ]
    )

    programme_edge_density = float(
        safe_region[
            "programme_structural_edge_density"
        ]
    )

    if enforce_overlay_region and (
        title_edge_density
        > MAX_TITLE_ZONE_EDGE_DENSITY
        or programme_edge_density
        > MAX_PROGRAMME_ZONE_EDGE_DENSITY
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "programme safe region is too visually dense "
            f"(title structural edge density "
            f"{title_edge_density:.3f}, "
            f"programme structural edge density "
            f"{programme_edge_density:.3f}, "
            f"outer structural edge density "
            f"{outer_edge_density:.3f})."
        )

    if enforce_overlay_region and (
        dense_center
        or insufficient_edge_bias
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "programme safe region is too visually dense "
            f"(center structural edge density "
            f"{center_edge_density:.3f}, "
            f"outer structural edge density "
            f"{outer_edge_density:.3f}, "
            f"center-to-outer ratio "
            f"{center_to_outer_ratio:.2f})."
        )

    if enforce_overlay_region and (
        excessive_center_contrast
        and insufficient_edge_bias
    ):
        raise ValueError(
            "Generated background failed visual quality validation: "
            "programme safe region has too much local contrast "
            f"(center local contrast "
            f"{center_local_contrast:.2f}, "
            f"center-to-outer ratio "
            f"{center_to_outer_ratio:.2f})."
        )

    return {
        "visual_quality_checked": True,
        "visual_quality_passed": True,

        "edge_density": (
            edge_density
        ),

        "structural_edge_density": (
            structural_edge_density
        ),

        "horizontal_edge_density": (
            horizontal_density
        ),

        "vertical_edge_density": (
            vertical_density
        ),

        "directional_edge_dominance": (
            directional_edge_dominance
        ),

        "full_span_line_ratio": (
            full_span_line_ratio
        ),

        "fine_horizontal_pair_ratio": (
            fine_lines[
                "fine_horizontal_pair_ratio"
            ]
        ),

        "fine_vertical_pair_ratio": (
            fine_lines[
                "fine_vertical_pair_ratio"
            ]
        ),

        "fine_line_pair_ratio": (
            fine_lines[
                "fine_line_pair_ratio"
            ]
        ),

        "fine_line_mean_delta": (
            fine_lines[
                "fine_line_mean_delta"
            ]
        ),

        "image_entropy": (
            entropy
        ),

        "largest_quantised_colour_ratio": (
            largest_colour_ratio
        ),

        "laplacian_variance": (
            laplacian_variance
        ),

        "highpass_mean": (
            highpass_mean
        ),

        "coherent_edge_ratio": (
            coherent_edge_ratio
        ),

        "center_structural_edge_density": (
            safe_region[
                "center_structural_edge_density"
            ]
        ),

        "outer_structural_edge_density": (
            safe_region[
                "outer_structural_edge_density"
            ]
        ),

        "center_local_contrast": (
            safe_region[
                "center_local_contrast"
            ]
        ),

        "title_structural_edge_density": (
            safe_region[
                "title_structural_edge_density"
            ]
        ),

        "title_local_contrast": (
            safe_region[
                "title_local_contrast"
            ]
        ),

        "programme_structural_edge_density": (
            safe_region[
                "programme_structural_edge_density"
            ]
        ),

        "programme_local_contrast": (
            safe_region[
                "programme_local_contrast"
            ]
        ),

        "center_to_outer_edge_ratio": (
            center_to_outer_ratio
        ),

        # The composer places adaptive blurred panels behind text. Raw image
        # detail is useful selection data, but is not itself a readability
        # failure. The optional strict mode remains for diagnostics.
        "center_safe_region_passed": not (
            title_edge_density > MAX_TITLE_ZONE_EDGE_DENSITY
            or programme_edge_density > MAX_PROGRAMME_ZONE_EDGE_DENSITY
            or dense_center
            or insufficient_edge_bias
            or (excessive_center_contrast and insufficient_edge_bias)
        ),

        
    }
