"""
colour_filter.py — Colour-based hold filtering for route detection.

PROBLEM
=======
A climbing wall has many routes set simultaneously, each using holds of a
specific colour. YOLO detects ALL holds regardless of colour. To find the
optimal route for a specific problem, we need to filter to only the holds
belonging to that route's colour.

APPROACH
========
For each detected hold, we sample the pixels inside its bounding box from
the original image, compute the dominant colour in HSV space using K-means
clustering, and assign it to a named colour category.

WHY HSV NOT RGB?
================
RGB mixes colour and brightness — a dark red and a light red look very
different in RGB but both have the same Hue in HSV. HSV separates:
    H (Hue)        — the actual colour (0-179 in OpenCV)
    S (Saturation) — how vivid vs grey (0-255)
    V (Value)      — brightness (0-255)

We cluster on Hue only (for well-lit, saturated holds) and use Saturation
to exclude grey/white/black holds (chalk, wall texture, shadows).

COLOUR CATEGORIES
=================
Maps hue ranges to named colours. OpenCV uses 0-179 for hue (half of 360°).

    Red:    0-10 and 165-179  (wraps around)
    Orange: 10-25
    Yellow: 25-35
    Green:  35-85
    Blue:   85-130
    Purple: 130-165
    Pink:   Same range as red but lower saturation
    White:  Low saturation, high value
    Black:  Low saturation, low value

EXTENSION IDEAS
===============
    - Let user click on a hold in the image to auto-select its colour
    - Handle multi-colour routes (some gyms use two colours)
    - Confidence score per hold: how sure are we it's the right colour?
    - Handle chalk-covered holds: chalk reduces saturation, shifts apparent colour
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from src.model.detected_object import DetectedObject


# ── Colour definitions ────────────────────────────────────────────────────────

# Named colours available for filtering
COLOUR_NAMES = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black"
]

# Hue ranges in OpenCV HSV (0-179)
# Each entry: (hue_low, hue_high, sat_min, val_min)
# Red wraps around 0/179 so it has two ranges
COLOUR_RANGES: Dict[str, List[Tuple[int, int, int, int]]] = {
    "red":    [(0, 10, 100, 60), (165, 179, 100, 60)],
    "orange": [(10, 25, 120, 80)],
    "yellow": [(25, 38, 100, 100)],
    "green":  [(38, 85, 80, 60)],
    "blue":   [(85, 130, 80, 60)],
    "purple": [(130, 165, 60, 60)],
    "pink":   [(0, 15, 50, 150), (160, 179, 50, 150)],
    "white":  [(0, 179, 0, 180)],   # any hue, low sat, high val
    "black":  [(0, 179, 0, 0)],     # any hue, low sat, low val
}

# Saturation thresholds for white/black detection
WHITE_SAT_MAX = 60
WHITE_VAL_MIN = 160
BLACK_SAT_MAX = 60
BLACK_VAL_MAX = 80


@dataclass
class ColourAnnotatedHold:
    """A detected hold with its colour classification."""
    detected_object: DetectedObject
    colour: str              # e.g. "red", "blue", "green"
    confidence: float        # 0-1, fraction of pixels matching the colour
    dominant_hue: int        # dominant hue value (0-179)
    dominant_sat: int        # dominant saturation


def classify_hold_colour(
    img: cv2.typing.MatLike,
    det: DetectedObject,
    min_saturation: int = 60,
) -> ColourAnnotatedHold:
    """
    Classify the colour of a hold by analysing pixels inside its bounding box.

    Steps:
        1. Crop the bounding box from the image
        2. Shrink the crop by 15% to avoid wall texture at hold edges
        3. Convert to HSV
        4. Filter out low-saturation pixels (chalk, wall texture, shadows)
        5. Find dominant hue using a histogram
        6. Map dominant hue to a named colour category

    Args:
        img            : Original BGR image
        det            : Detected hold with bounding box
        min_saturation : Pixels below this saturation are ignored (chalk/wall)
    """
    x1, y1, x2, y2 = det.bbox

    # Shrink crop by 15% to avoid wall edges bleeding into the sample
    pad_x = max(2, int((x2 - x1) * 0.15))
    pad_y = max(2, int((y2 - y1) * 0.15))
    x1c = min(x1 + pad_x, x2 - 1)
    y1c = min(y1 + pad_y, y2 - 1)
    x2c = max(x2 - pad_x, x1c + 1)
    y2c = max(y2 - pad_y, y1c + 1)

    crop = img[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return ColourAnnotatedHold(det, "unknown", 0.0, 0, 0)

    # Convert to HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # White detection: low saturation, high value
    white_mask = (s < WHITE_SAT_MAX) & (v > WHITE_VAL_MIN)
    if white_mask.sum() > 0.5 * crop.size // 3:
        return ColourAnnotatedHold(det, "white", float(white_mask.mean()), 0, 0)

    # Black detection: low saturation, low value
    black_mask = (s < BLACK_SAT_MAX) & (v < BLACK_VAL_MAX)
    if black_mask.sum() > 0.5 * crop.size // 3:
        return ColourAnnotatedHold(det, "black", float(black_mask.mean()), 0, 0)

    # Filter: keep only well-saturated, well-lit pixels (actual hold colour)
    valid_mask = (s >= min_saturation) & (v >= 40)
    valid_hues = h[valid_mask]
    valid_sats = s[valid_mask]

    if len(valid_hues) < 10:
        # Not enough coloured pixels — probably a grey/white hold
        return ColourAnnotatedHold(det, "white", 0.3, 0, 0)

    # Dominant hue: use histogram with 36 bins (5° per bin)
    hist, bin_edges = np.histogram(valid_hues, bins=36, range=(0, 180))
    dominant_bin = int(np.argmax(hist))
    dominant_hue = int(bin_edges[dominant_bin] + 2.5)  # bin centre
    dominant_sat = int(np.median(valid_sats))
    confidence = float(hist[dominant_bin] / len(valid_hues))

    # Map hue to colour name
    colour = _hue_to_colour_name(dominant_hue, dominant_sat)

    return ColourAnnotatedHold(
        detected_object=det,
        colour=colour,
        confidence=confidence,
        dominant_hue=dominant_hue,
        dominant_sat=dominant_sat,
    )


def _hue_to_colour_name(hue: int, saturation: int) -> str:
    """Map a hue value (0-179) to a named colour."""
    if saturation < 50:
        return "white"

    # Red wraps around — check both ends
    if hue <= 10 or hue >= 165:
        return "red"
    if 10 < hue <= 25:
        return "orange"
    if 25 < hue <= 38:
        return "yellow"
    if 38 < hue <= 85:
        return "green"
    if 85 < hue <= 130:
        return "blue"
    if 130 < hue < 165:
        return "purple"
    return "unknown"


def classify_all_holds(
    img: cv2.typing.MatLike,
    detected_objects: List[DetectedObject],
) -> List[ColourAnnotatedHold]:
    """Classify the colour of every detected hold."""
    return [classify_hold_colour(img, det) for det in detected_objects]


def filter_by_colour(
    annotated_holds: List[ColourAnnotatedHold],
    target_colour: str,
    min_confidence: float = 0.2,
) -> List[DetectedObject]:
    """
    Return only holds matching the target colour.

    Args:
        annotated_holds : All holds with colour classifications
        target_colour   : Colour name to keep e.g. "yellow", "blue"
        min_confidence  : Minimum classification confidence (0-1)

    Returns:
        List of DetectedObject for holds matching the colour.
    """
    target = target_colour.lower().strip()
    return [
        h.detected_object for h in annotated_holds
        if h.colour == target and h.confidence >= min_confidence
    ]


def get_colour_summary(
    annotated_holds: List[ColourAnnotatedHold],
) -> Dict[str, int]:
    """
    Count holds by colour — useful for discovering which colours are on the wall.

    Returns dict like: {"red": 12, "blue": 9, "yellow": 7, "green": 5}
    """
    summary: Dict[str, int] = {}
    for h in annotated_holds:
        summary[h.colour] = summary.get(h.colour, 0) + 1
    return dict(sorted(summary.items(), key=lambda x: -x[1]))


def draw_colour_annotations(
    img: cv2.typing.MatLike,
    annotated_holds: List[ColourAnnotatedHold],
    target_colour: Optional[str] = None,
) -> cv2.typing.MatLike:
    """
    Draw colour labels on holds. If target_colour given, highlight matches
    and fade non-matches.

    Useful for debugging colour detection and for the /wall/colours endpoint.
    """
    img_out = img.copy()

    # BGR colour mapping for drawing
    DRAW_COLOURS = {
        "red":    (0, 0, 220),
        "orange": (0, 140, 255),
        "yellow": (0, 220, 220),
        "green":  (0, 200, 0),
        "blue":   (220, 100, 0),
        "purple": (200, 0, 180),
        "pink":   (180, 100, 200),
        "white":  (240, 240, 240),
        "black":  (30, 30, 30),
        "unknown": (128, 128, 128),
    }

    for h in annotated_holds:
        x1, y1, x2, y2 = h.detected_object.bbox
        colour_bgr = DRAW_COLOURS.get(h.colour, (128, 128, 128))

        is_match = (target_colour is None) or (h.colour == target_colour.lower())
        alpha = 1.0 if is_match else 0.25
        thickness = 3 if is_match else 1

        # Draw bbox
        overlay = img_out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour_bgr, thickness)
        cv2.addWeighted(overlay, alpha, img_out, 1 - alpha, 0, img_out)

        # Label
        if is_match:
            label = f"{h.colour} {h.confidence:.0%}"
            cv2.putText(img_out, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour_bgr, 1)

    return img_out
