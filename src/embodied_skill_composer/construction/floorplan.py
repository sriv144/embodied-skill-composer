from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from embodied_skill_composer.construction.models import (
    Opening,
    Room,
    Vec2,
    VectorFloorPlan,
    WallSegment,
)


MAX_FLOORPLAN_ENCODED_BYTES = 8 * 1024 * 1024
MAX_FLOORPLAN_DIMENSION_PX = 8_192
MAX_FLOORPLAN_PIXELS = 24_000_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_START_OF_FRAME_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


@dataclass(frozen=True)
class _FootprintCandidate:
    x: int
    y: int
    width: int
    height: int
    contour_area: float
    rectangularity: float
    side_support: tuple[float, float, float, float]

    @property
    def bounds_area(self) -> int:
        return self.width * self.height

    @property
    def mean_side_support(self) -> float:
        return float(np.mean(self.side_support))


def infer_orthogonal_floor_plan(
    image_bytes: bytes,
    *,
    known_width_m: float,
) -> VectorFloorPlan:
    """Infer one reviewable orthogonal exterior footprint from a raster plan.

    This intentionally stops at a conservative exterior-shell interpretation. Raster
    inference can propose wall gaps as openings, but the returned plan is never approved
    and its opening types remain explicitly provisional.
    """
    if not isfinite(known_width_m) or known_width_m <= 0:
        raise ValueError("known floor-plan width must be a finite positive number")
    if len(image_bytes) > MAX_FLOORPLAN_ENCODED_BYTES:
        raise ValueError(
            "uploaded floor plan exceeds the 8 MiB encoded-image limit"
        )
    encoded_width, encoded_height = _encoded_image_dimensions(image_bytes)
    _require_bounded_image_dimensions(encoded_width, encoded_height)

    cv2 = _load_opencv()
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) if encoded.size else None
    if image is None:
        raise ValueError("uploaded file is not a decodable image")

    image_height, image_width = image.shape
    _require_bounded_image_dimensions(image_width, image_height)
    if image_width < 64 or image_height < 64:
        raise ValueError("uploaded floor plan is too small to calibrate (minimum 64 x 64 pixels)")

    low, high = (float(value) for value in np.percentile(image, (5, 95)))
    contrast = high - low
    if contrast < 18:
        raise ValueError("floor plan has insufficient contrast to identify geometry")

    foreground = _foreground_mask(image, cv2)
    raw_candidates = _find_candidates(foreground, cv2, require_closed=True)
    _reject_competing_footprints(raw_candidates)

    bridge_x = max(7, int(round(image_width * 0.13)))
    bridge_y = max(7, int(round(image_height * 0.13)))
    horizontal = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_x, 3)),
    )
    vertical = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, bridge_y)),
    )
    bridged = cv2.bitwise_or(horizontal, vertical)
    candidates = _find_candidates(bridged, cv2, require_closed=False)
    _reject_competing_footprints(candidates)
    if not candidates:
        raise ValueError(
            "floor plan is ambiguous: no supported orthogonal exterior rectangle was found"
        )

    candidate = max(candidates, key=lambda item: item.bounds_area)
    if candidate.bounds_area < image_width * image_height * 0.08:
        raise ValueError("detected floor plan is too small to calibrate")
    if candidate.rectangularity < 0.78 or min(candidate.side_support) < 0.52:
        raise ValueError(
            "floor plan is ambiguous: exterior geometry is not a supported orthogonal rectangle"
        )

    x, y = candidate.x, candidate.y
    width_px, height_px = candidate.width, candidate.height
    if width_px < 32 or height_px < 32:
        raise ValueError("detected floor plan is too small to calibrate")

    depth_m = known_width_m * height_px / width_px
    half_w, half_d = known_width_m / 2, depth_m / 2
    walls = [
        WallSegment(
            wall_id="north",
            start=Vec2(x=-half_w, y=half_d),
            end=Vec2(x=half_w, y=half_d),
        ),
        WallSegment(
            wall_id="east",
            start=Vec2(x=half_w, y=half_d),
            end=Vec2(x=half_w, y=-half_d),
        ),
        WallSegment(
            wall_id="south",
            start=Vec2(x=half_w, y=-half_d),
            end=Vec2(x=-half_w, y=-half_d),
        ),
        WallSegment(
            wall_id="west",
            start=Vec2(x=-half_w, y=-half_d),
            end=Vec2(x=-half_w, y=half_d),
        ),
    ]
    openings = _infer_openings(
        foreground,
        candidate,
        width_m=known_width_m,
        depth_m=depth_m,
        cv2=cv2,
    )

    contrast_score = min(1.0, contrast / 120.0)
    confidence = float(
        np.clip(
            0.38
            + 0.25 * candidate.mean_side_support
            + 0.13 * candidate.rectangularity
            + 0.08 * contrast_score
            - (0.03 if openings else 0.0),
            0.45,
            0.86,
        )
    )
    warnings = [
        "Exterior footprint inferred from one dominant orthogonal contour.",
        "Room boundaries are unclassified and require review.",
    ]
    if openings:
        warnings.append(
            f"{len(openings)} exterior opening candidate(s) inferred from wall gaps; "
            "door/window labels and dimensions are provisional."
        )
    else:
        warnings.append("No exterior openings were inferred; review or add openings manually.")
    warnings.append(
        f"Source bounds: x={x}, y={y}, width={width_px}, height={height_px} pixels."
    )

    return VectorFloorPlan(
        walls=walls,
        openings=openings,
        rooms=[
            Room(
                room_id="unclassified_space",
                name="Review room boundaries",
                polygon=[
                    Vec2(x=-half_w, y=-half_d),
                    Vec2(x=half_w, y=-half_d),
                    Vec2(x=half_w, y=half_d),
                    Vec2(x=-half_w, y=half_d),
                ],
            )
        ],
        confidence=confidence,
        warnings=warnings,
        approved=False,
    )


def _encoded_image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without allocating a decoded raster."""

    if image_bytes.startswith(_PNG_SIGNATURE):
        if (
            len(image_bytes) < 24
            or image_bytes[12:16] != b"IHDR"
        ):
            raise ValueError(
                "uploaded file is not a decodable image "
                "(supported formats: PNG and JPEG)"
            )
        return (
            int.from_bytes(image_bytes[16:20], "big"),
            int.from_bytes(image_bytes[20:24], "big"),
        )
    if not image_bytes.startswith(b"\xff\xd8"):
        raise ValueError(
            "uploaded file is not a decodable image "
            "(supported formats: PNG and JPEG)"
        )

    index = 2
    while index < len(image_bytes):
        if image_bytes[index] != 0xFF:
            raise ValueError(
                "uploaded file is not a decodable image "
                "(supported formats: PNG and JPEG)"
            )
        while index < len(image_bytes) and image_bytes[index] == 0xFF:
            index += 1
        if index >= len(image_bytes):
            break
        marker = image_bytes[index]
        index += 1
        if marker in {0x01, *range(0xD0, 0xDA)}:
            continue
        if marker == 0x00 or index + 2 > len(image_bytes):
            break
        segment_length = int.from_bytes(image_bytes[index : index + 2], "big")
        if (
            segment_length < 2
            or index + segment_length > len(image_bytes)
        ):
            break
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            if segment_length < 7:
                break
            height = int.from_bytes(image_bytes[index + 3 : index + 5], "big")
            width = int.from_bytes(image_bytes[index + 5 : index + 7], "big")
            return width, height
        if marker == 0xDA:
            break
        index += segment_length
    raise ValueError(
        "uploaded file is not a decodable image "
        "(supported formats: PNG and JPEG)"
    )


def _require_bounded_image_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(
            "uploaded file is not a decodable image "
            "(supported formats: PNG and JPEG)"
        )
    if (
        width > MAX_FLOORPLAN_DIMENSION_PX
        or height > MAX_FLOORPLAN_DIMENSION_PX
    ):
        raise ValueError(
            "uploaded floor plan exceeds the 8192-pixel image-dimension limit"
        )
    if width * height > MAX_FLOORPLAN_PIXELS:
        raise ValueError(
            "uploaded floor plan exceeds the 24-megapixel decoded-image limit"
        )


def _load_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for floor-plan image parsing") from exc
    return cv2


def _foreground_mask(image: np.ndarray, cv2) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, dark = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    _, light = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    masks = [dark, light]
    ratios = [float(np.count_nonzero(mask)) / mask.size for mask in masks]
    viable = [
        (ratio, mask)
        for ratio, mask in zip(ratios, masks, strict=True)
        if 0.002 <= ratio <= 0.60
    ]
    if not viable:
        raise ValueError("floor plan has no separable foreground geometry")
    _, selected = min(viable, key=lambda item: item[0])

    component_floor = max(8, int(round(selected.size * 0.000025)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, connectivity=8)
    cleaned = np.zeros_like(selected)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= component_floor:
            cleaned[labels == label] = 255
    if not np.any(cleaned):
        raise ValueError("no floor-plan geometry was detected")
    return np.asarray(cleaned, dtype=np.uint8)


def _find_candidates(
    mask: np.ndarray,
    cv2,
    *,
    require_closed: bool,
) -> list[_FootprintCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = mask.shape
    candidates: list[_FootprintCandidate] = []
    for contour in contours:
        x, y, width, height = (int(value) for value in cv2.boundingRect(contour))
        if width < max(24, int(image_width * 0.10)):
            continue
        if height < max(24, int(image_height * 0.10)):
            continue
        contour_area = float(cv2.contourArea(contour))
        rectangularity = contour_area / max(1, width * height)
        side_support = _side_support(mask, x, y, width, height)
        if require_closed and (rectangularity < 0.72 or min(side_support) < 0.70):
            continue
        if not require_closed and (
            rectangularity < 0.40 or float(np.mean(side_support)) < 0.58
        ):
            continue
        candidates.append(
            _FootprintCandidate(
                x=x,
                y=y,
                width=width,
                height=height,
                contour_area=contour_area,
                rectangularity=rectangularity,
                side_support=side_support,
            )
        )
    return sorted(candidates, key=lambda item: item.bounds_area, reverse=True)


def _side_support(
    mask: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    band = max(3, int(round(min(width, height) * 0.045)))
    right = min(mask.shape[1], x + width)
    bottom = min(mask.shape[0], y + height)
    top_presence = np.any(mask[y : min(bottom, y + band), x:right] > 0, axis=0)
    bottom_presence = np.any(mask[max(y, bottom - band) : bottom, x:right] > 0, axis=0)
    left_presence = np.any(mask[y:bottom, x : min(right, x + band)] > 0, axis=1)
    right_presence = np.any(mask[y:bottom, max(x, right - band) : right] > 0, axis=1)
    return (
        float(np.mean(top_presence)),
        float(np.mean(right_presence)),
        float(np.mean(bottom_presence)),
        float(np.mean(left_presence)),
    )


def _reject_competing_footprints(candidates: list[_FootprintCandidate]) -> None:
    if len(candidates) < 2:
        return
    largest, runner_up = candidates[:2]
    if runner_up.bounds_area >= largest.bounds_area * 0.45:
        raise ValueError("floor plan is ambiguous: multiple plausible exterior footprints detected")


def _infer_openings(
    mask: np.ndarray,
    footprint: _FootprintCandidate,
    *,
    width_m: float,
    depth_m: float,
    cv2,
) -> list[Opening]:
    x, y = footprint.x, footprint.y
    width_px, height_px = footprint.width, footprint.height
    band = max(3, int(round(min(width_px, height_px) * 0.045)))
    right, bottom = x + width_px, y + height_px
    side_presence = {
        "north": np.any(mask[y : y + band, x:right] > 0, axis=0),
        "east": np.any(mask[y:bottom, right - band : right] > 0, axis=1),
        "south": np.any(mask[bottom - band : bottom, x:right] > 0, axis=0),
        "west": np.any(mask[y:bottom, x : x + band] > 0, axis=1),
    }
    side_lengths_m = {
        "north": width_m,
        "east": depth_m,
        "south": width_m,
        "west": depth_m,
    }
    openings: list[Opening] = []
    for wall_id in ("north", "east", "south", "west"):
        presence = _close_short_signal_holes(side_presence[wall_id], cv2)
        side_length_px = int(presence.size)
        margin = max(4, int(round(side_length_px * 0.035)))
        minimum_gap = max(7, int(round(side_length_px * 0.035)))
        maximum_gap = int(round(side_length_px * 0.24))
        intervals = _false_intervals(presence)
        candidate_number = 0
        for start, end in intervals:
            gap = end - start
            if start < margin or end > side_length_px - margin:
                continue
            if gap < minimum_gap or gap > maximum_gap:
                continue
            candidate_number += 1
            center_px = (start + end) / 2
            metric_length = side_lengths_m[wall_id]
            opening_width_m = metric_length * gap / side_length_px
            offset_m = metric_length * center_px / side_length_px
            if wall_id in {"south", "west"}:
                offset_m = metric_length - offset_m
            kind: Literal["door", "window"] = (
                "door" if opening_width_m <= 1.25 else "window"
            )
            openings.append(
                Opening(
                    opening_id=f"candidate_{wall_id}_{candidate_number:02d}",
                    wall_id=wall_id,
                    kind=kind,
                    offset_m=offset_m,
                    width_m=opening_width_m,
                    height_m=2.1 if kind == "door" else 1.2,
                    sill_height_m=0.0 if kind == "door" else 0.9,
                )
            )
    return openings


def _close_short_signal_holes(presence: np.ndarray, cv2) -> np.ndarray:
    signal = np.asarray(presence, dtype=np.uint8).reshape(1, -1) * 255
    kernel_width = max(3, int(round(signal.shape[1] * 0.012)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    closed = cv2.morphologyEx(signal, cv2.MORPH_CLOSE, kernel)
    return np.asarray(closed.reshape(-1) > 0, dtype=bool)


def _false_intervals(signal: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(
        (
            np.array([True], dtype=bool),
            np.asarray(signal, dtype=bool),
            np.array([True], dtype=bool),
        )
    )
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(end))
        for start, end in zip(transitions[::2], transitions[1::2], strict=True)
    ]
