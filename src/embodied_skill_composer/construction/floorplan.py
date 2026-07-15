from __future__ import annotations

import numpy as np

from embodied_skill_composer.construction.models import (
    Opening,
    Room,
    Vec2,
    VectorFloorPlan,
    WallSegment,
)


def infer_orthogonal_floor_plan(
    image_bytes: bytes,
    *,
    known_width_m: float,
) -> VectorFloorPlan:
    """Infer a reviewable exterior rectangle from a clean high-contrast floor plan."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for floor-plan image parsing") from exc
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("uploaded file is not a decodable image")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no floor-plan geometry was detected")
    x, y, width_px, height_px = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if width_px < 20 or height_px < 20:
        raise ValueError("detected floor plan is too small to calibrate")
    depth_m = known_width_m * height_px / width_px
    half_w, half_d = known_width_m / 2, depth_m / 2
    walls = [
        WallSegment(wall_id="north", start=Vec2(x=-half_w, y=half_d), end=Vec2(x=half_w, y=half_d)),
        WallSegment(wall_id="east", start=Vec2(x=half_w, y=half_d), end=Vec2(x=half_w, y=-half_d)),
        WallSegment(wall_id="south", start=Vec2(x=half_w, y=-half_d), end=Vec2(x=-half_w, y=-half_d)),
        WallSegment(wall_id="west", start=Vec2(x=-half_w, y=-half_d), end=Vec2(x=-half_w, y=half_d)),
    ]
    return VectorFloorPlan(
        walls=walls,
        openings=[
            Opening(
                opening_id="candidate_front_door",
                wall_id="south",
                kind="door",
                offset_m=known_width_m / 2,
                width_m=1.0,
                height_m=2.1,
            )
        ],
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
        confidence=0.62,
        warnings=[
            "Exterior footprint inferred from the largest contour.",
            "Door and room boundaries are candidates and require review.",
            f"Source bounds: x={x}, y={y}, width={width_px}, height={height_px} pixels.",
        ],
        approved=False,
    )
