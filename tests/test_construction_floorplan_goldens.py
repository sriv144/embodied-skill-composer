from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from embodied_skill_composer.construction import floorplan as floorplan_module
from embodied_skill_composer.construction.floorplan import (
    MAX_FLOORPLAN_ENCODED_BYTES,
    infer_orthogonal_floor_plan,
)


MANIFEST_PATH = (
    Path(__file__).parent / "fixtures" / "construction_floorplans" / "manifest.json"
)
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
GOLDENS: list[dict[str, Any]] = MANIFEST["fixtures"]


def _render_fixture(fixture: dict[str, Any]) -> bytes:
    if "raw_hex" in fixture:
        return bytes.fromhex(fixture["raw_hex"])

    height, width = fixture["canvas"]
    background = fixture["background"]
    foreground = fixture["foreground"]
    image = np.full((height, width), background, dtype=np.uint8)
    thickness = fixture["wall_thickness"]

    rectangles = fixture.get("rectangles")
    if rectangles is None and "rectangle" in fixture:
        rectangles = [fixture["rectangle"]]
    for left, top, right, bottom in rectangles or []:
        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            foreground,
            thickness=thickness,
        )

    if "polygon" in fixture:
        points = np.asarray(fixture["polygon"], dtype=np.int32)
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=foreground,
            thickness=thickness,
        )

    erase_color = background
    half_thickness = thickness
    if "rectangle" in fixture:
        left, top, right, bottom = fixture["rectangle"]
        for gap in fixture.get("gaps", []):
            start, end = gap["start"], gap["end"]
            side = gap["side"]
            if side == "north":
                cv2.rectangle(
                    image,
                    (start, top - half_thickness),
                    (end, top + half_thickness),
                    erase_color,
                    thickness=-1,
                )
            elif side == "south":
                cv2.rectangle(
                    image,
                    (start, bottom - half_thickness),
                    (end, bottom + half_thickness),
                    erase_color,
                    thickness=-1,
                )
            elif side == "east":
                cv2.rectangle(
                    image,
                    (right - half_thickness, start),
                    (right + half_thickness, end),
                    erase_color,
                    thickness=-1,
                )
            elif side == "west":
                cv2.rectangle(
                    image,
                    (left - half_thickness, start),
                    (left + half_thickness, end),
                    erase_color,
                    thickness=-1,
                )
            else:
                raise AssertionError(f"unsupported fixture gap side: {side}")

    noise = fixture.get("noise")
    if noise:
        rng = np.random.default_rng(noise["seed"])
        points_y = rng.integers(0, height, size=noise["count"])
        points_x = rng.integers(0, width, size=noise["count"])
        radius = noise["radius"]
        for point_x, point_y in zip(points_x, points_y, strict=True):
            cv2.circle(
                image,
                (int(point_x), int(point_y)),
                radius,
                foreground,
                thickness=-1,
            )

    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize("fixture", GOLDENS, ids=[item["name"] for item in GOLDENS])
def test_floorplan_parser_golden(fixture: dict[str, Any]) -> None:
    expected = fixture["expect"]
    payload = _render_fixture(fixture)
    if expected["result"] == "error":
        with pytest.raises(ValueError, match=expected["message"]):
            infer_orthogonal_floor_plan(
                payload,
                known_width_m=fixture["known_width_m"],
            )
        return

    inferred = infer_orthogonal_floor_plan(
        payload,
        known_width_m=fixture["known_width_m"],
    )
    width = abs(inferred.walls[0].end.x - inferred.walls[0].start.x)
    depth = abs(inferred.walls[1].end.y - inferred.walls[1].start.y)

    assert inferred.approved is False
    assert len(inferred.walls) == 4
    assert len(inferred.rooms) == 1
    assert width == pytest.approx(fixture["known_width_m"])
    assert expected["depth_m"][0] <= depth <= expected["depth_m"][1]
    assert len(inferred.openings) == expected["openings"]
    assert expected["confidence"][0] <= inferred.confidence <= expected["confidence"][1]
    assert inferred.warnings
    assert any("review" in warning.lower() for warning in inferred.warnings)


def test_golden_manifest_is_versioned_and_has_exactly_twelve_unique_cases() -> None:
    assert MANIFEST["schema_version"] == 1
    assert len(GOLDENS) == 12
    names = [item["name"] for item in GOLDENS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("known_width_m", [0.0, -2.0, float("inf"), float("nan")])
def test_floorplan_parser_rejects_invalid_calibration(known_width_m: float) -> None:
    payload = _render_fixture(GOLDENS[0])
    with pytest.raises(ValueError, match="finite positive"):
        infer_orthogonal_floor_plan(payload, known_width_m=known_width_m)


def test_ambiguous_goldens_can_never_become_approved() -> None:
    ambiguous = [
        fixture
        for fixture in GOLDENS
        if fixture["expect"]["result"] == "error"
        and fixture["name"].startswith("ambiguous_")
    ]
    assert ambiguous
    for fixture in ambiguous:
        with pytest.raises(ValueError, match="ambiguous"):
            infer_orthogonal_floor_plan(
                _render_fixture(fixture),
                known_width_m=fixture["known_width_m"],
            )


def test_floorplan_parser_rejects_encoded_payload_over_limit() -> None:
    with pytest.raises(ValueError, match="8 MiB"):
        infer_orthogonal_floor_plan(
            b"x" * (MAX_FLOORPLAN_ENCODED_BYTES + 1),
            known_width_m=8,
        )


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (8_193, 64, "8192-pixel"),
        (5_000, 5_000, "24-megapixel"),
    ],
)
def test_oversized_compressed_header_is_rejected_before_decode(
    width: int,
    height: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )

    def decoding_must_not_start() -> object:
        raise AssertionError("OpenCV must not load for an oversized image header")

    monkeypatch.setattr(
        floorplan_module,
        "_load_opencv",
        decoding_must_not_start,
    )
    with pytest.raises(ValueError, match=message):
        infer_orthogonal_floor_plan(header, known_width_m=8)


def test_jpeg_header_dimensions_are_supported_without_decode() -> None:
    jpeg = (
        b"\xff\xd8"
        b"\xff\xe0\x00\x04\x00\x00"
        b"\xff\xc0\x00\x11\x08"
        b"\x01\xe0\x02\x80"
        b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    )

    assert floorplan_module._encoded_image_dimensions(jpeg) == (640, 480)
