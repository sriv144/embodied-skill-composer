from __future__ import annotations

from math import hypot, isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from embodied_skill_composer.construction.models import HouseDesign, Opening, WallSegment


GEOMETRY_EPSILON_M = 1e-9

DesignValidationCode = Literal[
    "duplicate_id",
    "insufficient_walls",
    "invalid_id",
    "non_finite_dimension",
    "non_positive_dimension",
    "roof_pitch_out_of_range",
    "wall_not_axis_aligned",
    "wall_zero_length",
    "wall_out_of_bounds",
    "wall_intersection",
    "wall_overlap",
    "opening_unknown_wall",
    "opening_out_of_bounds",
    "opening_overlap",
    "opening_vertical_overflow",
]


class DesignValidationIssue(BaseModel):
    """A stable, field-addressable design error for API and editor consumers."""

    model_config = ConfigDict(frozen=True)

    code: DesignValidationCode
    path: str
    message: str


class DesignValidationResult(BaseModel):
    """Deterministic validation result shared by preview, rebuild, and compilation."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    issues: list[DesignValidationIssue] = Field(default_factory=list)


class DesignValidationError(ValueError):
    """Raised when an operation requires a valid architectural design."""

    def __init__(self, result: DesignValidationResult):
        self.result = result
        summary = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        super().__init__(summary or "house design is invalid")


def validate_house_design(design: HouseDesign) -> DesignValidationResult:
    """Validate editable metric geometry without changing the HouseDesign wire format."""

    issues: list[DesignValidationIssue] = []
    _validate_identifier(design.design_id, "design_id", issues)
    _validate_positive(design.footprint_width_m, "footprint_width_m", issues)
    _validate_positive(design.footprint_depth_m, "footprint_depth_m", issues)
    _validate_nonnegative(design.roof.overhang_m, "roof.overhang_m", issues)
    _validate_finite(design.roof.pitch_degrees, "roof.pitch_degrees", issues)
    if isfinite(design.roof.pitch_degrees) and not 0 <= design.roof.pitch_degrees <= 60:
        _issue(
            issues,
            "roof_pitch_out_of_range",
            "roof.pitch_degrees",
            "roof pitch must be between 0 and 60 degrees",
        )
    if len(design.floor_plan.walls) < 4:
        _issue(
            issues,
            "insufficient_walls",
            "floor_plan.walls",
            "floor plan must contain at least four walls",
        )

    _validate_unique_ids(
        [(wall.wall_id, f"floor_plan.walls[{index}].wall_id") for index, wall in enumerate(
            design.floor_plan.walls
        )],
        issues,
    )
    _validate_unique_ids(
        [
            (opening.opening_id, f"floor_plan.openings[{index}].opening_id")
            for index, opening in enumerate(design.floor_plan.openings)
        ],
        issues,
    )
    _validate_unique_ids(
        [(room.room_id, f"floor_plan.rooms[{index}].room_id") for index, room in enumerate(
            design.floor_plan.rooms
        )],
        issues,
    )

    valid_footprint = _is_positive_finite(design.footprint_width_m) and _is_positive_finite(
        design.footprint_depth_m
    )
    for index, wall in enumerate(design.floor_plan.walls):
        _validate_wall(wall, index, design, valid_footprint, issues)
    _validate_wall_relationships(design.floor_plan.walls, issues)
    _validate_openings(design, issues)

    ordered = sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message))
    return DesignValidationResult(valid=not ordered, issues=ordered)


def require_valid_house_design(design: HouseDesign) -> None:
    """Raise a typed error if any shared design-safety rule fails."""

    result = validate_house_design(design)
    if not result.valid:
        raise DesignValidationError(result)


def _validate_identifier(
    identifier: str,
    path: str,
    issues: list[DesignValidationIssue],
) -> None:
    if not identifier.strip():
        _issue(issues, "invalid_id", path, "ID must contain at least one non-whitespace character")


def _validate_unique_ids(
    identifiers: list[tuple[str, str]],
    issues: list[DesignValidationIssue],
) -> None:
    first_path_by_id: dict[str, str] = {}
    for identifier, path in identifiers:
        _validate_identifier(identifier, path, issues)
        first_path = first_path_by_id.setdefault(identifier, path)
        if first_path != path:
            _issue(
                issues,
                "duplicate_id",
                path,
                f"ID {identifier!r} duplicates {first_path}",
            )


def _validate_wall(
    wall: WallSegment,
    index: int,
    design: HouseDesign,
    valid_footprint: bool,
    issues: list[DesignValidationIssue],
) -> None:
    path = f"floor_plan.walls[{index}]"
    coordinates = {
        "start.x": wall.start.x,
        "start.y": wall.start.y,
        "end.x": wall.end.x,
        "end.y": wall.end.y,
    }
    for suffix, value in coordinates.items():
        _validate_finite(value, f"{path}.{suffix}", issues)
    _validate_positive(wall.thickness_m, f"{path}.thickness_m", issues)
    _validate_positive(wall.height_m, f"{path}.height_m", issues)

    if not all(isfinite(value) for value in coordinates.values()):
        return
    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y
    if abs(dx) <= GEOMETRY_EPSILON_M and abs(dy) <= GEOMETRY_EPSILON_M:
        _issue(issues, "wall_zero_length", path, "wall start and end must be different")
    elif abs(dx) > GEOMETRY_EPSILON_M and abs(dy) > GEOMETRY_EPSILON_M:
        _issue(issues, "wall_not_axis_aligned", path, "wall must be horizontal or vertical")

    if valid_footprint:
        half_width = design.footprint_width_m / 2
        half_depth = design.footprint_depth_m / 2
        for endpoint_name, x, y in (
            ("start", wall.start.x, wall.start.y),
            ("end", wall.end.x, wall.end.y),
        ):
            if (
                x < -half_width - GEOMETRY_EPSILON_M
                or x > half_width + GEOMETRY_EPSILON_M
                or y < -half_depth - GEOMETRY_EPSILON_M
                or y > half_depth + GEOMETRY_EPSILON_M
            ):
                _issue(
                    issues,
                    "wall_out_of_bounds",
                    f"{path}.{endpoint_name}",
                    "wall endpoint must lie inside the centered footprint bounds",
                )


def _validate_wall_relationships(
    walls: list[WallSegment],
    issues: list[DesignValidationIssue],
) -> None:
    for left_index, left in enumerate(walls):
        if not _is_axis_aligned_nonzero_finite(left):
            continue
        for right_index in range(left_index + 1, len(walls)):
            right = walls[right_index]
            if not _is_axis_aligned_nonzero_finite(right):
                continue
            relationship = _wall_relationship(left, right)
            if relationship is None:
                continue
            code, point = relationship
            path = f"floor_plan.walls[{right_index}]"
            if code == "wall_overlap":
                message = f"wall overlaps floor_plan.walls[{left_index}]"
            else:
                message = (
                    f"wall intersects floor_plan.walls[{left_index}] at "
                    f"({point[0]:g}, {point[1]:g}) without a shared endpoint"
                )
            _issue(issues, code, path, message)


def _wall_relationship(
    left: WallSegment,
    right: WallSegment,
) -> tuple[Literal["wall_overlap"], tuple[float, float]] | tuple[
    Literal["wall_intersection"], tuple[float, float]
] | None:
    left_horizontal = abs(left.start.y - left.end.y) <= GEOMETRY_EPSILON_M
    right_horizontal = abs(right.start.y - right.end.y) <= GEOMETRY_EPSILON_M
    if left_horizontal == right_horizontal:
        left_fixed = left.start.y if left_horizontal else left.start.x
        right_fixed = right.start.y if right_horizontal else right.start.x
        if abs(left_fixed - right_fixed) > GEOMETRY_EPSILON_M:
            return None
        left_min, left_max = _wall_axis_interval(left, left_horizontal)
        right_min, right_max = _wall_axis_interval(right, right_horizontal)
        overlap = min(left_max, right_max) - max(left_min, right_min)
        if overlap > GEOMETRY_EPSILON_M:
            return "wall_overlap", (0.0, 0.0)
        return None

    horizontal = left if left_horizontal else right
    vertical = right if left_horizontal else left
    horizontal_min, horizontal_max = sorted((horizontal.start.x, horizontal.end.x))
    vertical_min, vertical_max = sorted((vertical.start.y, vertical.end.y))
    point = (vertical.start.x, horizontal.start.y)
    if not (
        horizontal_min - GEOMETRY_EPSILON_M
        <= point[0]
        <= horizontal_max + GEOMETRY_EPSILON_M
        and vertical_min - GEOMETRY_EPSILON_M
        <= point[1]
        <= vertical_max + GEOMETRY_EPSILON_M
    ):
        return None
    if _is_endpoint(horizontal, point) and _is_endpoint(vertical, point):
        return None
    return "wall_intersection", point


def _validate_openings(
    design: HouseDesign,
    issues: list[DesignValidationIssue],
) -> None:
    walls_by_id: dict[str, WallSegment] = {}
    for wall in design.floor_plan.walls:
        walls_by_id.setdefault(wall.wall_id, wall)

    valid_spans_by_wall: dict[str, list[tuple[float, float, int]]] = {}
    for index, opening in enumerate(design.floor_plan.openings):
        path = f"floor_plan.openings[{index}]"
        _validate_nonnegative(opening.offset_m, f"{path}.offset_m", issues)
        _validate_positive(opening.width_m, f"{path}.width_m", issues)
        _validate_positive(opening.height_m, f"{path}.height_m", issues)
        _validate_nonnegative(opening.sill_height_m, f"{path}.sill_height_m", issues)

        referenced_wall = walls_by_id.get(opening.wall_id)
        if referenced_wall is None:
            _issue(
                issues,
                "opening_unknown_wall",
                f"{path}.wall_id",
                f"opening references unknown wall {opening.wall_id!r}",
            )
            continue
        if not _opening_geometry_is_finite(opening) or not _is_axis_aligned_nonzero_finite(
            referenced_wall
        ):
            continue

        wall_length = hypot(
            referenced_wall.end.x - referenced_wall.start.x,
            referenced_wall.end.y - referenced_wall.start.y,
        )
        span_start = opening.offset_m - opening.width_m / 2
        span_end = opening.offset_m + opening.width_m / 2
        if (
            span_start < -GEOMETRY_EPSILON_M
            or span_end > wall_length + GEOMETRY_EPSILON_M
        ):
            _issue(
                issues,
                "opening_out_of_bounds",
                path,
                (
                    f"opening span [{span_start:g}, {span_end:g}] must fit wall "
                    f"length {wall_length:g}"
                ),
            )
        else:
            valid_spans_by_wall.setdefault(opening.wall_id, []).append(
                (span_start, span_end, index)
            )
        if (
            opening.sill_height_m + opening.height_m
            > referenced_wall.height_m + GEOMETRY_EPSILON_M
        ):
            _issue(
                issues,
                "opening_vertical_overflow",
                path,
                (
                    f"opening top at {opening.sill_height_m + opening.height_m:g} m "
                    f"exceeds wall height {referenced_wall.height_m:g} m"
                ),
            )

    for wall_id, spans in valid_spans_by_wall.items():
        ordered = sorted(spans)
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right[0] < left[1] - GEOMETRY_EPSILON_M:
                _issue(
                    issues,
                    "opening_overlap",
                    f"floor_plan.openings[{right[2]}]",
                    (
                        f"opening overlaps floor_plan.openings[{left[2]}] "
                        f"on wall {wall_id!r}"
                    ),
                )


def _validate_finite(
    value: float,
    path: str,
    issues: list[DesignValidationIssue],
) -> None:
    if not isfinite(value):
        _issue(issues, "non_finite_dimension", path, "value must be finite")


def _validate_positive(
    value: float,
    path: str,
    issues: list[DesignValidationIssue],
) -> None:
    _validate_finite(value, path, issues)
    if isfinite(value) and value <= 0:
        _issue(issues, "non_positive_dimension", path, "value must be greater than zero")


def _validate_nonnegative(
    value: float,
    path: str,
    issues: list[DesignValidationIssue],
) -> None:
    _validate_finite(value, path, issues)
    if isfinite(value) and value < 0:
        _issue(issues, "non_positive_dimension", path, "value must be zero or greater")


def _is_positive_finite(value: float) -> bool:
    return isfinite(value) and value > 0


def _is_axis_aligned_nonzero_finite(wall: WallSegment) -> bool:
    values = (wall.start.x, wall.start.y, wall.end.x, wall.end.y)
    if not all(isfinite(value) for value in values):
        return False
    dx = abs(wall.end.x - wall.start.x)
    dy = abs(wall.end.y - wall.start.y)
    return (
        (dx <= GEOMETRY_EPSILON_M) != (dy <= GEOMETRY_EPSILON_M)
        and max(dx, dy) > GEOMETRY_EPSILON_M
    )


def _wall_axis_interval(wall: WallSegment, horizontal: bool) -> tuple[float, float]:
    values = (wall.start.x, wall.end.x) if horizontal else (wall.start.y, wall.end.y)
    return min(values), max(values)


def _is_endpoint(wall: WallSegment, point: tuple[float, float]) -> bool:
    return any(
        abs(endpoint_x - point[0]) <= GEOMETRY_EPSILON_M
        and abs(endpoint_y - point[1]) <= GEOMETRY_EPSILON_M
        for endpoint_x, endpoint_y in (
            (wall.start.x, wall.start.y),
            (wall.end.x, wall.end.y),
        )
    )


def _opening_geometry_is_finite(opening: Opening) -> bool:
    return all(
        isfinite(value)
        for value in (
            opening.offset_m,
            opening.width_m,
            opening.height_m,
            opening.sill_height_m,
        )
    )


def _issue(
    issues: list[DesignValidationIssue],
    code: DesignValidationCode,
    path: str,
    message: str,
) -> None:
    issues.append(DesignValidationIssue(code=code, path=path, message=message))
