"""Enforce unrounded line coverage independently of Coverage.py's combined gate."""

from __future__ import annotations

import argparse
import math
import re
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree


WORKSPACE = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LineCoverage:
    covered: int
    valid: int

    def __post_init__(self) -> None:
        if type(self.covered) is not int or type(self.valid) is not int:
            raise ValueError("line counters must be integers, not booleans or floats")
        if self.valid <= 0:
            raise ValueError("lines-valid must be greater than zero")
        if not 0 <= self.covered <= self.valid:
            raise ValueError("lines-covered must be between zero and lines-valid")

    @property
    def percent(self) -> float:
        # Form the bounded ratio first, avoiding float overflow on large counters.
        result = float(Fraction(100 * self.covered, self.valid))
        if not math.isfinite(result):
            raise ValueError("line coverage percentage must be finite")
        return result

    def passes(self, floor: Decimal) -> bool:
        if not floor.is_finite() or not 0 <= floor <= 100:
            raise ValueError("line coverage floor must be finite and between zero and 100")
        # Neither the displayed percentage nor XML's rounded line-rate decides the gate.
        return Fraction(100 * self.covered, self.valid) >= Fraction(floor)


def read_line_coverage(path: Path) -> LineCoverage:
    root = ElementTree.parse(path).getroot()
    if root.tag != "coverage":
        raise ValueError("coverage XML must have a coverage root element")

    counters: dict[str, int] = {}
    for name in ("lines-covered", "lines-valid"):
        raw = root.get(name)
        if raw is None or re.fullmatch(r"[0-9]+", raw) is None:
            raise ValueError(f"{name} must be a nonnegative integer counter")
        counters[name] = int(raw)
    return LineCoverage(covered=counters["lines-covered"], valid=counters["lines-valid"])


def read_line_floor(path: Path) -> Decimal:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    try:
        value = config["tool"]["construction_intelligence"]["coverage"]["line_fail_under"]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing tool.construction_intelligence.coverage.line_fail_under") from exc
    if type(value) not in (int, float):
        raise ValueError("line_fail_under must be a numeric percentage")
    floor = Decimal(str(value))
    if not floor.is_finite() or not 80 <= floor <= 100:
        raise ValueError("line_fail_under must be finite and between 80 and 100")
    return floor


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-xml", type=Path, default=WORKSPACE / "coverage.xml")
    parser.add_argument("--config", type=Path, default=WORKSPACE / "pyproject.toml")
    args = parser.parse_args(argv)
    try:
        coverage = read_line_coverage(args.coverage_xml)
        floor = read_line_floor(args.config)
        passed = coverage.passes(floor)
        percentage = coverage.percent
    except (OSError, ValueError, ElementTree.ParseError) as exc:
        print(f"line coverage check failed: {exc}", file=sys.stderr)
        return 2

    status = "PASS" if passed else "FAIL"
    print(
        f"Line coverage: {coverage.covered}/{coverage.valid} = {percentage:.4f}%; "
        f"required >= {floor}%: {status}. "
        "The combined statement/branch gate is enforced separately by pytest-cov."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
