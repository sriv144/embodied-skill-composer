from __future__ import annotations

import tomllib
from decimal import Decimal
from pathlib import Path

import pytest

from scripts import check_line_coverage as gate


def _xml(tmp_path: Path, attributes: str) -> Path:
    path = tmp_path / "coverage.xml"
    path.write_text(f"<coverage {attributes}/>", encoding="utf-8")
    return path


def _config(tmp_path: Path, value: str = "87.85") -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f"[tool.construction_intelligence.coverage]\nline_fail_under = {value}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("covered,passed", [(8784, False), (8785, True), (8786, True)])
def test_line_floor_is_independent_of_a_claimed_combined_or_rounded_rate(
    tmp_path: Path, covered: int, passed: bool
) -> None:
    path = _xml(tmp_path, f'lines-covered="{covered}" lines-valid="10000" line-rate="1"')
    coverage = gate.read_line_coverage(path)
    assert coverage.passes(gate.read_line_floor(_config(tmp_path))) is passed


def test_unrounded_integer_ratio_controls_the_gate() -> None:
    below = gate.LineCoverage(covered=87_849_999, valid=100_000_000)
    assert round(below.percent, 2) == 87.85
    assert not below.passes(Decimal("87.85"))
    assert gate.LineCoverage(covered=87_850_000, valid=100_000_000).passes(Decimal("87.85"))


def test_preserved_phase2_line_counts_are_not_the_combined_baseline() -> None:
    coverage = gate.LineCoverage(covered=7942, valid=9465)
    assert coverage.percent == pytest.approx(83.90913893291072)
    assert not coverage.passes(Decimal("83.91"))
    assert coverage.passes(Decimal("83.9091389329107"))


def test_current_verified_line_baseline_is_stronger_than_the_historical_baseline() -> None:
    coverage = gate.LineCoverage(covered=16033, valid=18252)
    assert coverage.percent == pytest.approx(87.84242822704361)
    assert coverage.passes(Decimal("83.91"))
    # The canonical floor conservatively rounds the observed baseline upward.
    assert not coverage.passes(Decimal("87.85"))


@pytest.mark.parametrize("value", ["-1", "+1", "1.0", "1e2", "nan", "inf", "True", " 1", "", "١"])
@pytest.mark.parametrize("attribute", ["lines-covered", "lines-valid"])
def test_xml_rejects_non_integer_counter_strings(tmp_path: Path, value: str, attribute: str) -> None:
    counters = {"lines-covered": "1", "lines-valid": "2", attribute: value}
    path = _xml(tmp_path, " ".join(f'{key}="{item}"' for key, item in counters.items()))
    with pytest.raises(ValueError, match="integer counter"):
        gate.read_line_coverage(path)


@pytest.mark.parametrize("attributes", ['', 'lines-covered="1"', 'lines-valid="1"'])
def test_xml_requires_both_counters(tmp_path: Path, attributes: str) -> None:
    with pytest.raises(ValueError, match="integer counter"):
        gate.read_line_coverage(_xml(tmp_path, attributes))


@pytest.mark.parametrize("covered,valid", [(0, 0), (1, 0), (2, 1), (-1, 1), (1, -1)])
def test_counters_require_a_valid_positive_denominator_and_covered_range(covered, valid) -> None:
    with pytest.raises(ValueError):
        gate.LineCoverage(covered=covered, valid=valid)


@pytest.mark.parametrize("covered,valid", [(True, 1), (1, False), (1.0, 2), (1, 2.0)])
def test_programmatic_counters_are_strict_integers(covered, valid) -> None:
    with pytest.raises(ValueError, match="integers"):
        gate.LineCoverage(covered=covered, valid=valid)


def test_large_valid_counters_produce_a_finite_percentage_without_float_overflow() -> None:
    coverage = gate.LineCoverage(covered=10**400, valid=2 * 10**400)
    assert coverage.percent == 50
    assert not coverage.passes(Decimal("87.85"))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "79.99", "100.01", '"83.91"', "true"])
def test_configuration_rejects_non_finite_non_numeric_and_out_of_range_floors(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError):
        gate.read_line_floor(_config(tmp_path, value))


def test_configuration_requires_the_canonical_line_floor_key(tmp_path: Path) -> None:
    path = tmp_path / "empty.toml"
    path.write_text("[tool.coverage.report]\nfail_under=82.19\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line_fail_under"):
        gate.read_line_floor(path)


@pytest.mark.parametrize("floor", ["NaN", "Infinity", "-Infinity", "-1", "101"])
def test_programmatic_floor_must_be_finite_and_bounded(floor: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        gate.LineCoverage(covered=1, valid=1).passes(Decimal(floor))


@pytest.mark.parametrize("covered,expected", [(8785, 0), (8784, 1)])
def test_cli_reports_line_coverage_separately_and_exits_for_below_floor(
    tmp_path: Path, capsys, covered: int, expected: int
) -> None:
    path = _xml(tmp_path, f'lines-covered="{covered}" lines-valid="10000"')
    result = gate.main(["--coverage-xml", str(path), "--config", str(_config(tmp_path))])
    assert result == expected
    output = capsys.readouterr()
    assert "Line coverage:" in output.out
    assert "required >= 87.85%" in output.out
    assert "combined statement/branch gate is enforced separately" in output.out
    assert not output.err


@pytest.mark.parametrize("contents", ["not XML", "<other/>", '<coverage lines-covered="1" lines-valid="0"/>'])
def test_cli_returns_invalid_input_status_with_a_clear_error(tmp_path: Path, capsys, contents) -> None:
    path = tmp_path / "coverage.xml"
    path.write_text(contents, encoding="utf-8")
    result = gate.main(["--coverage-xml", str(path), "--config", str(_config(tmp_path))])
    assert result == 2
    assert "line coverage check failed:" in capsys.readouterr().err


def test_cli_rejects_missing_xml(tmp_path: Path, capsys) -> None:
    result = gate.main([
        "--coverage-xml", str(tmp_path / "missing.xml"), "--config", str(_config(tmp_path)),
    ])
    assert result == 2
    assert "line coverage check failed:" in capsys.readouterr().err


def test_repository_keeps_both_independent_floor_values() -> None:
    path = gate.WORKSPACE / "pyproject.toml"
    assert gate.read_line_floor(path) == Decimal("87.85")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    assert config["tool"]["coverage"]["run"]["branch"] is True
    assert config["tool"]["coverage"]["report"]["fail_under"] == 82.19
    baseline = config["tool"]["construction_intelligence"]["coverage"]
    assert baseline["baseline_source_commit"] == "bd06db9a0443d4719be8d4aca34db882128d617e"
    assert baseline["baseline_workflow_run"] == 38111292762
    assert baseline["baseline_lines_covered"] == 16033
    assert baseline["baseline_lines_valid"] == 18252
