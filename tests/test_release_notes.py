from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.check_release_notes import (
    EXPECTED_RELEASE_TITLE,
    ReleaseNotesError,
    canonical_release_notes,
    verify_release_notes,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def test_release_notes_accept_transport_newline_differences(tmp_path: Path) -> None:
    expected = tmp_path / "notes.md"
    expected.write_text("# Release\n\nEvidence and limitations.\n", encoding="utf-8")

    digest = verify_release_notes(
        expected,
        {
            "name": EXPECTED_RELEASE_TITLE,
            "body": "# Release\r\n\r\nEvidence and limitations.\r\n\r\n",
        },
    )

    canonical = canonical_release_notes(expected.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"name": "Wrong", "body": "notes"}, "title differs"),
        ({"name": EXPECTED_RELEASE_TITLE, "body": 1}, "body must be a string"),
        ({"name": EXPECTED_RELEASE_TITLE, "body": "tampered"}, "notes differ"),
        ([], "must be a JSON object"),
    ],
)
def test_release_notes_reject_mismatched_metadata(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    expected = tmp_path / "notes.md"
    expected.write_text("canonical\n", encoding="utf-8")

    with pytest.raises(ReleaseNotesError, match=message):
        verify_release_notes(expected, payload)


def test_release_workflow_rechecks_notes_and_non_prerelease_state() -> None:
    workflow = (
        WORKSPACE / ".github" / "workflows" / "release.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("--json isDraft,isPrerelease") == 2
    assert workflow.count("'.isPrerelease'") == 2
    assert "release_notes_sha256: ${{ steps.release_notes.outputs.release_notes_sha256 }}" in workflow
    assert "EXPECTED_RELEASE_NOTES_SHA256: ${{ needs.verify.outputs.release_notes_sha256 }}" in workflow
