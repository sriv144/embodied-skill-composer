from __future__ import annotations

import pytest

from scripts.resolve_construction_pages import resolve_pages_channel


def test_pages_preview_is_available_only_before_publication() -> None:
    assert resolve_pages_channel(None) == "preview"
    draft = {"tag_name": "v1.0.0", "draft": True, "prerelease": False}
    assert resolve_pages_channel(draft) == "preview"
    assert resolve_pages_channel(draft, requested="release", allow_draft=True) == "release"
    draft["draft"] = False
    assert resolve_pages_channel(draft) == "release"
    assert resolve_pages_channel(draft, requested="release") == "release"


@pytest.mark.parametrize(
    "release",
    [
        {},
        [],
        {"tag_name": "v0.9.0", "draft": False, "prerelease": False},
        {"tag_name": "v1.0.0", "draft": "false", "prerelease": False},
        {"tag_name": "v1.0.0", "draft": False},
        {"tag_name": "v1.0.0", "draft": False, "prerelease": True},
    ],
)
def test_pages_never_falls_back_for_invalid_release_state(release: object) -> None:
    with pytest.raises(ValueError):
        resolve_pages_channel(release)


@pytest.mark.parametrize(
    "release",
    [None, {"tag_name": "v1.0.0", "draft": True, "prerelease": False}],
)
def test_explicit_canonical_pages_cannot_fall_back_to_preview(release: object) -> None:
    with pytest.raises(ValueError, match="available published"):
        resolve_pages_channel(release, requested="release")
