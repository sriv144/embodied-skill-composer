from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _jobs(name: str) -> dict[str, Any]:
    workflow = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return dict(workflow["jobs"])


def test_release_publishes_only_after_authorized_synchronous_pages_smoke() -> None:
    jobs = _jobs("release.yml")
    approval = jobs["authorize_deployment"]
    assert approval["environment"] == "construction-intelligence-v1-release"
    assert set(approval["needs"]) == {"verify", "canonical_pages"}

    deployment = jobs["deploy_canonical_pages"]
    assert set(deployment["needs"]) == {"verify", "authorize_deployment"}
    assert deployment["uses"] == "./.github/workflows/pages.yml"
    assert deployment["with"] == {
        "channel": "release",
        "deploy": True,
        "allow_draft": True,
        "expected_asset_set_sha256": "${{ needs.verify.outputs.asset_set_sha256 }}",
    }
    assert deployment["permissions"]["contents"] == "write"  # Draft visibility needs push access.

    publish = jobs["publish"]
    assert set(publish["needs"]) == {"verify", "deploy_canonical_pages"}
    assert "if" not in publish  # No always()/failure() bypass of the deployment gate.
    assert publish["permissions"] == {"contents": "write"}
    steps = publish["steps"]
    assert "Revalidate protected main" in steps[0]["name"]
    assert steps[-2]["name"] == "Recheck deployed identity immediately before publication"
    assert "sha256sum --check expected.sha256" in steps[-2]["run"]
    assert steps[-2]["env"]["EXPECTED_DEMO_IDENTITY"] == (
        "${{ needs.deploy_canonical_pages.outputs.demo_identity }}"
    )
    assert "gh release edit" in steps[-1]["run"]
    assert "--draft=false" in steps[-1]["run"]
    assert all("gh workflow run" not in step.get("run", "") for step in steps)


def test_draft_pages_calls_require_bound_assets_and_fixed_tag_for_deployment() -> None:
    jobs = _jobs("pages.yml")
    steps = jobs["build"]["steps"]
    guard = next(step for step in steps if step.get("if") == "inputs.allow_draft == true")
    assert guard["env"]["EXPECTED_ASSET_SET_SHA256"] == (
        "${{ inputs.expected_asset_set_sha256 }}"
    )
    assert 'test "$REQUESTED_CHANNEL" = release' in guard["run"]
    assert "^[0-9a-f]{64}$" in guard["run"]
    assert 'if [ "$DEPLOY_DRAFT" = true ]' in guard["run"]
    assert 'test "$GITHUB_REF" = refs/tags/v1.0.0' in guard["run"]
    assert steps.index(guard) < next(
        index for index, step in enumerate(steps) if step.get("id") == "channel"
    )
    for step in steps:
        if "ALLOW_DRAFT" in step.get("env", {}):
            assert "expected_asset_set_sha256 != ''" in step["env"]["ALLOW_DRAFT"]
    diagnostics = next(step for step in steps if step.get("name") == "Upload browser diagnostics")
    assert diagnostics["with"]["name"] == (
        "pages-canonical-browser-diagnostics-${{ inputs.deploy == false && 'verify' || 'deploy' }}"
    )  # Reusable preflight/deploy jobs cannot upload the same immutable artifact name.

    deployment = jobs["deploy"]
    assert deployment["needs"] == "build"
    assert deployment["timeout-minutes"] == 15
    assert deployment["if"] == "inputs.deploy != false || inputs.channel == ''"
    smoke = deployment["steps"][-1]
    assert smoke["name"] == "Smoke deployed workbench"
    assert "sha256sum --check expected.sha256" in smoke["run"]
    assert smoke["run"].count("--connect-timeout 10 --max-time 20 --retry 2") == 2
    assert "exit 1" in smoke["run"]
    assert "continue-on-error" not in smoke


def test_draft_preflight_remains_non_mutating_and_never_deploys() -> None:
    preflight = _jobs("release.yml")["canonical_pages"]
    assert preflight["with"]["deploy"] is False
    assert preflight["with"]["allow_draft"] is True
    assert preflight["with"]["expected_asset_set_sha256"] == (
        "${{ needs.verify.outputs.asset_set_sha256 }}"
    )


def test_draft_lookup_uses_cli_draft_resolution_and_visible_token_scope() -> None:
    release_jobs = _jobs("release.yml")
    assert release_jobs["verify"]["permissions"]["contents"] == "write"
    assert release_jobs["canonical_pages"]["permissions"]["contents"] == "write"
    pages_jobs = _jobs("pages.yml")
    assert pages_jobs["build"]["permissions"]["contents"] == "write"
    assert pages_jobs["deploy"]["permissions"]["contents"] == "read"
    steps = pages_jobs["build"]["steps"]
    resolution = next(step for step in steps if step.get("id") == "channel")
    assert 'if [ "$ALLOW_DRAFT" = true ]; then' in resolution["run"]
    draft_lookup = resolution["run"].split("elif gh api", 1)[0]
    assert "gh release view v1.0.0 --json tagName,isDraft,isPrerelease" in draft_lookup
    assert "releases/tags" not in draft_lookup
    assert "tag_name: .tagName, draft: .isDraft, prerelease: .isPrerelease" in draft_lookup
    staging = next(step for step in steps if step.get("name", "").startswith("Download and stage"))
    assert "gh release view v1.0.0 --json tagName,isDraft,isPrerelease" in staging["run"]


def test_pages_exposes_the_exact_identity_that_publication_rechecks() -> None:
    workflow = yaml.safe_load((WORKFLOWS / "pages.yml").read_text(encoding="utf-8"))
    # YAML 1.1 treats the unquoted GitHub Actions key 'on' as True.
    outputs = workflow.get("on", workflow.get(True))["workflow_call"]["outputs"]
    assert outputs["demo_identity"]["value"] == "${{ jobs.build.outputs.demo_identity }}"
    assert outputs["page_url"]["value"] == "${{ jobs.deploy.outputs.page_url }}"
    assert workflow["jobs"]["deploy"]["outputs"]["page_url"] == (
        "${{ steps.deployment.outputs.page_url }}"
    )
