from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
BINARY_SUFFIXES = {
    ".avi",
    ".blend",
    ".ckpt",
    ".gif",
    ".glb",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".onnx",
    ".png",
    ".pt",
    ".pyc",
    ".ttt",
}
PATTERNS = {
    "OpenAI API key": re.compile(r"\bsk-" + r"(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def candidate_paths(*, staged: bool) -> list[Path]:
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
        if staged
        else ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    )
    result = subprocess.run(
        command,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        check=True,
    )
    return [WORKSPACE / item for item in result.stdout.splitlines() if item]


def scan(*, staged: bool) -> list[tuple[Path, int, str]]:
    findings: list[tuple[Path, int, str]] = []
    for path in candidate_paths(staged=staged):
        if not path.is_file() or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((path.relative_to(WORKSPACE), line_number, label))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan repository text without printing secrets.")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    findings = scan(staged=args.staged)
    if findings:
        print("Potential secrets detected:")
        for path, line_number, label in findings:
            print(f"- {path}:{line_number}: {label}")
        return 1
    scope = "staged changes" if args.staged else "tracked and untracked repository text"
    print(f"Secret scan passed for {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
