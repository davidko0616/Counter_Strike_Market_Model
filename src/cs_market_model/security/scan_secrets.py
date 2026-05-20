"""Lightweight repository secret scanner.

This is intentionally dependency-free so it can run before the project environment
is fully installed. It catches the main failure modes for this repo: committed .env
files, concrete API key assignments, bearer tokens, and long high-entropy literals.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".env",
    ".example",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "reports",
    "venv",
}

PATTERNS = [
    (
        "concrete API key assignment",
        re.compile(
            r"""(?ix)
            (api[_-]?key|token|secret|authorization)
            \s*[:=]\s*
            ["']?
            (?!<|your_|replace_|paste_|$)
            [A-Za-z0-9_\-]{16,}
            ["']?
            """
        ),
    ),
    (
        "bearer token literal",
        re.compile(r"""(?i)\bBearer\s+[A-Za-z0-9_\-.=]{16,}"""),
    ),
    (
        "CSFloat-like token",
        re.compile(r"""\b(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9]{6,}_[A-Za-z0-9_-]{16,}\b"""),
    ),
]


@dataclass(frozen=True)
class Finding:
    path: Path
    line_number: int
    kind: str

    def format(self, root: Path) -> str:
        relative = self.path.relative_to(root)
        return f"{relative}:{self.line_number}: {self.kind}"


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    return -sum((count / len(value)) * math.log2(count / len(value)) for count in counts.values())


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith(".env")


def iter_candidate_files(root: Path) -> list[Path]:
    tracked = iter_git_tracked_files(root)
    if tracked is not None:
        return [path for path in tracked if is_text_candidate(path)]

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = set(path.relative_to(root).parts)
        if relative_parts & SKIP_DIRS:
            continue
        if path.name == ".env" or path.name.startswith(".env."):
            if path.name != ".env.example":
                continue
            candidates.append(path)
            continue
        if is_text_candidate(path):
            candidates.append(path)
    return sorted(candidates)


def iter_git_tracked_files(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    paths = []
    for line in result.stdout.splitlines():
        path = root / line
        if path.is_file():
            paths.append(path)
    return sorted(paths)


def scan_line(path: Path, line_number: int, line: str) -> list[Finding]:
    findings: list[Finding] = []
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        findings.append(Finding(path, line_number, "local env file must not be committed"))
        return findings

    for kind, pattern in PATTERNS:
        if pattern.search(line):
            findings.append(Finding(path, line_number, kind))

    for match in re.finditer(r"""["']([A-Za-z0-9_\-+/=]{32,})["']""", line):
        candidate = match.group(1)
        if shannon_entropy(candidate) >= 4.2:
            findings.append(Finding(path, line_number, "high-entropy string literal"))

    return findings


def scan_repo(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_candidate_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            findings.extend(scan_line(path, line_number, line))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan repository files for likely secrets.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()

    root = args.root.resolve()
    findings = scan_repo(root)
    if findings:
        print("Potential secrets found:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding.format(root)}", file=sys.stderr)
        return 1

    print("No likely committed secrets found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
