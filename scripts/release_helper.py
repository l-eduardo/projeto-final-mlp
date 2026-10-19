#!/usr/bin/env python3
"""scripts/release_helper.py - Automated SemVer Tagging & GitLab Release Helper.

Automates semantic version calculation, changelog generation from conventional
commit history, git tagging, and GitLab release publishing.

Key capabilities:
1. Calculates next semantic version (Major.Minor.Patch) from Git history and tags,
   adhering to Conventional Commits specification (baseline: v0.1.0).
2. Generates organized Markdown release notes categorizing features, bug fixes,
   infrastructure/CI, and maintenance tasks.
3. Supports local dry-run inspection, Git tag creation, and remote GitLab Release
   publishing via standard library HTTP (REST API v4) or release-cli.
4. Operates using Python standard library alone (zero pip dependencies required).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclasses.dataclass
class SemVer:
    """Represents a Semantic Version (Major.Minor.Patch)."""

    major: int = 0
    minor: int = 1
    patch: int = 0
    prefix: str = "v"

    @classmethod
    def parse(cls, version_str: str) -> "SemVer":
        clean = version_str.strip()
        prefix = "v" if clean.startswith("v") else ""
        if prefix:
            clean = clean[1:]
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", clean)
        if not match:
            return cls(major=0, minor=1, patch=0, prefix=prefix)
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prefix=prefix,
        )

    def bump_major(self) -> "SemVer":
        return SemVer(
            major=self.major + 1, minor=0, patch=0, prefix=self.prefix
        )

    def bump_minor(self) -> "SemVer":
        return SemVer(
            major=self.major, minor=self.minor + 1, patch=0, prefix=self.prefix
        )

    def bump_patch(self) -> "SemVer":
        return SemVer(
            major=self.major, minor=self.minor, patch=self.patch + 1, prefix=self.prefix
        )

    def tag_name(self) -> str:
        return f"{self.prefix}{self.major}.{self.minor}.{self.patch}"

    def __str__(self) -> str:
        return self.tag_name()


@dataclasses.dataclass
class CommitInfo:
    """Metadata of a parsed git commit."""

    commit_hash: str
    author: str
    subject: str
    body: str

    @property
    def category(self) -> str:
        sub = self.subject.strip().lower()
        is_breaking = (
            bool(re.match(r"^[a-zA-Z]+(\([^\)]+\))?!:", sub))
            or "breaking change" in (sub + self.body).lower()
            or "breaking-change" in (sub + self.body).lower()
        )
        if is_breaking:
            return "breaking"
        if sub.startswith("feat"):
            return "feat"
        if sub.startswith("fix"):
            return "fix"
        if sub.startswith("perf"):
            return "perf"
        if sub.startswith("refactor"):
            return "refactor"
        if sub.startswith("ci"):
            return "ci"
        if sub.startswith("docs"):
            return "docs"
        if sub.startswith("test"):
            return "test"
        if sub.startswith("chore"):
            return "chore"
        return "other"


def run_git(
    args: Sequence[str], repo_dir: Path, check: bool = False
) -> Tuple[int, str, str]:
    """Executes a git command in the target repository directory."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=check,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def get_latest_tag(repo_dir: Path) -> Optional[str]:
    """Retrieves the latest git tag matching v* or None."""
    code, out, _ = run_git(["tag", "--list", "v*", "--sort=-v:refname"], repo_dir)
    if code == 0 and out:
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if lines:
            return lines[0]
    # Fallback to git describe
    code, out, _ = run_git(["describe", "--tags", "--abbrev=0"], repo_dir)
    if code == 0 and out.strip():
        return out.strip()
    return None


def get_version_from_pyproject(repo_dir: Path) -> Optional[str]:
    """Extracts project version from pyproject.toml if present."""
    pyproject_file = repo_dir / "pyproject.toml"
    if not pyproject_file.exists():
        return None
    try:
        content = pyproject_file.read_text(encoding="utf-8")
        match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', content, re.M)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return None


def get_commits_since(
    repo_dir: Path, tag: Optional[str] = None
) -> List[CommitInfo]:
    """Retrieves list of commits since tag (or all commits if tag is None)."""
    rev_range = f"{tag}..HEAD" if tag else "HEAD"
    delimiter = "---END_OF_COMMIT---"
    cmd = [
        "log",
        rev_range,
        f"--pretty=format:%h%x1f%an%x1f%s%x1f%b%x1e{delimiter}",
    ]
    code, out, _ = run_git(cmd, repo_dir)
    if code != 0 or not out:
        return []

    commits: List[CommitInfo] = []
    raw_blocks = out.split(f"\x1e{delimiter}")
    for block in raw_blocks:
        trimmed = block.strip()
        if not trimmed:
            continue
        fields = trimmed.split("\x1f")
        if len(fields) >= 3:
            commit_hash = fields[0].strip()
            author = fields[1].strip()
            subject = fields[2].strip()
            body = fields[3].strip() if len(fields) > 3 else ""
            if subject:
                commits.append(
                    CommitInfo(
                        commit_hash=commit_hash,
                        author=author,
                        subject=subject,
                        body=body,
                    )
                )
    return commits


def compute_next_version(
    repo_dir: Path, baseline: str = "v0.1.0"
) -> Tuple[SemVer, Optional[str], List[CommitInfo]]:
    """Computes the next SemVer version based on git history."""
    latest_tag = get_latest_tag(repo_dir)
    commits = get_commits_since(repo_dir, latest_tag)

    if latest_tag is None:
        # Initial release
        pyproj_version = get_version_from_pyproject(repo_dir)
        if pyproj_version and baseline.startswith("v") and not pyproj_version.startswith("v"):
            initial_str = f"v{pyproj_version}"
        else:
            initial_str = pyproj_version or baseline
        return SemVer.parse(initial_str), None, commits

    current_semver = SemVer.parse(latest_tag)
    if not commits:
        return current_semver, latest_tag, commits

    # Analyze commit types to determine SemVer increment
    has_breaking = any(
        c.category == "breaking"
        or bool(re.match(r"^[a-zA-Z]+(\([^\)]+\))?!:", c.subject.strip().lower()))
        or "breaking change" in (c.subject + c.body).lower()
        or "breaking-change" in (c.subject + c.body).lower()
        for c in commits
    )
    has_feature = any(c.category == "feat" for c in commits)

    if has_breaking:
        next_ver = current_semver.bump_major()
    elif has_feature:
        next_ver = current_semver.bump_minor()
    else:
        next_ver = current_semver.bump_patch()

    return next_ver, latest_tag, commits


def generate_release_notes(
    version: SemVer,
    latest_tag: Optional[str],
    commits: List[CommitInfo],
    project_title: str = "MLP - Multi-Layer Perceptron (UFSC)",
) -> str:
    """Generates structured Markdown release notes."""
    today = datetime.date.today().isoformat()
    lines: List[str] = [
        f"# Release {version} ({today})",
        "",
        f"Automated release for **{project_title}**.",
        "",
    ]

    if not commits:
        lines.append("No new commits recorded since previous release.")
        return "\n".join(lines)

    category_labels: Dict[str, Tuple[str, str]] = {
        "breaking": ("⚠️ Breaking Changes", "breaking"),
        "feat": ("🚀 Features & Enhancements", "feat"),
        "fix": ("🐛 Bug Fixes", "fix"),
        "perf": ("⚡ Performance Improvements", "perf"),
        "refactor": ("♻️ Refactoring & Code Quality", "refactor"),
        "ci": ("👷 CI/CD & Automation", "ci"),
        "docs": ("📚 Documentation", "docs"),
        "test": ("🧪 Tests & Verification", "test"),
        "chore": ("🔧 Maintenance & Dependencies", "chore"),
        "other": ("📦 Other Changes", "other"),
    }

    categorized: Dict[str, List[CommitInfo]] = {key: [] for key in category_labels}
    for c in commits:
        cat = c.category if c.category in categorized else "other"
        categorized[cat].append(c)

    lines.append("## What's Changed")
    lines.append("")

    for cat_key, (label, _) in category_labels.items():
        items = categorized[cat_key]
        if items:
            lines.append(f"### {label}")
            for item in items:
                lines.append(f"- {item.subject} (`{item.commit_hash}`)")
            lines.append("")

    if latest_tag:
        lines.append(
            f"**Full Diff**: Comparing `{latest_tag}` to `{version}`"
        )
    else:
        lines.append(f"**Initial Release**: Version `{version}`")

    return "\n".join(lines)


def create_git_tag(
    repo_dir: Path, tag: str, message: str, dry_run: bool = False
) -> bool:
    """Creates an annotated Git tag locally."""
    if dry_run:
        print(f"[DRY-RUN] Would create git tag: git tag -a {tag} -m '{message}'")
        return True

    code, _, stderr = run_git(["tag", "-a", tag, "-m", message], repo_dir)
    if code != 0:
        print(f"[ERROR] Failed to create git tag '{tag}': {stderr}", file=sys.stderr)
        return False
    print(f"[SUCCESS] Git tag '{tag}' created successfully.")
    return True


def push_git_tag(repo_dir: Path, tag: str, dry_run: bool = False) -> bool:
    """Pushes a Git tag to the remote repository."""
    if dry_run:
        print(f"[DRY-RUN] Would push git tag: git push origin {tag}")
        return True

    code, _, stderr = run_git(["push", "origin", tag], repo_dir)
    if code != 0:
        print(f"[ERROR] Failed to push git tag '{tag}': {stderr}", file=sys.stderr)
        return False
    print(f"[SUCCESS] Git tag '{tag}' pushed successfully.")
    return True


def publish_gitlab_release(
    version: str,
    notes: str,
    ref_sha: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Publishes a GitLab release via REST API v4 using standard library."""
    api_url = os.environ.get("CI_API_V4_URL")
    project_id = os.environ.get("CI_PROJECT_ID")
    job_token = os.environ.get("CI_JOB_TOKEN")
    private_token = os.environ.get("GITLAB_TOKEN")
    ref = ref_sha or os.environ.get("CI_COMMIT_SHA") or "HEAD"

    payload = {
        "name": f"Release {version}",
        "tag_name": version,
        "description": notes,
        "ref": ref,
    }

    if dry_run:
        print("[DRY-RUN] Would publish GitLab release with payload:")
        print(json.dumps(payload, indent=2))
        return True

    if not api_url or not project_id or (not job_token and not private_token):
        print(
            "[INFO] GitLab CI environment variables (CI_API_V4_URL, CI_PROJECT_ID, "
            "CI_JOB_TOKEN/GITLAB_TOKEN) not detected. Skipping remote REST API publish."
        )
        return False

    endpoint = f"{api_url}/projects/{project_id}/releases"
    headers = {"Content-Type": "application/json"}
    if job_token:
        headers["JOB-TOKEN"] = job_token
    elif private_token:
        headers["PRIVATE-TOKEN"] = private_token

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=req_data, headers=headers, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status in (200, 201):
                print(f"[SUCCESS] GitLab Release '{version}' created successfully via API.")
                return True
            print(f"[WARNING] GitLab API returned unexpected status code: {resp.status}")
            return False
    except urllib.error.HTTPError as err:
        if err.code == 409:
            print(f"[INFO] Release '{version}' already exists in GitLab (HTTP 409 Conflict).")
            return True
        err_msg = err.read().decode("utf-8", errors="replace")
        print(f"[ERROR] GitLab API HTTP error ({err.code}): {err_msg}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[ERROR] Failed to publish GitLab release via API: {exc}", file=sys.stderr)
        return False


def write_file(path: Path, content: str) -> None:
    """Writes text content to file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automated SemVer Tagging & GitLab Release Helper for CI/CD"
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("."),
        help="Path to Git repository root (default: current directory)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="v0.1.0",
        help="Baseline SemVer tag when no tags exist (default: v0.1.0)",
    )
    parser.add_argument(
        "--version-override",
        type=str,
        default=None,
        help="Explicitly override computed SemVer tag",
    )
    parser.add_argument(
        "--notes-file",
        type=Path,
        default=None,
        help="File path to write generated markdown release notes",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="File path to write environment variables (dotenv format)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate version computation and release without making changes",
    )
    parser.add_argument(
        "--create-tag",
        action="store_true",
        help="Create annotated git tag locally",
    )
    parser.add_argument(
        "--push-tag",
        action="store_true",
        help="Push created git tag to origin remote",
    )
    parser.add_argument(
        "--gitlab-release",
        action="store_true",
        help="Publish release to GitLab via API or environment",
    )

    args = parser.parse_args(argv)
    repo_dir = args.repo_dir.resolve()

    computed_ver, latest_tag, commits = compute_next_version(
        repo_dir, baseline=args.baseline
    )
    version = (
        SemVer.parse(args.version_override)
        if args.version_override
        else computed_ver
    )
    tag_str = version.tag_name()

    print("==================================================")
    print(f" GitLab Auto-Release Helper: {tag_str}")
    print("==================================================")
    print(f"Previous Tag:      {latest_tag or 'None (initial release)'}")
    print(f"Computed Version:  {tag_str}")
    print(f"Commits Analyzed:  {len(commits)}")
    print("--------------------------------------------------")

    notes = generate_release_notes(version, latest_tag, commits)

    if args.notes_file:
        write_file(args.notes_file, notes)
        print(f"[INFO] Release notes written to: {args.notes_file}")

    if args.env_file:
        env_content = (
            f"RELEASE_VERSION={tag_str}\n"
            f"RELEASE_TAG={tag_str}\n"
            f"RELEASE_NAME=Release {tag_str}\n"
        )
        write_file(args.env_file, env_content)
        print(f"[INFO] Environment variables written to: {args.env_file}")

    if args.create_tag:
        create_git_tag(
            repo_dir, tag_str, f"Release {tag_str}", dry_run=args.dry_run
        )

    if args.push_tag:
        push_git_tag(repo_dir, tag_str, dry_run=args.dry_run)

    if args.gitlab_release:
        publish_gitlab_release(tag_str, notes, dry_run=args.dry_run)

    if args.dry_run or not (args.create_tag or args.push_tag or args.gitlab_release):
        print("\n--- Generated Release Notes Preview ---")
        print(notes)
        print("---------------------------------------")

    return 0


if __name__ == "__main__":
    sys.exit(main())
