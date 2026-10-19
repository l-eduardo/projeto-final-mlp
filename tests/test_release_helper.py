"""tests/test_release_helper.py - Unit tests for CD release helper.

Validates semantic versioning calculations, conventional commit categorization,
markdown release notes generation, and CLI dry-run execution.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add repository root / scripts to sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from release_helper import (  # noqa: E402
    CommitInfo,
    SemVer,
    compute_next_version,
    generate_release_notes,
    main,
    publish_gitlab_release,
)


class TestSemVer(unittest.TestCase):
    """Tests for SemVer data structure and bumping logic."""

    def test_parse_standard_versions(self):
        v = SemVer.parse("v0.1.0")
        self.assertEqual(v.major, 0)
        self.assertEqual(v.minor, 1)
        self.assertEqual(v.patch, 0)
        self.assertEqual(v.tag_name(), "v0.1.0")

        v2 = SemVer.parse("1.2.3")
        self.assertEqual(v2.major, 1)
        self.assertEqual(v2.minor, 2)
        self.assertEqual(v2.patch, 3)
        self.assertEqual(v2.tag_name(), "1.2.3")

    def test_parse_fallback_on_malformed(self):
        v = SemVer.parse("not-a-version")
        self.assertEqual(v.major, 0)
        self.assertEqual(v.minor, 1)
        self.assertEqual(v.patch, 0)

    def test_bump_major(self):
        v = SemVer(major=0, minor=2, patch=5, prefix="v")
        bumped = v.bump_major()
        self.assertEqual(bumped.tag_name(), "v1.0.0")

    def test_bump_minor(self):
        v = SemVer(major=0, minor=1, patch=3, prefix="v")
        bumped = v.bump_minor()
        self.assertEqual(bumped.tag_name(), "v0.2.0")

    def test_bump_patch(self):
        v = SemVer(major=1, minor=0, patch=0, prefix="v")
        bumped = v.bump_patch()
        self.assertEqual(bumped.tag_name(), "v1.0.1")


class TestCommitInfo(unittest.TestCase):
    """Tests for CommitInfo categorization based on Conventional Commits."""

    def test_categories(self):
        cases = [
            ("feat: add softmax layer", "feat"),
            ("feat(activations): add relu", "feat"),
            ("fix: prevent division by zero", "fix"),
            ("perf: optimize matrix multiplication", "perf"),
            ("refactor: split layers into modules", "refactor"),
            ("ci: add gitlab-ci pipeline", "ci"),
            ("docs: update readme with setup", "docs"),
            ("test: add smoke tests", "test"),
            ("chore: update .gitignore", "chore"),
            ("feat!: rewrite core contract", "breaking"),
            ("random commit message", "other"),
        ]
        for subject, expected_cat in cases:
            c = CommitInfo("abc1234", "Tester", subject, "")
            self.assertEqual(c.category, expected_cat, f"Failed for '{subject}'")

    def test_breaking_change_in_body(self):
        c = CommitInfo(
            commit_hash="def5678",
            author="Dev",
            subject="refactor: overhaul base class",
            body="BREAKING CHANGE: changed abstract method signature",
        )
        self.assertEqual(c.category, "breaking")


class TestVersionComputation(unittest.TestCase):
    """Tests for next version computation with mocked Git calls."""

    @patch("release_helper.get_latest_tag")
    @patch("release_helper.get_commits_since")
    def test_initial_release_without_tags(self, mock_commits, mock_tag):
        mock_tag.return_value = None
        mock_commits.return_value = []
        next_ver, prev_tag, commits = compute_next_version(
            Path("."), baseline="v0.1.0"
        )
        self.assertEqual(next_ver.tag_name(), "v0.1.0")
        self.assertIsNone(prev_tag)

    @patch("release_helper.get_latest_tag")
    @patch("release_helper.get_commits_since")
    def test_bump_minor_on_feature(self, mock_commits, mock_tag):
        mock_tag.return_value = "v0.1.0"
        mock_commits.return_value = [
            CommitInfo("111", "Dev", "feat: add dense layer", ""),
            CommitInfo("222", "Dev", "docs: update docs", ""),
        ]
        next_ver, prev_tag, _ = compute_next_version(Path("."))
        self.assertEqual(next_ver.tag_name(), "v0.2.0")
        self.assertEqual(prev_tag, "v0.1.0")

    @patch("release_helper.get_latest_tag")
    @patch("release_helper.get_commits_since")
    def test_bump_major_on_breaking(self, mock_commits, mock_tag):
        mock_tag.return_value = "v0.1.0"
        mock_commits.return_value = [
            CommitInfo("111", "Dev", "feat!: overhaul api", "")
        ]
        next_ver, _, _ = compute_next_version(Path("."))
        self.assertEqual(next_ver.tag_name(), "v1.0.0")

    @patch("release_helper.get_latest_tag")
    @patch("release_helper.get_commits_since")
    def test_bump_patch_on_fix(self, mock_commits, mock_tag):
        mock_tag.return_value = "v0.1.0"
        mock_commits.return_value = [
            CommitInfo("111", "Dev", "fix: correct bias shape", "")
        ]
        next_ver, _, _ = compute_next_version(Path("."))
        self.assertEqual(next_ver.tag_name(), "v0.1.1")


class TestReleaseNotes(unittest.TestCase):
    """Tests for Markdown release notes generation."""

    def test_generate_notes_with_categories(self):
        commits = [
            CommitInfo("aaa", "Author", "feat: initial MLP structure", ""),
            CommitInfo("bbb", "Author", "fix: resolve shape mismatch", ""),
            CommitInfo("ccc", "Author", "ci: setup gitlab pipeline", ""),
        ]
        notes = generate_release_notes(
            version=SemVer(0, 1, 0, "v"),
            latest_tag=None,
            commits=commits,
        )
        self.assertIn("# Release v0.1.0", notes)
        self.assertIn("### 🚀 Features & Enhancements", notes)
        self.assertIn("feat: initial MLP structure (`aaa`)", notes)
        self.assertIn("### 🐛 Bug Fixes", notes)
        self.assertIn("### 👷 CI/CD & Automation", notes)

    def test_generate_notes_empty_commits(self):
        notes = generate_release_notes(
            version=SemVer(0, 1, 0, "v"), latest_tag="v0.1.0", commits=[]
        )
        self.assertIn("No new commits recorded", notes)


class TestGitLabReleasePublishing(unittest.TestCase):
    """Tests for remote publishing and dry-run guard."""

    def test_dry_run_does_not_network(self):
        result = publish_gitlab_release("v0.1.0", "Notes", dry_run=True)
        self.assertTrue(result)

    def test_skip_when_missing_env(self):
        with patch.dict("os.environ", {}, clear=True):
            result = publish_gitlab_release("v0.1.0", "Notes", dry_run=False)
            self.assertFalse(result)


class TestCLIExecution(unittest.TestCase):
    """Tests CLI execution in dry-run mode and file output."""

    def test_dry_run_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            notes_file = Path(tmpdir) / "notes.md"
            env_file = Path(tmpdir) / "release.env"
            exit_code = main(
                [
                    "--dry-run",
                    "--baseline",
                    "v0.1.0",
                    "--notes-file",
                    str(notes_file),
                    "--env-file",
                    str(env_file),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(notes_file.exists())
            self.assertTrue(env_file.exists())
            env_content = env_file.read_text()
            self.assertIn("RELEASE_VERSION=v0.1.0", env_content)


if __name__ == "__main__":
    unittest.main()
