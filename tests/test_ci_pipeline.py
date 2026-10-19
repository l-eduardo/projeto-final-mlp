"""tests/test_ci_pipeline.py - Unit tests for GitLab CI/CD pipeline configuration.

Verifies that .gitlab-ci.yml is valid YAML and adheres to all project specifications:
- Correct stages (lint, audit, test, release).
- Default container image (python:3.10-slim).
- Cache configuration keyed to pyproject.toml.
- AST audit job with zero pip dependencies.
- Release job restricted strictly to the main branch.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
CI_CONFIG_PATH = REPO_DIR / ".gitlab-ci.yml"


class TestGitLabCIPipeline(unittest.TestCase):
    """Test suite validating .gitlab-ci.yml configuration integrity."""

    @classmethod
    def setUpClass(cls):
        cls.assertTrue(
            CI_CONFIG_PATH.exists(),
            f"Expected .gitlab-ci.yml to exist at {CI_CONFIG_PATH}",
        )
        content = CI_CONFIG_PATH.read_text(encoding="utf-8")
        cls.config = yaml.safe_load(content)

    def test_pipeline_stages(self):
        expected_stages = ["lint", "audit", "test", "release"]
        self.assertEqual(self.config.get("stages"), expected_stages)

    def test_default_image(self):
        default = self.config.get("default", {})
        self.assertEqual(default.get("image"), "python:3.10-slim")

    def test_pip_cache_configuration(self):
        variables = self.config.get("variables", {})
        self.assertIn("PIP_CACHE_DIR", variables)
        self.assertEqual(variables["PIP_CACHE_DIR"], "$CI_PROJECT_DIR/.cache/pip")

        cache = self.config.get("cache", {})
        key_files = cache.get("key", {}).get("files", [])
        self.assertIn("pyproject.toml", key_files)
        self.assertIn(".cache/pip", cache.get("paths", []))

    def test_lint_job(self):
        lint = self.config.get("lint", {})
        self.assertEqual(lint.get("stage"), "lint")
        scripts = lint.get("script", [])
        self.assertTrue(any("ruff check" in s for s in scripts))
        self.assertTrue(any("ruff format" in s for s in scripts))

    def test_audit_job(self):
        audit = self.config.get("audit", {})
        self.assertEqual(audit.get("stage"), "audit")
        self.assertEqual(
            audit.get("before_script"),
            [],
            "Audit job must not install dependencies (runs with stdlib alone)",
        )
        scripts = audit.get("script", [])
        self.assertTrue(any("audit_ast.py" in s for s in scripts))

    def test_test_job(self):
        test_job = self.config.get("test", {})
        self.assertEqual(test_job.get("stage"), "test")
        scripts = test_job.get("script", [])
        self.assertTrue(any("pytest" in s for s in scripts))
        self.assertIn("coverage", test_job)
        artifacts = test_job.get("artifacts", {})
        reports = artifacts.get("reports", {})
        self.assertIn("junit", reports)
        self.assertIn("coverage_report", reports)

    def test_release_job(self):
        release = self.config.get("release", {})
        self.assertEqual(release.get("stage"), "release")
        rules = release.get("rules", [])
        self.assertTrue(
            any('$CI_COMMIT_BRANCH == "main"' in r.get("if", "") for r in rules),
            "Release job must only trigger on main branch",
        )
        scripts = release.get("script", [])
        self.assertTrue(any("release_helper.py" in s for s in scripts))


if __name__ == "__main__":
    unittest.main()
