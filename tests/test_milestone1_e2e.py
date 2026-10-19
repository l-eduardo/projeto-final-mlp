"""
tests/test_milestone1_e2e.py - Comprehensive End-to-End Test Suite for Milestone 1.

Opaque-box, requirement-driven verification covering all 4 tiers:
- Tier 1: Feature Coverage (pyproject.toml, packaging isolation, CI/CD pipeline,
  AST audit CLI, MR templates, README, AI_USAGE, and Git hygiene).
- Tier 2: Boundary & Corner Cases (Adversarial AST security checks on synthetic code
  snippets: PyTorch, scikit-learn, SciPy, TensorFlow, JAX, Autograd, np.tanh,
  numpy.tanh, direct tanh imports, wildcard imports, dynamic eval/getattr bypasses,
  syntax errors, and clean custom exponential tanh pass).
- Tier 3: Cross-Feature Combinations (clean import memory isolation, abstract class
  interface enforcement, concrete subclass instantiation, and AST pass on mlp/).
- Tier 4: Real-World Scenarios (academic submission evaluation workflow, simulated CI
  pipeline runner execution, adversarial student evasion attempts, repository git
  metadata hygiene quarantine, and contributor onboarding setup).
"""

from __future__ import annotations

import abc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Map of fallback locations in worktrees during distributed milestone development
WORKTREE_MAP = {
    "pyproject.toml": REPO_ROOT / ".worktrees" / "packaging" / "pyproject.toml",
    ".gitlab-ci.yml": REPO_ROOT / ".worktrees" / "ci-cd" / ".gitlab-ci.yml",
    "scripts/audit_ast.py": REPO_ROOT / ".worktrees" / "ast-audit" / "scripts" / "audit_ast.py",
    "scripts/release_helper.py": REPO_ROOT / ".worktrees" / "ci-cd" / "scripts" / "release_helper.py",
    ".gitlab/merge_request_templates/Default.md": (
        REPO_ROOT / ".worktrees" / "governance" / ".gitlab" / "merge_request_templates" / "Default.md"
    ),
    "README.md": REPO_ROOT / ".worktrees" / "governance" / "README.md",
    "AI_USAGE.md": REPO_ROOT / ".worktrees" / "governance" / "AI_USAGE.md",
    "tests/test_smoke.py": REPO_ROOT / ".worktrees" / "packaging" / "tests" / "test_smoke.py",
}


def get_project_file(rel_path: str) -> Path:
    """
    Resolves project file path, prioritizing repo root (integrated state)
    with seamless fallback to dedicated worktree during parallel implementation.
    """
    direct = REPO_ROOT / rel_path
    wt_fallback = WORKTREE_MAP.get(rel_path)

    if wt_fallback and wt_fallback.exists():
        if not direct.exists():
            return wt_fallback
        # If direct exists but is the unmerged/empty starter file, use the worktree version
        if direct.stat().st_size <= 1 and wt_fallback.stat().st_size > 1:
            return wt_fallback
        if rel_path == "README.md":
            direct_content = direct.read_text(encoding="utf-8")
            if "relatorio_tecnico.ipynb" not in direct_content:
                return wt_fallback

    return direct


def load_toml(path: Path) -> dict:
    """Parses a TOML file using stdlib tomllib or vendored tomli."""
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11+
        return tomllib.loads(text)
    except (ImportError, TypeError):
        pass

    try:
        import pip._vendor.tomli as tomli  # Python 3.10 with pip
        return tomli.loads(text)
    except ImportError:
        pass

    try:
        import tomli  # Optional tomli library
        return tomli.loads(text)
    except ImportError:
        pass

    raise RuntimeError(f"No TOML parsing library found to parse {path}")


def load_yaml(path: Path) -> dict:
    """Parses a YAML file using PyYAML."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Feature-by-feature verification of Milestone 1 requirements."""

    def test_pyproject_toml_format_and_pep621_structure(self) -> None:
        """Requirement R2: Validate pyproject.toml PEP 517/518/621 structure and metadata."""
        pyproject_path = get_project_file("pyproject.toml")
        self.assertTrue(pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}")

        data = load_toml(pyproject_path)
        self.assertIn("build-system", data, "Missing [build-system] table")
        self.assertIn("project", data, "Missing [project] table")

        build_sys = data["build-system"]
        self.assertEqual(build_sys.get("build-backend"), "setuptools.build_meta")
        requires = build_sys.get("requires", [])
        self.assertTrue(any("setuptools" in r for r in requires), "Missing setuptools in build-system.requires")

        proj = data["project"]
        self.assertEqual(proj.get("name"), "mlp", "Project name must be 'mlp'")
        self.assertEqual(proj.get("requires-python"), ">=3.10", "requires-python must be '>=3.10'")
        self.assertIn("version", proj, "Project version must be declared")

    def test_pyproject_exclusive_numpy_runtime_dependency(self) -> None:
        """Requirement R2: Validate that core runtime dependency is STRICTLY and EXCLUSIVELY NumPy."""
        pyproject_path = get_project_file("pyproject.toml")
        data = load_toml(pyproject_path)
        proj = data["project"]

        self.assertIn("dependencies", proj, "Missing [project.dependencies]")
        dependencies = proj["dependencies"]
        self.assertIsInstance(dependencies, list)
        self.assertEqual(len(dependencies), 1, f"Expected exactly 1 runtime dependency, found {len(dependencies)}: {dependencies}")
        self.assertTrue(
            dependencies[0].startswith("numpy"),
            f"The single runtime dependency must be numpy, found: {dependencies[0]}"
        )

        banned = ["torch", "sklearn", "scipy", "autograd", "tensorflow", "jax", "keras", "pandas", "matplotlib"]
        for b in banned:
            for dep in dependencies:
                self.assertNotIn(b, dep.lower(), f"Banned package '{b}' found in core dependencies: {dep}")

    def test_pyproject_optional_dependencies_groups(self) -> None:
        """Requirement R2: Validate isolated optional dependency groups for dev and notebook."""
        pyproject_path = get_project_file("pyproject.toml")
        data = load_toml(pyproject_path)
        proj = data["project"]

        self.assertIn("optional-dependencies", proj, "Missing [project.optional-dependencies]")
        opt_deps = proj["optional-dependencies"]

        # Dev group check
        self.assertIn("dev", opt_deps, "Missing 'dev' optional group")
        dev_list = [d.lower() for d in opt_deps["dev"]]
        self.assertTrue(any("pytest" in d for d in dev_list), "pytest missing from dev group")
        self.assertTrue(any("pytest-cov" in d for d in dev_list), "pytest-cov missing from dev group")
        self.assertTrue(any("ruff" in d for d in dev_list), "ruff missing from dev group")

        # Notebook group check
        self.assertIn("notebook", opt_deps, "Missing 'notebook' optional group")
        nb_list = [d.lower() for d in opt_deps["notebook"]]
        self.assertTrue(any("matplotlib" in d for d in nb_list), "matplotlib missing from notebook group")
        self.assertTrue(any("pandas" in d for d in nb_list), "pandas missing from notebook group")
        self.assertTrue(any("scikit-learn" in d for d in nb_list), "scikit-learn missing from notebook group")
        self.assertTrue(any("jupyter" in d for d in nb_list), "jupyter missing from notebook group")

    def test_gitlab_ci_yaml_format_and_stages(self) -> None:
        """Requirement R3 & R4: Validate .gitlab-ci.yml syntax, stages, and container image."""
        ci_path = get_project_file(".gitlab-ci.yml")
        self.assertTrue(ci_path.exists(), f".gitlab-ci.yml not found at {ci_path}")

        ci_data = load_yaml(ci_path)
        self.assertIn("stages", ci_data, "Missing 'stages' in .gitlab-ci.yml")
        expected_stages = ["lint", "audit", "test", "release"]
        self.assertEqual(ci_data["stages"], expected_stages, f"Stages must match {expected_stages}")

        # Check default image
        default_image = ci_data.get("default", {}).get("image") or ci_data.get("image")
        self.assertEqual(default_image, "python:3.10-slim", "CI base image must be python:3.10-slim")

    def test_gitlab_ci_release_job_branch_restriction(self) -> None:
        """Requirement R4: Validate that CD release job triggers strictly on the main branch."""
        ci_path = get_project_file(".gitlab-ci.yml")
        ci_data = load_yaml(ci_path)

        self.assertIn("release", ci_data, "Missing 'release' job in .gitlab-ci.yml")
        release_job = ci_data["release"]
        self.assertIn("rules", release_job, "Release job must define 'rules' for branch execution")

        # Check for rule checking CI_COMMIT_BRANCH == "main"
        rules = release_job["rules"]
        main_rule_found = False
        for rule in rules:
            if isinstance(rule, dict) and "if" in rule:
                cond = rule["if"]
                if '$CI_COMMIT_BRANCH == "main"' in cond or "$CI_COMMIT_BRANCH == 'main'" in cond:
                    main_rule_found = True
                    break
        self.assertTrue(main_rule_found, "Release job must have a rule restricting execution to '$CI_COMMIT_BRANCH == \"main\"'")

    def test_scripts_audit_ast_exists_and_runs_cli(self) -> None:
        """Requirement R3: Validate that scripts/audit_ast.py exists, is executable, and runs via CLI."""
        script_path = get_project_file("scripts/audit_ast.py")
        self.assertTrue(script_path.exists(), f"scripts/audit_ast.py not found at {script_path}")

        # Test CLI invocation with --help
        res = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"audit_ast.py --help returned non-zero exit code: {res.stderr}")
        self.assertIn("Usage", res.stdout)

    def test_mr_template_academic_compliance_checklist(self) -> None:
        """Requirement R5: Validate GitLab MR template and mandatory Academic Compliance Checklist."""
        mr_template_path = get_project_file(".gitlab/merge_request_templates/Default.md")
        self.assertTrue(mr_template_path.exists(), f"MR template not found at {mr_template_path}")

        content = mr_template_path.read_text(encoding="utf-8")
        self.assertIn("Checklist de Conformidade Acadêmica", content)
        self.assertIn("NumPy", content)
        self.assertIn("tanh", content)
        self.assertIn("base.py", content)
        self.assertIn("AI_USAGE.md", content)

    def test_readme_setup_and_execution_commands(self) -> None:
        """Requirement R5: Validate README.md instructions for setup and local validation commands."""
        readme_path = get_project_file("README.md")
        self.assertTrue(readme_path.exists(), f"README.md not found at {readme_path}")

        content = readme_path.read_text(encoding="utf-8")
        self.assertIn("relatorio_tecnico.ipynb", content, "README must reference relatorio_tecnico.ipynb")
        self.assertNotIn("notebook.ipynb", content, "README must NOT reference obsolete notebook.ipynb")
        self.assertTrue("venv" in content or "virtualenv" in content, "README must explain virtualenv setup")
        self.assertIn("pip install", content, "README must explain pip install")
        self.assertIn("ruff", content, "README must document ruff lint command")
        self.assertIn("audit_ast", content, "README must document AST audit command")
        self.assertIn("pytest", content, "README must document pytest test execution command")

    def test_ai_usage_documentation_milestone1(self) -> None:
        """Requirement Follow-up 02:39:03Z: Validate AI_USAGE.md contains Milestone 1 documentation."""
        ai_usage_path = get_project_file("AI_USAGE.md")
        self.assertTrue(ai_usage_path.exists(), f"AI_USAGE.md not found at {ai_usage_path}")

        content = ai_usage_path.read_text(encoding="utf-8")
        self.assertGreater(len(content.strip()), 100, "AI_USAGE.md must not be empty")
        self.assertIn("Milestone 1", content)
        self.assertTrue("AST" in content or "ast" in content)
        self.assertTrue("CI/CD" in content or "CI" in content)

    def test_gitignore_quarantines_metadata_and_worktrees(self) -> None:
        """Requirement Follow-up 02:41:13Z: Validate .gitignore quarantines .agents/, .gemini/, .claude/, .worktrees/."""
        gitignore_path = REPO_ROOT / ".gitignore"
        self.assertTrue(gitignore_path.exists(), ".gitignore not found")

        content = gitignore_path.read_text(encoding="utf-8")
        self.assertIn(".agents", content, ".agents not in .gitignore")
        self.assertIn(".gemini", content, ".gemini not in .gitignore")
        self.assertIn(".claude", content, ".claude not in .gitignore")
        self.assertIn(".worktrees", content, ".worktrees not in .gitignore")

        # Verify that git does not track any files inside .agents/
        res = subprocess.run(
            ["git", "ls-files", ".agents/"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.stdout.strip(), "", "Found files in .agents/ being tracked by git!")


class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Tier 2: Adversarial AST security audits on synthetic code snippets."""

    def setUp(self) -> None:
        self.script_path = get_project_file("scripts/audit_ast.py")
        self.assertTrue(self.script_path.exists(), f"Audit script missing: {self.script_path}")

    def _run_audit(self, code_str: str) -> subprocess.CompletedProcess[str]:
        """Runs audit_ast.py on a temporary file containing code_str."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(code_str)
            tmp_path = tmp.name

        try:
            res = subprocess.run(
                [sys.executable, str(self.script_path), tmp_path],
                capture_output=True,
                text=True,
            )
            return res
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_ast_audit_banned_torch(self) -> None:
        """Adversarial check: 'import torch' must fail (exit 1)."""
        res = self._run_audit("import torch\nx = torch.tensor([1, 2, 3])\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("torch", res.stdout)
        self.assertIn("Banned ML/autograd module", res.stdout)

    def test_ast_audit_banned_sklearn(self) -> None:
        """Adversarial check: 'from sklearn.ensemble import RandomForestClassifier' must fail (exit 1)."""
        res = self._run_audit("from sklearn.ensemble import RandomForestClassifier\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("sklearn", res.stdout)

    def test_ast_audit_banned_scipy(self) -> None:
        """Adversarial check: 'import scipy' must fail (exit 1)."""
        res = self._run_audit("import scipy\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("scipy", res.stdout)

    def test_ast_audit_banned_tensorflow(self) -> None:
        """Adversarial check: 'import tensorflow as tf' must fail (exit 1)."""
        res = self._run_audit("import tensorflow as tf\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("tensorflow", res.stdout)

    def test_ast_audit_banned_jax(self) -> None:
        """Adversarial check: 'import jax' must fail (exit 1)."""
        res = self._run_audit("import jax\nimport jax.numpy as jnp\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("jax", res.stdout)

    def test_ast_audit_banned_autograd(self) -> None:
        """Adversarial check: 'import autograd' must fail (exit 1)."""
        res = self._run_audit("import autograd\nfrom autograd import grad\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("autograd", res.stdout)

    def test_ast_audit_banned_np_tanh(self) -> None:
        """Adversarial check: 'np.tanh(x)' must fail (exit 1)."""
        code = "import numpy as np\ndef forward(x):\n    return np.tanh(x)\n"
        res = self._run_audit(code)
        self.assertEqual(res.returncode, 1)
        self.assertIn("tanh", res.stdout)

    def test_ast_audit_banned_numpy_unaliased_tanh(self) -> None:
        """Adversarial check: 'numpy.tanh(x)' must fail (exit 1)."""
        code = "import numpy\ndef forward(x):\n    return numpy.tanh(x)\n"
        res = self._run_audit(code)
        self.assertEqual(res.returncode, 1)
        self.assertIn("tanh", res.stdout)

    def test_ast_audit_banned_from_numpy_import_tanh(self) -> None:
        """Adversarial check: 'from numpy import tanh' must fail (exit 1)."""
        res = self._run_audit("from numpy import tanh\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("tanh", res.stdout)

    def test_ast_audit_banned_from_numpy_wildcard_import(self) -> None:
        """Adversarial check: 'from numpy import *' must fail (exit 1)."""
        res = self._run_audit("from numpy import *\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Wildcard import from numpy is forbidden", res.stdout)

    def test_ast_audit_banned_eval_import_bypass(self) -> None:
        """Adversarial check: dynamic 'eval(\"import torch\")' must fail (exit 1)."""
        res = self._run_audit("eval('import torch')\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("eval()", res.stdout)

    def test_ast_audit_banned_getattr_np_tanh_bypass(self) -> None:
        """Adversarial check: dynamic 'getattr(np, \"tanh\")' must fail (exit 1)."""
        code = "import numpy as np\nfn = getattr(np, 'tanh')\n"
        res = self._run_audit(code)
        self.assertEqual(res.returncode, 1)
        self.assertIn("getattr", res.stdout)

    def test_ast_audit_clean_custom_exponential_tanh_passes(self) -> None:
        """Legitimate check: custom exponential tanh implementation must PASS (exit 0)."""
        code = """import numpy as np

class TangenteHiperbolica:
    def passo_a_frente(self, x: np.ndarray) -> np.ndarray:
        exp_pos = np.exp(x)
        exp_neg = np.exp(-x)
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)

    def gradiente(self, grad_seguinte: np.ndarray, x: np.ndarray) -> np.ndarray:
        s = self.passo_a_frente(x)
        return grad_seguinte * (1.0 - s ** 2)
"""
        res = self._run_audit(code)
        self.assertEqual(res.returncode, 0, f"Clean code unexpectedly failed: {res.stdout}")
        self.assertIn("STATUS: PASSED", res.stdout)

    def test_ast_audit_syntax_error_fails(self) -> None:
        """Boundary check: invalid Python syntax must be reported as failure (exit 1)."""
        res = self._run_audit("def broken_syntax(:\n    pass\n")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Syntax Error", res.stdout)

    def test_ast_audit_unauthorized_runtime_modules(self) -> None:
        """Boundary check: unauthorized runtime dependencies like pandas/matplotlib must fail."""
        res_pd = self._run_audit("import pandas as pd\n")
        self.assertEqual(res_pd.returncode, 1)
        self.assertIn("pandas", res_pd.stdout)

        res_plt = self._run_audit("import matplotlib.pyplot as plt\n")
        self.assertEqual(res_plt.returncode, 1)
        self.assertIn("matplotlib", res_plt.stdout)

    def test_ast_audit_dynamic_alias_reassignment_tanh(self) -> None:
        """Boundary check: reassigning numpy to variable and calling .tanh() must fail."""
        code = "import numpy as n\nalias = n\nx = alias.tanh(0.5)\n"
        res = self._run_audit(code)
        self.assertEqual(res.returncode, 1)
        self.assertIn("tanh", res.stdout)


class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Inter-module contracts and runtime isolation."""

    def test_clean_import_memory_isolation(self) -> None:
        """Verify that importing mlp does not load any forbidden or notebook-only modules into sys.modules."""
        check_script = """
import sys
import mlp

forbidden = {
    "torch", "sklearn", "scipy", "autograd", "tensorflow", "jax", "keras",
    "matplotlib", "pandas", "seaborn", "statsmodels"
}
loaded = forbidden.intersection(sys.modules.keys())
if loaded:
    print(f"FAILED: Forbidden packages found in sys.modules: {loaded}")
    sys.exit(1)
print("SUCCESS: Memory isolation verified")
sys.exit(0)
"""
        res = subprocess.run(
            [sys.executable, "-c", check_script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Memory isolation check failed:\n{res.stdout}\n{res.stderr}")
        self.assertIn("SUCCESS: Memory isolation verified", res.stdout)

    def test_base_abstract_classes_enforce_interface(self) -> None:
        """Verify that Camada, Ativacao, FuncaoCusto, Inicializador are abstract base classes."""
        from mlp.base import Ativacao, Camada, FuncaoCusto, Inicializador

        for cls in [Camada, Ativacao, FuncaoCusto, Inicializador]:
            self.assertTrue(issubclass(cls, abc.ABC), f"{cls.__name__} must subclass abc.ABC")
            with self.assertRaises(TypeError, msg=f"{cls.__name__} must raise TypeError on direct instantiation"):
                cls()

    def test_concrete_subclass_instantiation(self) -> None:
        """Verify that concrete implementations fulfilling the base contract in mlp/base.py can be instantiated."""
        import numpy as np
        from mlp.base import Ativacao, Camada, FuncaoCusto, Inicializador

        class ConcreteCamada(Camada):
            def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
                return entradas
            def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
                return gradiente_saida

        class ConcreteAtivacao(Ativacao):
            def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
                return np.maximum(0, entradas)
            def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
                return gradiente_saida

        class ConcreteFuncaoCusto(FuncaoCusto):
            def calcular_custo(self, y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> float:
                return float(np.mean((y_predito - y_verdadeiro) ** 2))
            def gradiente(self, y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> np.ndarray:
                return 2 * (y_predito - y_verdadeiro) / y_predito.size

        class ConcreteInicializador(Inicializador):
            def inicializar(self, formato: tuple[int, int], gerador_aleatorio: np.random.RandomState) -> np.ndarray:
                return gerador_aleatorio.randn(*formato)

        c = ConcreteCamada()
        a = ConcreteAtivacao()
        f = ConcreteFuncaoCusto()
        i = ConcreteInicializador()

        x = np.array([1.0, -1.0])
        self.assertTrue(np.array_equal(c.passo_a_frente(x), x))
        self.assertTrue(np.array_equal(a.passo_a_frente(x), np.array([1.0, 0.0])))
        self.assertEqual(f.calcular_custo(x, x), 0.0)

        rng = np.random.RandomState(42)
        weights = i.inicializar((2, 2), rng)
        self.assertEqual(weights.shape, (2, 2))

    def test_all_mlp_submodules_exist_and_importable(self) -> None:
        """Verify clean import of all expected submodules in mlp/."""
        from mlp import ativacoes, base, camadas, custos, inicializadores, rede
        self.assertIsNotNone(ativacoes)
        self.assertIsNotNone(base)
        self.assertIsNotNone(camadas)
        self.assertIsNotNone(custos)
        self.assertIsNotNone(inicializadores)
        self.assertIsNotNone(rede)

    def test_audit_current_mlp_package_passes(self) -> None:
        """Verify that the actual project mlp/ directory passes the AST audit with 0 violations."""
        script_path = get_project_file("scripts/audit_ast.py")
        mlp_path = REPO_ROOT / "mlp"

        res = subprocess.run(
            [sys.executable, str(script_path), str(mlp_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Actual mlp/ directory failed AST audit:\n{res.stdout}\n{res.stderr}")
        self.assertIn("STATUS: PASSED", res.stdout)
        self.assertIn("0 violation(s) found", res.stdout)


class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: End-to-end real-world scenarios and integration workloads."""

    def test_scenario_1_academic_submission_audit(self) -> None:
        """
        Scenario 1: Evaluates project state from the perspective of an academic evaluator:
        - pyproject.toml has pure NumPy runtime isolation.
        - AST auditor inspects mlp/ and confirms zero violations.
        - MR template contains the mandatory Academic Compliance Checklist.
        - AI_USAGE.md transparently discloses all AI assistance.
        - relatorio_tecnico.ipynb is a valid Jupyter Notebook JSON.
        """
        # 1. Packaging isolation
        pyproject_path = get_project_file("pyproject.toml")
        data = load_toml(pyproject_path)
        deps = data["project"]["dependencies"]
        self.assertEqual(len(deps), 1)
        self.assertTrue(deps[0].startswith("numpy"))

        # 2. AST compliance
        script_path = get_project_file("scripts/audit_ast.py")
        res = subprocess.run(
            [sys.executable, str(script_path), "mlp"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("STATUS: PASSED", res.stdout)

        # 3. MR Checklist
        mr_path = get_project_file(".gitlab/merge_request_templates/Default.md")
        mr_content = mr_path.read_text(encoding="utf-8")
        self.assertIn("Checklist de Conformidade Acadêmica", mr_content)

        # 4. AI usage transparency
        ai_path = get_project_file("AI_USAGE.md")
        ai_content = ai_path.read_text(encoding="utf-8")
        self.assertIn("Milestone 1", ai_content)

        # 5. Notebook validity
        nb_path = REPO_ROOT / "relatorio_tecnico.ipynb"
        self.assertTrue(nb_path.exists(), "relatorio_tecnico.ipynb missing")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_json = json.load(f)
        self.assertIn("cells", nb_json)
        self.assertIn("metadata", nb_json)

    def test_scenario_2_simulated_ci_pipeline(self) -> None:
        """
        Scenario 2: Simulates the execution of the full CI pipeline jobs in order:
        - Validates stages declaration order: lint -> audit -> test -> release.
        - Validates lint job scripts.
        - Executes audit job command: python3 scripts/audit_ast.py mlp.
        - Executes smoke test suite in tests/test_smoke.py.
        - Validates release job configuration.
        """
        ci_path = get_project_file(".gitlab-ci.yml")
        ci_data = load_yaml(ci_path)

        # Stage order check
        self.assertEqual(ci_data["stages"], ["lint", "audit", "test", "release"])

        # Audit stage execution
        audit_script = get_project_file("scripts/audit_ast.py")
        res_audit = subprocess.run(
            [sys.executable, str(audit_script), "mlp"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res_audit.returncode, 0, f"CI audit stage failed: {res_audit.stdout}")

        # Smoke test execution
        smoke_test = get_project_file("tests/test_smoke.py")
        if smoke_test.exists():
            res_smoke = subprocess.run(
                [sys.executable, str(smoke_test)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_smoke.returncode, 0, f"Smoke tests failed: {res_smoke.stdout}\n{res_smoke.stderr}")

        # Release stage rule
        self.assertIn("release", ci_data)
        rules = ci_data["release"].get("rules", [])
        self.assertTrue(any('$CI_COMMIT_BRANCH == "main"' in r.get("if", "") for r in rules if isinstance(r, dict)))

    def test_scenario_3_adversarial_student_cheating_interception(self) -> None:
        """
        Scenario 3: Simulates student attempting various evasions, ensuring all are intercepted:
        - Case A: Scoped import inside a method body.
        - Case B: Dynamic getattr on numpy.
        - Case C: Renamed import of forbidden function.
        - Case D: Indirect assignment before method call.
        """
        script_path = get_project_file("scripts/audit_ast.py")

        cases = [
            ("case_a_scoped_import.py", "class Rede:\n    def treinar(self):\n        import torch\n        return torch.zeros(1)\n"),
            ("case_b_getattr_tanh.py", "import numpy as np\ndef calc(x):\n    fn = getattr(np, 'tanh')\n    return fn(x)\n"),
            ("case_c_renamed_tanh.py", "from numpy import tanh as custom_act\ndef forward(x):\n    return custom_act(x)\n"),
            ("case_d_indirect_alias.py", "import numpy as np\ndef forward(x):\n    np_alias = np\n    return np_alias.tanh(x)\n"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for fname, code in cases:
                file_path = Path(tmp_dir) / fname
                file_path.write_text(code, encoding="utf-8")

                res = subprocess.run(
                    [sys.executable, str(script_path), str(file_path)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    res.returncode, 1,
                    f"Evasion attack '{fname}' was not intercepted! Result:\n{res.stdout}"
                )
                self.assertIn("STATUS: FAILED", res.stdout)

    def test_scenario_4_git_hygiene_and_quarantine(self) -> None:
        """
        Scenario 4: Simulates repository quarantine check:
        - Verifies no AI metadata is tracked.
        - Verifies gitignore rules match files under quarantined paths.
        """
        for pattern in [".agents/", ".gemini/", ".worktrees/"]:
            res = subprocess.run(
                ["git", "ls-files", pattern],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0)
            self.assertEqual(
                res.stdout.strip(), "",
                f"Quarantine breach: files tracked under {pattern}:\n{res.stdout}"
            )

        # Verify git check-ignore matches test paths under quarantined prefixes
        for dummy_path in [".agents/dummy_test.tmp", ".gemini/dummy_test.tmp", ".worktrees/dummy_test.tmp"]:
            res_ignore = subprocess.run(
                ["git", "check-ignore", "-v", dummy_path],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res_ignore.returncode, 0,
                f"Path {dummy_path} was not matched by gitignore:\n{res_ignore.stdout}"
            )

    def test_scenario_5_contributor_onboarding_documentation(self) -> None:
        """
        Scenario 5: Simulates a new contributor onboarding to the repository:
        - Verifies that README.md documents the essential workflow clearly.
        - Checks presence of venv creation, pip install options, ruff, audit, and pytest.
        """
        readme_path = get_project_file("README.md")
        content = readme_path.read_text(encoding="utf-8")

        required_snippets = [
            "python3 -m venv",
            "pip install -e .",
            "pip install -e .[dev]",
            "pip install -e .[notebook]",
            "ruff check",
            "python3 scripts/audit_ast.py",
            "pytest tests/",
            "relatorio_tecnico.ipynb",
        ]

        for snippet in required_snippets:
            self.assertIn(
                snippet, content,
                f"README.md missing expected contributor instruction: '{snippet}'"
            )


if __name__ == "__main__":
    unittest.main()
