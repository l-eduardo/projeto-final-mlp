"""tests/test_ast_audit.py - Unit test suite for AST compliance auditor.

Validates that scripts/audit_ast.py:
1. Passes clean code (NumPy math, custom tanh implementations, class hierarchies, stdlib).
2. Intercepts all banned ML/autograd libraries (torch, sklearn, scipy, autograd, tensorflow, jax, keras).
3. Intercepts unauthorized runtime modules (pandas, matplotlib, seaborn, statsmodels).
4. Forbids explicit np.tanh, numpy.tanh, aliased tanh, direct import, and wildcard imports.
5. Forbids dynamic execution bypasses (eval, exec, compile, __import__, importlib, getattr).
6. Provides robust CLI behavior, syntax error diagnostics, and proper exit codes (0 vs 1).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_ast import (
    BANNED_BUILTINS,
    BANNED_MODULES,
    UNAUTHORIZED_RUNTIME_MODULES,
    ASTAuditor,
    audit_file,
    audit_source,
    audit_target,
    main,
)


class TestASTAuditorCleanCode(unittest.TestCase):
    """Verifies that legitimate, compliant Python code passes without any violations."""

    def test_clean_numpy_operations(self) -> None:
        source = """
import numpy as np

def compute_layer(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.dot(x, w) + b
    exp_z = np.exp(-z)
    sigmoid = 1.0 / (1.0 + exp_z)
    relu = np.maximum(0.0, z)
    return sigmoid + relu
"""
        violations = audit_source(source)
        self.assertEqual(violations, [])

    def test_custom_tanh_approximation_passes(self) -> None:
        """Academic integrity requirement: manual/analytical tanh formulas MUST pass cleanly."""
        source = """
import numpy as np

def tanh_manual(x: np.ndarray) -> np.ndarray:
    # Explicit mathematical definition using exponential functions
    numer = np.exp(x) - np.exp(-x)
    denom = np.exp(x) + np.exp(-x)
    return numer / denom

def tanh_grad(output: np.ndarray) -> np.ndarray:
    return 1.0 - output ** 2
"""
        violations = audit_source(source)
        self.assertEqual(violations, [])

    def test_clean_layer_and_activation_classes(self) -> None:
        source = """
from abc import ABC, abstractmethod
import numpy as np
from .base import Ativacao, Camada

class TangenteHiperbolica(Ativacao):
    def passo_a_frente(self, x: np.ndarray) -> np.ndarray:
        self.ultima_saida = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.ultima_saida

    def gradiente(self, grad_seguinte: np.ndarray) -> np.ndarray:
        return grad_seguinte * (1.0 - self.ultima_saida ** 2)
"""
        violations = audit_source(source)
        self.assertEqual(violations, [])

    def test_standard_library_and_relative_imports(self) -> None:
        source = """
import math
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from .base import Camada, Ativacao, FuncaoCusto, Inicializador
from .ativacoes import TangenteHiperbolica
"""
        violations = audit_source(source)
        self.assertEqual(violations, [])

    def test_worktree_mlp_is_clean(self) -> None:
        """Verifies that the actual mlp/ directory in the worktree has 0 violations."""
        mlp_dir = REPO_ROOT / "mlp"
        if mlp_dir.exists():
            exit_code = audit_target(mlp_dir)
            self.assertEqual(exit_code, 0)


class TestASTAuditorBannedImports(unittest.TestCase):
    """Verifies that forbidden ML and autograd packages are strictly detected."""

    def test_banned_modules_constants(self) -> None:
        expected = {"torch", "sklearn", "scipy", "autograd", "tensorflow", "jax", "keras"}
        self.assertTrue(expected.issubset(BANNED_MODULES))

    def test_banned_import_statements(self) -> None:
        banned_cases = [
            "import torch",
            "import torch.nn as nn",
            "import sklearn",
            "import sklearn.linear_model",
            "import scipy",
            "import scipy.special",
            "import autograd",
            "import tensorflow as tf",
            "import jax",
            "import keras",
        ]
        for statement in banned_cases:
            with self.subTest(statement=statement):
                violations = audit_source(statement)
                self.assertGreaterEqual(len(violations), 1)
                self.assertIn("Integrity Violation", violations[0])
                self.assertIn("Banned ML/autograd module", violations[0])

    def test_banned_import_from_statements(self) -> None:
        banned_from_cases = [
            "from torch import Tensor",
            "from torch.nn import Linear",
            "from sklearn.linear_model import LogisticRegression",
            "from scipy.special import expit",
            "from autograd import grad",
            "from tensorflow import keras",
            "from jax import numpy as jnp",
            "from keras.models import Sequential",
        ]
        for statement in banned_from_cases:
            with self.subTest(statement=statement):
                violations = audit_source(statement)
                self.assertGreaterEqual(len(violations), 1)
                self.assertIn("Integrity Violation", violations[0])
                self.assertIn("Banned ML/autograd module", violations[0])


class TestASTAuditorUnauthorizedRuntimeImports(unittest.TestCase):
    """Verifies that unauthorized runtime modules are forbidden in the core package."""

    def test_unauthorized_runtime_constants(self) -> None:
        expected = {"pandas", "matplotlib", "seaborn", "statsmodels"}
        self.assertTrue(expected.issubset(UNAUTHORIZED_RUNTIME_MODULES))

    def test_unauthorized_imports(self) -> None:
        unauthorized_cases = [
            "import pandas as pd",
            "from pandas import DataFrame",
            "import matplotlib.pyplot as plt",
            "from matplotlib import pyplot",
            "import seaborn as sns",
            "import statsmodels",
            "from statsmodels.api import OLS",
        ]
        for statement in unauthorized_cases:
            with self.subTest(statement=statement):
                violations = audit_source(statement)
                self.assertGreaterEqual(len(violations), 1)
                self.assertIn("Dependency Violation", violations[0])
                self.assertIn("Unauthorized runtime module", violations[0])


class TestASTAuditorTanhUsage(unittest.TestCase):
    """Verifies that direct or aliased usage of np.tanh is strictly forbidden."""

    def test_np_tanh_direct_call(self) -> None:
        source = """
import numpy as np
def forward(x):
    return np.tanh(x)
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("tanh" in v for v in violations))

    def test_numpy_tanh_unaliased_call(self) -> None:
        source = """
import numpy
def forward(x):
    return numpy.tanh(x)
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("tanh" in v for v in violations))

    def test_custom_numpy_alias_tanh(self) -> None:
        source = """
import numpy as n
def forward(x):
    return n.tanh(x)
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("tanh" in v for v in violations))

    def test_variable_reassigned_numpy_alias(self) -> None:
        source = """
import numpy as np
alias = np
def forward(x):
    return alias.tanh(x)
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("tanh" in v for v in violations))

    def test_attribute_reference_without_call(self) -> None:
        source = """
import numpy as np
fn = np.tanh
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("tanh" in v for v in violations))

    def test_from_numpy_import_tanh(self) -> None:
        source = "from numpy import tanh"
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertIn("Direct import of 'tanh' from numpy is forbidden", violations[0])

    def test_from_numpy_import_tanh_aliased(self) -> None:
        source = "from numpy import tanh as my_tanh"
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertIn("Direct import of 'tanh' from numpy is forbidden", violations[0])

    def test_from_numpy_wildcard_import(self) -> None:
        source = "from numpy import *"
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertIn("Wildcard import from numpy is forbidden", violations[0])

    def test_from_numpy_submodule_wildcard_import(self) -> None:
        source = "from numpy.core import *"
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertIn("Wildcard import from numpy is forbidden", violations[0])

    def test_chained_attribute_numpy_tanh(self) -> None:
        source = """
import numpy
x = numpy.core.umath.tanh(1.0)
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("tanh" in v for v in violations))

    def test_tanh_attribute_on_arbitrary_object(self) -> None:
        source = """
class Wrapper:
    pass
obj = Wrapper()
y = obj.tanh(0.5)
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("tanh" in v for v in violations))


class TestASTAuditorDynamicExecutionBypasses(unittest.TestCase):
    """Verifies that dynamic evasion vectors are completely intercepted."""

    def test_eval_call(self) -> None:
        source = "x = eval('np.tanh(1.0)')"
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertIn("eval()", violations[0])

    def test_exec_call(self) -> None:
        source = "exec('import torch')"
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertIn("exec()", violations[0])

    def test_compile_call(self) -> None:
        source = "c = compile('x = 1', '<string>', 'exec')"
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertIn("compile()", violations[0])

    def test_dunder_import_call(self) -> None:
        source = "mod = __import__('torch')"
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertIn("__import__()", violations[0])

    def test_importlib_import_module_attribute(self) -> None:
        source = """
import importlib
t = importlib.import_module('torch')
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("import_module" in v or "importlib" in v for v in violations))

    def test_import_from_importlib(self) -> None:
        source = """
from importlib import import_module
t = import_module('torch')
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("importlib" in v or "import_module" in v for v in violations))

    def test_getattr_numpy_tanh(self) -> None:
        source = """
import numpy as np
f = getattr(np, 'tanh')
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("getattr" in v and "tanh" in v for v in violations))

    def test_getattr_arbitrary_object_tanh(self) -> None:
        source = """
f = getattr(some_module, 'tanh')
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("getattr" in v and "tanh" in v for v in violations))

    def test_getattr_numpy_dynamic(self) -> None:
        source = """
import numpy as np
f = getattr(np, 'exp')
"""
        violations = audit_source(source)
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(any("getattr on numpy is forbidden" in v for v in violations))


class TestASTAuditorDiagnosticsAndCLI(unittest.TestCase):
    """Verifies file scanning, CLI execution, syntax errors, and exit codes."""

    def test_syntax_error_handling(self) -> None:
        source = "def broken_code(:\n    pass\n"
        violations = audit_source(source, filename="broken.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("Syntax Error", violations[0])

    def test_audit_directory_with_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            clean_file = tmp_path / "clean.py"
            clean_file.write_text("import numpy as np\nx = np.zeros(5)\n", encoding="utf-8")

            bad_file = tmp_path / "bad.py"
            bad_file.write_text("import torch\n", encoding="utf-8")

            exit_code = audit_target(tmp_path)
            self.assertEqual(exit_code, 1)

    def test_audit_directory_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            clean_file1 = tmp_path / "a.py"
            clean_file1.write_text("import numpy as np\nx = np.ones(3)\n", encoding="utf-8")

            clean_file2 = tmp_path / "b.py"
            clean_file2.write_text("from typing import List\ny = [1, 2, 3]\n", encoding="utf-8")

            exit_code = audit_target(tmp_path)
            self.assertEqual(exit_code, 0)

    def test_nonexistent_target_returns_error(self) -> None:
        exit_code = audit_target(Path("/path/to/nonexistent/directory"))
        self.assertEqual(exit_code, 1)

    def test_main_cli_help(self) -> None:
        exit_code = main(["--help"])
        self.assertEqual(exit_code, 0)

    def test_cli_subprocess_execution(self) -> None:
        script_path = REPO_ROOT / "scripts" / "audit_ast.py"
        self.assertTrue(script_path.exists())

        # Test running against actual mlp/ directory
        result_clean = subprocess.run(
            [sys.executable, str(script_path), "mlp"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result_clean.returncode, 0)
        self.assertIn("STATUS: PASSED", result_clean.stdout)

        # Test running against a temporary violating directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / "violator.py"
            tmp_file.write_text("import torch\n", encoding="utf-8")

            result_bad = subprocess.run(
                [sys.executable, str(script_path), tmp_dir],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result_bad.returncode, 1)
            self.assertIn("STATUS: FAILED", result_bad.stdout)
            self.assertIn("Banned ML/autograd module", result_bad.stdout)


if __name__ == "__main__":
    unittest.main()
