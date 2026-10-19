#!/usr/bin/env python3
"""scripts/audit_ast.py - Academic Integrity & AST Compliance Auditor

Statically inspects Python files to ensure strict academic integrity and runtime isolation:
1. Intercepts forbidden ML / autograd frameworks (torch, sklearn, scipy, autograd, tensorflow, jax, keras).
2. Intercepts unauthorized runtime modules (pandas, matplotlib, seaborn, statsmodels) in core package.
3. Dynamically tracks NumPy aliases (e.g. 'import numpy as np', 'import numpy as n', variable aliasing).
4. Forbids explicit access to np.tanh, numpy.tanh, or any aliased tanh, as well as direct import
   of tanh ('from numpy import tanh') and wildcard imports ('from numpy import *').
5. Prevents dynamic execution bypasses (eval, exec, compile, __import__, importlib, dynamic getattr).
6. Exits with code 0 if all audited files pass cleanly; exits with code 1 upon violation or syntax error.

Pure Python standard library implementation with zero third-party dependencies.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Sequence, Set

BANNED_MODULES: Set[str] = {
    "torch",
    "sklearn",
    "scipy",
    "autograd",
    "tensorflow",
    "jax",
    "keras",
    "importlib",
}

UNAUTHORIZED_RUNTIME_MODULES: Set[str] = {
    "pandas",
    "matplotlib",
    "seaborn",
    "statsmodels",
}

BANNED_BUILTINS: Set[str] = {
    "eval",
    "exec",
    "__import__",
    "compile",
}


class ASTAuditor(ast.NodeVisitor):
    """AST visitor enforcing academic integrity and compliance rules."""

    def __init__(self, filename: str = "<string>") -> None:
        self.filename: str = filename
        self.violations: List[str] = []
        self.numpy_aliases: Set[str] = {"numpy", "np"}

    def _record_violation(self, node: ast.AST, category: str, message: str) -> None:
        lineno = getattr(node, "lineno", 1)
        col_offset = getattr(node, "col_offset", 0)
        self.violations.append(
            f"{self.filename}:{lineno}:{col_offset}: {category} - {message}"
        )

    def _is_numpy_receiver(self, node: ast.AST) -> bool:
        """Recursively checks if an AST node resolves to a known NumPy alias."""
        if isinstance(node, ast.Name):
            return node.id in self.numpy_aliases
        if isinstance(node, ast.Attribute):
            return self._is_numpy_receiver(node.value)
        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "modules"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "sys"
            ):
                if isinstance(node.slice, ast.Constant) and node.slice.value in self.numpy_aliases:
                    return True
        return False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root == "importlib":
                self._record_violation(
                    node,
                    "Security Violation",
                    "Dynamic module loader 'importlib' is forbidden",
                )
            elif root in BANNED_MODULES:
                self._record_violation(
                    node,
                    "Integrity Violation",
                    f"Banned ML/autograd module imported: '{alias.name}'",
                )
            elif root in UNAUTHORIZED_RUNTIME_MODULES:
                self._record_violation(
                    node,
                    "Dependency Violation",
                    f"Unauthorized runtime module in mlp/: '{alias.name}'",
                )
            if root == "numpy":
                self.numpy_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Relative package imports (e.g. 'from .base import Camada') are legitimate internal imports
        if node.level > 0:
            self.generic_visit(node)
            return

        if node.module:
            root = node.module.split(".")[0]
            if root == "importlib":
                self._record_violation(
                    node,
                    "Security Violation",
                    "Dynamic module loader 'importlib' is forbidden",
                )
            elif root in BANNED_MODULES:
                self._record_violation(
                    node,
                    "Integrity Violation",
                    f"Banned ML/autograd module imported: '{node.module}'",
                )
            elif root in UNAUTHORIZED_RUNTIME_MODULES:
                self._record_violation(
                    node,
                    "Dependency Violation",
                    f"Unauthorized runtime module in mlp/: '{node.module}'",
                )

            # Catch direct 'tanh' import from ANY non-internal module (numpy, math, cmath, etc.)
            for alias in node.names:
                if alias.name == "tanh":
                    if root == "numpy":
                        msg = "Direct import of 'tanh' from numpy is forbidden"
                    else:
                        msg = f"Direct import of 'tanh' from '{node.module}' is forbidden"
                    self._record_violation(
                        node,
                        "Integrity Violation",
                        msg,
                    )
                elif root == "numpy" and alias.name == "*":
                    self._record_violation(
                        node,
                        "Integrity Violation",
                        "Wildcard import from numpy is forbidden (exposes np.tanh)",
                    )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Dynamically track variable reassignments of numpy aliases (e.g. 'alias = np')
        if isinstance(node.value, ast.Name) and node.value.id in self.numpy_aliases:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.numpy_aliases.add(target.id)
        self.generic_visit(node)

    def _check_getattr_call(self, node: ast.Call) -> None:
        if len(node.args) >= 2:
            receiver = node.args[0]
            attr_arg = node.args[1]
            is_numpy = self._is_numpy_receiver(receiver)
            if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                if attr_arg.value == "tanh":
                    self._record_violation(
                        node,
                        "Integrity Violation",
                        "Dynamic getattr for 'tanh' is forbidden",
                    )
                elif attr_arg.value in BANNED_BUILTINS:
                    self._record_violation(
                        node,
                        "Security Violation",
                        f"Dynamic getattr for banned builtin '{attr_arg.value}' is forbidden",
                    )
                elif is_numpy:
                    self._record_violation(
                        node,
                        "Security Violation",
                        "Dynamic getattr on numpy is forbidden",
                    )
            elif is_numpy:
                self._record_violation(
                    node,
                    "Security Violation",
                    "Dynamic getattr on numpy is forbidden",
                )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._is_numpy_receiver(node.value) and node.attr in {"__dict__", "__getattribute__"}:
            self._record_violation(
                node,
                "Security Violation",
                f"Dynamic reflection '{node.attr}' on numpy is forbidden",
            )
        if node.attr == "tanh":
            if self._is_numpy_receiver(node.value):
                self._record_violation(
                    node,
                    "Integrity Violation",
                    "Forbidden usage of numpy tanh attribute",
                )
            elif isinstance(node.value, ast.Name) and node.value.id in self.numpy_aliases:
                self._record_violation(
                    node,
                    "Integrity Violation",
                    f"Forbidden usage of '{node.value.id}.tanh'",
                )
            elif isinstance(node.value, ast.Name) and node.value.id != "self":
                self._record_violation(
                    node,
                    "Integrity Violation",
                    f"Forbidden access to '.tanh' on '{node.value.id}'",
                )
            elif not isinstance(node.value, ast.Name):
                self._record_violation(
                    node,
                    "Integrity Violation",
                    "Forbidden access to '.tanh' attribute",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct calls to banned builtins and dynamic loaders
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in BANNED_BUILTINS:
                self._record_violation(
                    node,
                    "Security Violation",
                    f"Banned dynamic execution builtin: '{func_name}()'",
                )
            elif func_name == "import_module":
                self._record_violation(
                    node,
                    "Security Violation",
                    "Dynamic module loading via 'import_module()' is forbidden",
                )
            elif func_name == "getattr":
                self._check_getattr_call(node)
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in BANNED_BUILTINS:
                self._record_violation(
                    node,
                    "Security Violation",
                    f"Banned dynamic execution attribute call: '{attr_name}()'",
                )
            elif attr_name == "import_module":
                self._record_violation(
                    node,
                    "Security Violation",
                    "Dynamic module loading via 'import_module()' is forbidden",
                )
            elif attr_name == "getattr":
                self._check_getattr_call(node)
        self.generic_visit(node)


def audit_source(source: str, filename: str = "<string>") -> List[str]:
    """Parses and audits a Python source code string for integrity violations."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as err:
        lineno = err.lineno or 1
        offset = err.offset or 0
        return [f"{filename}:{lineno}:{offset}: Syntax Error - {err.msg}"]

    auditor = ASTAuditor(filename=filename)
    auditor.visit(tree)
    return auditor.violations


def audit_file(file_path: Path, display_name: str | None = None) -> List[str]:
    """Audits a single Python file from disk."""
    name = display_name or str(file_path)
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"{name}:1:0: File Read Error - {exc}"]
    return audit_source(content, filename=name)


def audit_target(target_path: Path, root_display_dir: Path | None = None) -> int:
    """Audits a target directory or file, printing diagnostics and returning exit code (0 or 1)."""
    if not target_path.exists():
        print(f"[AST AUDIT ERROR] Target does not exist: {target_path}", file=sys.stderr)
        return 1

    if target_path.is_file():
        py_files = [target_path]
    elif target_path.is_dir():
        py_files = sorted(target_path.rglob("*.py"))
    else:
        print(f"[AST AUDIT ERROR] Target is not a file or directory: {target_path}", file=sys.stderr)
        return 1

    base_dir = root_display_dir or (target_path.parent if target_path.is_file() else target_path.parent)

    print("==================================================")
    print(f" AST Compliance & Integrity Audit: {target_path}")
    print("==================================================")

    total_files = 0
    total_violations = 0

    for py_file in py_files:
        total_files += 1
        try:
            display_name = str(py_file.relative_to(base_dir))
        except ValueError:
            display_name = str(py_file)

        violations = audit_file(py_file, display_name=display_name)
        for violation in violations:
            print(f"  [FAIL] {violation}")
            total_violations += 1

    print("--------------------------------------------------")
    print(f"Audit Summary: {total_files} file(s) scanned, {total_violations} violation(s) found.")

    if total_violations > 0:
        print("STATUS: FAILED (Violations must be fixed before integration).")
        return 1

    print("STATUS: PASSED (All files comply with academic integrity rules).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if "-h" in args or "--help" in args:
        print("Usage: audit_ast.py [TARGET_PATH]")
        print("Statically inspects Python files for academic integrity compliance.")
        print("Defaults to inspecting 'mlp' directory.")
        return 0

    target = Path(args[0]) if args else Path("mlp")
    return audit_target(target)


if __name__ == "__main__":
    sys.exit(main())
