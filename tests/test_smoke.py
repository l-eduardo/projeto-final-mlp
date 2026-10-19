"""
tests/test_smoke.py - Smoke tests for package mlp.
Validates clean import, exported symbols, ABC interface enforcement, and runtime isolation.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path when executed directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    import pytest
except ImportError:
    # Minimal pytest stub for running tests directly without pytest installed
    class _PytestRaisesContext:
        def __init__(self, expected_exception):
            self.expected_exception = expected_exception

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.expected_exception.__name__} was not raised")
            return issubclass(exc_type, self.expected_exception)

    class _PytestStub:
        @staticmethod
        def raises(expected_exception):
            return _PytestRaisesContext(expected_exception)

    pytest = _PytestStub()

import mlp
from mlp.base import Ativacao, Camada, FuncaoCusto, Inicializador


def test_package_exports():
    """Verify that core contract classes are exported in __all__ and present on mlp."""
    expected_symbols = {"Camada", "Ativacao", "FuncaoCusto", "Inicializador", "RedeNeural"}
    assert set(mlp.__all__) == expected_symbols, f"Exported symbols mismatch: {mlp.__all__}"
    for symbol in expected_symbols:
        assert hasattr(mlp, symbol), f"Symbol {symbol} missing from mlp package"
        assert getattr(mlp, symbol) is not None


def test_submodules_import():
    """Verify clean import of all internal submodules."""
    from mlp import ativacoes, base, camadas, custos, inicializadores, rede

    assert base is not None
    assert ativacoes is not None
    assert camadas is not None
    assert custos is not None
    assert inicializadores is not None
    assert rede is not None


def test_abstract_classes_uninstantiable():
    """Verify ABCs cannot be instantiated without implementing abstract methods."""
    with pytest.raises(TypeError):
        Camada()
    with pytest.raises(TypeError):
        Ativacao()
    with pytest.raises(TypeError):
        FuncaoCusto()
    with pytest.raises(TypeError):
        Inicializador()


def test_runtime_isolation():
    """Verify no forbidden ML or notebook-only modules are loaded into sys.modules."""
    forbidden = {
        "torch",
        "sklearn",
        "scipy",
        "autograd",
        "tensorflow",
        "jax",
        "keras",
        "matplotlib",
        "pandas",
        "seaborn",
    }
    loaded = forbidden.intersection(sys.modules.keys())
    assert not loaded, f"Unauthorized runtime libraries loaded in memory: {loaded}"


if __name__ == "__main__":
    print("Running smoke tests...")
    test_package_exports()
    print("  [PASS] test_package_exports")
    test_submodules_import()
    print("  [PASS] test_submodules_import")
    test_abstract_classes_uninstantiable()
    print("  [PASS] test_abstract_classes_uninstantiable")
    test_runtime_isolation()
    print("  [PASS] test_runtime_isolation")
    print("All smoke tests passed successfully!")
