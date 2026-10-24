import pytest
import numpy as np
import mlp


def test_package_import():
    """Valida importação limpa do pacote mlp."""
    assert hasattr(mlp, "__version__") or hasattr(mlp, "base")


def test_numpy_operations_available():
    """Valida disponibilidade do NumPy para operações matriciais."""
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert float(np.dot(a, b)) == 11.0
