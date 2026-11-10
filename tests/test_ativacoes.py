import numpy as np
import pytest

from mlp.ativacoes import LeakyReLU, ReLU, Sigmoide, Tanh

FORMATOS_TESTE = [
    (1, 1),
    (10, 5),
    (32, 64),
    (1, 8),
]


# =============================================================================
# 1. Sigmoide
# =============================================================================


def test_sigmoide_valores_conhecidos():
    """Valida se Sigmoide(0) = 0.5 e limites assintóticos."""
    ativacao = Sigmoide()
    z = np.array([-100.0, 0.0, 100.0])
    a = ativacao.passo_a_frente(z)

    assert np.isclose(a[1], 0.5)
    assert np.isclose(a[0], 0.0, atol=1e-5)
    assert np.isclose(a[2], 1.0, atol=1e-5)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_sigmoide_formato_preservado(formato):
    """Valida se o formato da saída é idêntico ao da entrada no forward e backward."""
    ativacao = Sigmoide()
    z = np.random.randn(*formato)
    a = ativacao.passo_a_frente(z)
    assert a.shape == formato

    grad_saida = np.ones_like(a)
    grad_entrada = ativacao.passo_atras(grad_saida)
    assert grad_entrada.shape == formato


def test_sigmoide_derivada_analitica():
    """Valida se dL/dZ = dL/dA * a * (1 - a)."""
    ativacao = Sigmoide()
    z = np.array([-1.0, 0.0, 2.0])
    a = ativacao.passo_a_frente(z)
    grad_saida = np.array([1.0, 2.0, 0.5])

    grad_entrada = ativacao.passo_atras(grad_saida)
    esperado = grad_saida * a * (1.0 - a)

    np.testing.assert_allclose(grad_entrada, esperado, rtol=1e-6)


def test_sigmoide_estabilidade_sem_overflow():
    """Valida se a Sigmoide não gera NaN/inf com valores extremos."""
    ativacao = Sigmoide()
    z = np.array([-1000.0, -500.0, 500.0, 1000.0])
    a = ativacao.passo_a_frente(z)

    assert not np.any(np.isnan(a))
    assert not np.any(np.isinf(a))
    assert np.all(a >= 0.0) and np.all(a <= 1.0)


# =============================================================================
# 2. ReLU
# =============================================================================


def test_relu_valores_conhecidos():
    """Valida se ReLU(z) = max(0, z)."""
    ativacao = ReLU()
    z = np.array([-5.0, -0.01, 0.0, 0.01, 5.0])
    a = ativacao.passo_a_frente(z)

    esperado = np.array([0.0, 0.0, 0.0, 0.01, 5.0])
    np.testing.assert_allclose(a, esperado)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_relu_formato_preservado(formato):
    """Valida dimensões de entrada e saída na ReLU."""
    ativacao = ReLU()
    z = np.random.randn(*formato)
    a = ativacao.passo_a_frente(z)
    assert a.shape == formato

    grad_entrada = ativacao.passo_atras(np.ones_like(a))
    assert grad_entrada.shape == formato


def test_relu_derivada_analitica():
    """Valida se o gradiente da ReLU é 1 para z > 0 e 0 para z <= 0."""
    ativacao = ReLU()
    z = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])
    ativacao.passo_a_frente(z)

    grad_saida = np.array([1.5, 2.0, 3.0, 4.0, 5.0])
    grad_entrada = ativacao.passo_atras(grad_saida)

    # Para z <= 0 gradiente é 0, para z > 0 repassa grad_saida
    esperado = np.array([0.0, 0.0, 0.0, 4.0, 5.0])
    np.testing.assert_allclose(grad_entrada, esperado)


# =============================================================================
# 3. LeakyReLU
# =============================================================================


def test_leaky_relu_valores_conhecidos():
    """Valida se LeakyReLU(z) = z se z > 0 senão alpha * z."""
    alpha = 0.01
    ativacao = LeakyReLU(alpha=alpha)
    z = np.array([-10.0, -2.0, 0.0, 3.0, 5.0])
    a = ativacao.passo_a_frente(z)

    esperado = np.array([-0.1, -0.02, 0.0, 3.0, 5.0])
    np.testing.assert_allclose(a, esperado)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_leaky_relu_formato_preservado(formato):
    """Valida dimensões de entrada e saída na LeakyReLU."""
    ativacao = LeakyReLU(alpha=0.1)
    z = np.random.randn(*formato)
    a = ativacao.passo_a_frente(z)
    assert a.shape == formato

    grad_entrada = ativacao.passo_atras(np.ones_like(a))
    assert grad_entrada.shape == formato


def test_leaky_relu_derivada_analitica():
    """Valida se o gradiente da LeakyReLU é 1 para z > 0 e alpha para z <= 0."""
    alpha = 0.05
    ativacao = LeakyReLU(alpha=alpha)
    z = np.array([-4.0, -1.0, 0.0, 2.0, 6.0])
    ativacao.passo_a_frente(z)

    grad_saida = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    grad_entrada = ativacao.passo_atras(grad_saida)

    esperado = np.array([2.0 * alpha, 2.0 * alpha, 2.0 * alpha, 2.0, 2.0])
    np.testing.assert_allclose(grad_entrada, esperado)


# =============================================================================
# 4. Tanh Analítica
# =============================================================================


def test_tanh_valores_conhecidos():
    """Valida valores de Tanh(0) = 0 e comportamento anti-simétrico Tanh(-z) = -Tanh(z)."""
    ativacao = Tanh()
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    a = ativacao.passo_a_frente(z)

    assert np.isclose(a[2], 0.0)
    np.testing.assert_allclose(a[:2], -a[4:2:-1], rtol=1e-5)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_tanh_formato_preservado(formato):
    """Valida dimensões de entrada e saída na Tanh analítica."""
    ativacao = Tanh()
    z = np.random.randn(*formato)
    a = ativacao.passo_a_frente(z)
    assert a.shape == formato

    grad_entrada = ativacao.passo_atras(np.ones_like(a))
    assert grad_entrada.shape == formato


def test_tanh_derivada_analitica():
    """Valida se dL/dZ = dL/dA * (1 - tanh^2(Z))."""
    ativacao = Tanh()
    z = np.array([-1.5, 0.0, 1.5])
    a = ativacao.passo_a_frente(z)

    grad_saida = np.array([1.0, 2.0, 0.5])
    grad_entrada = ativacao.passo_atras(grad_saida)

    esperado = grad_saida * (1.0 - a**2)
    np.testing.assert_allclose(grad_entrada, esperado, rtol=1e-6)


def test_tanh_estabilidade_sem_overflow():
    """Valida se a Tanh analítica satura em -1 e 1 sem gerar NaN/inf para entradas extremas."""
    ativacao = Tanh()
    z = np.array([-1000.0, -500.0, 500.0, 1000.0])
    a = ativacao.passo_a_frente(z)

    assert not np.any(np.isnan(a))
    assert not np.any(np.isinf(a))
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    assert np.isclose(a[0], -1.0)
    assert np.isclose(a[-1], 1.0)
