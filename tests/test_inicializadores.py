import numpy as np
import pytest

from mlp.inicializadores import (
    GlorotNormal,
    GlorotUniforme,
    HeNormal,
    Zeros,
)

FORMATOS_TESTE = [
    (1, 1),
    (5, 10),
    (10, 5),
    (64, 32),
    (1, 8),
]


# =============================================================================
# 1. Zeros
# =============================================================================


def test_zeros_mesma_seed_reprodutibilidade():
    """Valida se duas execuções de Zeros com a mesma semente produzem a mesma matriz."""
    inicializador = Zeros()
    formato = (10, 5)

    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(42)

    pesos_1 = inicializador.inicializar(formato, gerador_1)
    pesos_2 = inicializador.inicializar(formato, gerador_2)

    assert isinstance(pesos_1, np.ndarray)
    assert isinstance(pesos_2, np.ndarray)
    np.testing.assert_array_equal(pesos_1, pesos_2)


def test_zeros_independe_da_seed():
    """Valida se sementes distintas produzem a mesma matriz nula em Zeros."""
    inicializador = Zeros()
    formato = (10, 5)

    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(99)

    pesos_1 = inicializador.inicializar(formato, gerador_1)
    pesos_2 = inicializador.inicializar(formato, gerador_2)

    np.testing.assert_array_equal(pesos_1, pesos_2)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_zeros_formato_saida(formato):
    """Valida se o array retornado por Zeros tem exatamente as dimensões requisitadas."""
    inicializador = Zeros()
    gerador = np.random.RandomState(42)

    pesos = inicializador.inicializar(formato, gerador)

    assert isinstance(pesos, np.ndarray)
    assert pesos.shape == formato
    assert np.issubdtype(pesos.dtype, np.floating)


def test_zeros_valores_exatos():
    """Valida se todos os elementos gerados por Zeros são estritamente iguais a 0."""
    inicializador = Zeros()
    formato = (20, 15)
    gerador = np.random.RandomState(42)

    pesos = inicializador.inicializar(formato, gerador)

    assert np.all(pesos == 0.0)
    np.testing.assert_array_equal(pesos, np.zeros(formato, dtype=pesos.dtype))


# =============================================================================
# 2. GlorotNormal
# =============================================================================


def test_glorot_normal_mesma_seed_reprodutibilidade():
    """Valida se duas execuções de GlorotNormal com a mesma semente produzem a mesma matriz."""
    inicializador = GlorotNormal()
    formato = (10, 5)

    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(42)

    pesos_1 = inicializador.inicializar(formato, gerador_1)
    pesos_2 = inicializador.inicializar(formato, gerador_2)

    assert isinstance(pesos_1, np.ndarray)
    assert isinstance(pesos_2, np.ndarray)
    np.testing.assert_array_equal(pesos_1, pesos_2)


def test_glorot_normal_seeds_diferentes():
    """Valida se execuções de GlorotNormal com sementes distintas geram matrizes diferentes."""
    inicializador = GlorotNormal()
    formato = (10, 5)

    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(99)

    pesos_1 = inicializador.inicializar(formato, gerador_1)
    pesos_2 = inicializador.inicializar(formato, gerador_2)

    assert not np.array_equal(pesos_1, pesos_2)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_glorot_normal_formato_saida(formato):
    """Valida se o array retornado por GlorotNormal tem exatamente as dimensões requisitadas."""
    inicializador = GlorotNormal()
    gerador = np.random.RandomState(42)

    pesos = inicializador.inicializar(formato, gerador)

    assert isinstance(pesos, np.ndarray)
    assert pesos.shape == formato
    assert np.issubdtype(pesos.dtype, np.floating)


def test_glorot_normal_distribuicao_estatistica():
    """Valida se a média e a variância empíricas de GlorotNormal convergem para os valores teóricos."""
    inicializador = GlorotNormal()
    fan_in, fan_out = 500, 500
    formato = (fan_in, fan_out)
    gerador = np.random.RandomState(42)

    pesos = inicializador.inicializar(formato, gerador)

    # Média teórica é 0
    assert abs(np.mean(pesos)) < 0.01

    # Variância teórica de N(0, std^2) é 2 / (fan_in + fan_out)
    variancia_teorica = 2.0 / (fan_in + fan_out)
    variancia_empirica = np.var(pesos)

    np.testing.assert_allclose(variancia_empirica, variancia_teorica, rtol=0.1)


# =============================================================================
# 3. GlorotUniforme
# =============================================================================


def test_glorot_uniforme_mesma_seed_reprodutibilidade():
    """Valida se duas execuções de GlorotUniforme com a mesma semente produzem a mesma matriz."""
    inicializador = GlorotUniforme()
    formato = (10, 5)

    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(42)

    pesos_1 = inicializador.inicializar(formato, gerador_1)
    pesos_2 = inicializador.inicializar(formato, gerador_2)

    assert isinstance(pesos_1, np.ndarray)
    assert isinstance(pesos_2, np.ndarray)
    np.testing.assert_array_equal(pesos_1, pesos_2)


def test_glorot_uniforme_seeds_diferentes():
    """Valida se execuções de GlorotUniforme com sementes distintas geram matrizes diferentes."""
    inicializador = GlorotUniforme()
    formato = (10, 5)

    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(99)

    pesos_1 = inicializador.inicializar(formato, gerador_1)
    pesos_2 = inicializador.inicializar(formato, gerador_2)

    assert not np.array_equal(pesos_1, pesos_2)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_glorot_uniforme_formato_saida(formato):
    """Valida se o array retornado por GlorotUniforme tem exatamente as dimensões requisitadas."""
    inicializador = GlorotUniforme()
    gerador = np.random.RandomState(42)

    pesos = inicializador.inicializar(formato, gerador)

    assert isinstance(pesos, np.ndarray)
    assert pesos.shape == formato
    assert np.issubdtype(pesos.dtype, np.floating)


def test_glorot_uniforme_intervalo_teorico():
    """Valida se todos os valores gerados por GlorotUniforme estão contidos no intervalo [-L, L].

    Para Glorot Uniforme: L = sqrt(6 / (fan_in + fan_out)).
    """
    inicializador = GlorotUniforme()
    fan_in, fan_out = 50, 30
    formato = (fan_in, fan_out)
    gerador = np.random.RandomState(42)

    limite_esperado = np.sqrt(6.0 / (fan_in + fan_out))
    pesos = inicializador.inicializar(formato, gerador)

    assert np.all(pesos >= -limite_esperado)
    assert np.all(pesos <= limite_esperado)


def test_glorot_uniforme_distribuicao_estatistica():
    """Valida se a média e a variância empíricas de GlorotUniforme convergem para os valores teóricos."""
    inicializador = GlorotUniforme()
    fan_in, fan_out = 500, 500
    formato = (fan_in, fan_out)
    gerador = np.random.RandomState(42)

    pesos = inicializador.inicializar(formato, gerador)

    # Média teórica é 0
    assert abs(np.mean(pesos)) < 0.01

    # Variância teórica de U(-L, L) é L^2 / 3 = 2 / (fan_in + fan_out)
    variancia_teorica = 2.0 / (fan_in + fan_out)
    variancia_empirica = np.var(pesos)

    # Tolerância relativa de 10% para uma matriz de 250.000 pontos
    np.testing.assert_allclose(variancia_empirica, variancia_teorica, rtol=0.1)


# =============================================================================
# 4. HeNormal
# =============================================================================


def test_he_normal_mesma_seed_reprodutibilidade():
    """Valida se duas execuções de HeNormal com a mesma semente produzem a mesma matriz."""
    inicializador = HeNormal()
    formato = (10, 5)

    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(42)

    pesos_1 = inicializador.inicializar(formato, gerador_1)
    pesos_2 = inicializador.inicializar(formato, gerador_2)

    assert isinstance(pesos_1, np.ndarray)
    assert isinstance(pesos_2, np.ndarray)
    np.testing.assert_array_equal(pesos_1, pesos_2)


def test_he_normal_seeds_diferentes():
    """Valida se execuções de HeNormal com sementes distintas geram matrizes diferentes."""
    inicializador = HeNormal()
    formato = (10, 5)

    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(99)

    pesos_1 = inicializador.inicializar(formato, gerador_1)
    pesos_2 = inicializador.inicializar(formato, gerador_2)

    assert not np.array_equal(pesos_1, pesos_2)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_he_normal_formato_saida(formato):
    """Valida se o array retornado por HeNormal tem exatamente as dimensões requisitadas."""
    inicializador = HeNormal()
    gerador = np.random.RandomState(42)

    pesos = inicializador.inicializar(formato, gerador)

    assert isinstance(pesos, np.ndarray)
    assert pesos.shape == formato
    assert np.issubdtype(pesos.dtype, np.floating)


def test_he_normal_distribuicao_estatistica():
    """Valida se a média e a variância empíricas de HeNormal convergem para os valores teóricos."""
    inicializador = HeNormal()
    fan_in, fan_out = 500, 500
    formato = (fan_in, fan_out)
    gerador = np.random.RandomState(42)

    pesos = inicializador.inicializar(formato, gerador)

    # Média teórica é 0
    assert abs(np.mean(pesos)) < 0.01

    # Variância teórica de HeNormal é 2 / fan_in (independe de fan_out)
    variancia_teorica = 2.0 / fan_in
    variancia_empirica = np.var(pesos)

    np.testing.assert_allclose(variancia_empirica, variancia_teorica, rtol=0.1)


def test_he_normal_independe_de_fan_out_na_variancia():
    """Valida se a variância de HeNormal depende apenas de fan_in e não de fan_out."""
    inicializador = HeNormal()
    fan_in = 400
    gerador_1 = np.random.RandomState(42)
    gerador_2 = np.random.RandomState(99)

    pesos_fan_out_pequeno = inicializador.inicializar((fan_in, 200), gerador_1)
    pesos_fan_out_grande = inicializador.inicializar((fan_in, 1000), gerador_2)

    variancia_esperada = 2.0 / fan_in
    np.testing.assert_allclose(
        np.var(pesos_fan_out_pequeno), variancia_esperada, rtol=0.1
    )
    np.testing.assert_allclose(
        np.var(pesos_fan_out_grande), variancia_esperada, rtol=0.1
    )
