import numpy as np
import pytest

from mlp.custos import BCE, CCE, MSE

FORMATOS_TESTE = [
    (1, 1),
    (10, 1),
    (16, 4),
    (32, 10),
]


# =============================================================================
# 1. Erro Quadrático Médio (MSE)
# =============================================================================


def test_mse_valores_conhecidos():
    """Valida cálculo do MSE para predições perfeitas e desvios fixos."""
    custo = MSE()
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert np.isclose(custo.calcular_custo(y_true, y_pred), 0.0)

    y_pred_desvio = np.array(
        [[2.0, 2.0], [3.0, 6.0]]
    )  # erros: 1, 0, 0, 2 -> 1 + 4 = 5 / 2 amostras = 2.5
    assert np.isclose(custo.calcular_custo(y_true, y_pred_desvio), 2.5)


@pytest.mark.parametrize("formato", FORMATOS_TESTE)
def test_mse_formato_preservado(formato):
    """Valida se o formato do gradiente do MSE é idêntico ao de y_predito."""
    custo = MSE()
    y_true = np.random.randn(*formato)
    y_pred = np.random.randn(*formato)

    loss = custo.calcular_custo(y_true, y_pred)
    assert isinstance(loss, float)

    grad = custo.gradiente(y_true, y_pred)
    assert grad.shape == formato


def test_mse_gradiente_analitico_vs_diferencas_finitas():
    """Valida a derivada analítica do MSE contra aproximação por diferenças finitas centrais."""
    custo = MSE()
    rng = np.random.RandomState(42)
    y_true = rng.randn(5, 3)
    y_pred = rng.randn(5, 3)

    grad_analitico = custo.gradiente(y_true, y_pred)

    eps = 1e-6
    grad_num = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            y_mais = y_pred.copy()
            y_mais[i, j] += eps
            y_menos = y_pred.copy()
            y_menos[i, j] -= eps
            grad_num[i, j] = (
                custo.calcular_custo(y_true, y_mais)
                - custo.calcular_custo(y_true, y_menos)
            ) / (2 * eps)

    np.testing.assert_allclose(grad_analitico, grad_num, rtol=1e-5, atol=1e-6)


# =============================================================================
# 2. Entropia Cruzada Binária (BCE)
# =============================================================================


def test_bce_valores_conhecidos():
    """Valida BCE para predição de máxima incerteza (p=0.5) e quase perfeita."""
    custo = BCE()
    y_true = np.array([[1.0], [0.0]])
    y_pred = np.array([[0.5], [0.5]])

    # -ln(0.5) = ln(2) approx 0.693147
    assert np.isclose(custo.calcular_custo(y_true, y_pred), np.log(2.0), rtol=1e-5)


@pytest.mark.parametrize("formato", [(1, 1), (10, 1), (32, 1)])
def test_bce_formato_preservado(formato):
    """Valida dimensões de entrada e saída na BCE."""
    custo = BCE()
    y_true = np.random.randint(0, 2, size=formato).astype(np.float64)
    y_pred = np.random.uniform(0.1, 0.9, size=formato)

    loss = custo.calcular_custo(y_true, y_pred)
    assert isinstance(loss, float)

    grad = custo.gradiente(y_true, y_pred)
    assert grad.shape == formato


def test_bce_estabilidade_sem_overflow():
    """Valida que a BCE não gera NaN ou Inf para predições extremas (0.0 ou 1.0)."""
    custo = BCE()
    y_true = np.array([[1.0], [0.0]])
    y_pred_extremo = np.array([[0.0], [1.0]])  # Pior cenário (erro máximo)

    loss = custo.calcular_custo(y_true, y_pred_extremo)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    assert loss > 0.0

    grad = custo.gradiente(y_true, y_pred_extremo)
    assert not np.any(np.isnan(grad))
    assert not np.any(np.isinf(grad))


def test_bce_gradiente_analitico_vs_diferencas_finitas():
    """Valida a derivada analítica da BCE contra diferenças finitas centrais."""
    custo = BCE()
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=(4, 1)).astype(np.float64)
    y_pred = rng.uniform(0.2, 0.8, size=(4, 1))

    grad_analitico = custo.gradiente(y_true, y_pred)

    eps = 1e-6
    grad_num = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        y_mais = y_pred.copy()
        y_mais[i, 0] += eps
        y_menos = y_pred.copy()
        y_menos[i, 0] -= eps
        grad_num[i, 0] = (
            custo.calcular_custo(y_true, y_mais) - custo.calcular_custo(y_true, y_menos)
        ) / (2 * eps)

    np.testing.assert_allclose(grad_analitico, grad_num, rtol=1e-5, atol=1e-5)


# =============================================================================
# 3. Entropia Cruzada Categórica (CCE)
# =============================================================================


def test_cce_valores_conhecidos():
    """Valida CCE com probabilidades uniformes e alvos one-hot."""
    custo = CCE()
    n_classes = 3
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = np.full((2, n_classes), 1.0 / n_classes)

    # Para distribuicao uniforme, CCE = -ln(1/3) = ln(3)
    assert np.isclose(custo.calcular_custo(y_true, y_pred), np.log(3.0), rtol=1e-5)


def test_cce_alvos_inteiros_e_one_hot():
    """Valida equivalência de CCE com alvos em formato de vetor de inteiros vs. matriz one-hot."""
    custo = CCE()
    y_inteiros = np.array([0, 2, 1])
    y_one_hot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])

    loss_int = custo.calcular_custo(y_inteiros, y_pred)
    loss_onehot = custo.calcular_custo(y_one_hot, y_pred)
    assert np.isclose(loss_int, loss_onehot)

    grad_int = custo.gradiente(y_inteiros, y_pred)
    grad_onehot = custo.gradiente(y_one_hot, y_pred)
    np.testing.assert_allclose(grad_int, grad_onehot)


@pytest.mark.parametrize("formato", [(1, 4), (8, 3), (16, 5)])
def test_cce_formato_preservado(formato):
    """Valida preservação dimensional na CCE."""
    custo = CCE()
    n_amostras, n_classes = formato
    indices = np.random.randint(0, n_classes, size=n_amostras)
    y_true = np.zeros(formato)
    y_true[np.arange(n_amostras), indices] = 1.0

    exps = np.exp(np.random.randn(*formato))
    y_pred = exps / np.sum(exps, axis=-1, keepdims=True)

    loss = custo.calcular_custo(y_true, y_pred)
    assert isinstance(loss, float)

    grad = custo.gradiente(y_true, y_pred)
    assert grad.shape == formato


def test_cce_estabilidade_sem_overflow():
    """Valida se a CCE suporta probabilidades nulas sem gerar NaN/Inf."""
    custo = CCE()
    y_true = np.array([[1.0, 0.0, 0.0]])
    y_pred_zero = np.array([[0.0, 0.5, 0.5]])

    loss = custo.calcular_custo(y_true, y_pred_zero)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    assert loss > 0.0

    grad = custo.gradiente(y_true, y_pred_zero)
    assert not np.any(np.isnan(grad))
    assert not np.any(np.isinf(grad))


def test_cce_gradiente_analitico_vs_diferencas_finitas():
    """Valida a derivada analítica da CCE contra diferenças finitas centrais."""
    custo = CCE()
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])

    grad_analitico = custo.gradiente(y_true, y_pred)

    eps = 1e-6
    grad_num = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            y_mais = y_pred.copy()
            y_mais[i, j] += eps
            y_menos = y_pred.copy()
            y_menos[i, j] -= eps
            grad_num[i, j] = (
                custo.calcular_custo(y_true, y_mais)
                - custo.calcular_custo(y_true, y_menos)
            ) / (2 * eps)

    np.testing.assert_allclose(grad_analitico, grad_num, rtol=1e-4, atol=1e-5)
