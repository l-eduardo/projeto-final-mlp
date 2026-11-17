import numpy as np
import pytest

from mlp.camadas import Densa
from mlp.inicializadores import GlorotNormal, HeNormal, Zeros


def test_densa_inicializacao_dimensoes_e_vieses():
    """Valida se pesos e vieses são criados com as dimensões corretas e vieses zerados."""
    camada = Densa(dimensao_entrada=4, dimensao_saida=3)
    assert camada.pesos.shape == (4, 3)
    assert camada.vieses.shape == (1, 3)
    np.testing.assert_allclose(camada.vieses, np.zeros((1, 3)))
    assert camada.gradiente_pesos is None
    assert camada.gradiente_vieses is None


def test_densa_reprodutibilidade_com_semente():
    """Valida se o gerador pseudoaleatório garante reprodutibilidade exata dos pesos."""
    c1 = Densa(3, 2, gerador_aleatorio=np.random.RandomState(42))
    c2 = Densa(3, 2, gerador_aleatorio=np.random.RandomState(42))
    np.testing.assert_allclose(c1.pesos, c2.pesos)


def test_densa_inicializadores_customizados():
    """Valida a injeção de diferentes estratégias de inicialização."""
    c_he = Densa(8, 4, inicializador=HeNormal())
    assert c_he.pesos.shape == (8, 4)

    c_glorot = Densa(8, 4, inicializador=GlorotNormal())
    assert c_glorot.pesos.shape == (8, 4)

    c_zeros = Densa(5, 2, inicializador=Zeros())
    np.testing.assert_allclose(c_zeros.pesos, np.zeros((5, 2)))


def test_densa_passo_a_frente_valores_conhecidos():
    """Valida a propagação direta Z = X * W + b em um caso analítico determinístico."""
    camada = Densa(2, 2, inicializador=Zeros())
    camada.pesos = np.array([[1.0, 2.0], [3.0, 4.0]])
    camada.vieses = np.array([[0.5, -0.5]])

    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    # X @ W = [[4.0, 6.0], [2.0, 4.0]] + b = [[4.5, 5.5], [2.5, 3.5]]
    esperado = np.array([[4.5, 5.5], [2.5, 3.5]])

    saida = camada.passo_a_frente(X)
    np.testing.assert_allclose(saida, esperado)


@pytest.mark.parametrize("batch_size", [1, 5, 16, 32])
def test_densa_preservacao_dimensoes_batch(batch_size):
    """Valida que a camada preserva e projeta corretamente para qualquer tamanho de batch."""
    d_in, d_out = 6, 3
    camada = Densa(d_in, d_out)
    X = np.random.randn(batch_size, d_in)

    Z = camada.passo_a_frente(X)
    assert Z.shape == (batch_size, d_out)

    grad_saida = np.ones_like(Z)
    dX = camada.passo_atras(grad_saida)
    assert dX.shape == (batch_size, d_in)
    assert camada.gradiente_pesos.shape == (d_in, d_out)
    assert camada.gradiente_vieses.shape == (1, d_out)


def test_densa_passo_atras_valores_conhecidos():
    """Valida o cálculo exato de dW = X^T * dZ, db = sum(dZ) e dX = dZ * W^T."""
    camada = Densa(2, 2)
    camada.pesos = np.array([[1.0, 0.0], [0.0, 2.0]])
    camada.vieses = np.array([[0.0, 0.0]])

    X = np.array([[2.0, 3.0], [1.0, 4.0]])
    camada.passo_a_frente(X)

    grad_saida = np.array([[1.0, 1.0], [1.0, 2.0]])

    dX = camada.passo_atras(grad_saida)

    # dW = X.T @ grad_saida = [[2, 1], [3, 4]] @ [[1, 1], [1, 2]] = [[3, 4], [7, 11]]
    dW_esperado = np.array([[3.0, 4.0], [7.0, 11.0]])
    np.testing.assert_allclose(camada.gradiente_pesos, dW_esperado)

    # db = sum(grad_saida, axis=0) = [[2.0, 3.0]]
    db_esperado = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(camada.gradiente_vieses, db_esperado)

    # dX = grad_saida @ W.T = [[1, 1], [1, 2]] @ [[1, 0], [0, 2]] = [[1, 2], [1, 4]]
    dX_esperado = np.array([[1.0, 2.0], [1.0, 4.0]])
    np.testing.assert_allclose(dX, dX_esperado)


def test_densa_passo_atras_sem_forward_lanca_excecao():
    """Valida que chamar passo_atras sem executar passo_a_frente levanta RuntimeError."""
    camada = Densa(3, 2)
    with pytest.raises(RuntimeError):
        camada.passo_atras(np.ones((2, 2)))


def test_densa_gradientes_analiticos_vs_diferencas_finitas():
    """Valida rigorosamente os gradientes analíticos (dW, db, dX) contra diferenças finitas."""
    rng = np.random.RandomState(42)
    d_in, d_out, n_samples = 3, 2, 4
    camada = Densa(d_in, d_out, gerador_aleatorio=rng)
    X = rng.randn(n_samples, d_in)

    # Função escalar arbitrária para verificação de gradiente: L = 0.5 * sum(Z^2)
    # dL/dZ = Z
    Z = camada.passo_a_frente(X)
    grad_saida = Z.copy()

    dX_analitico = camada.passo_atras(grad_saida)
    dW_analitico = camada.gradiente_pesos.copy()
    db_analitico = camada.gradiente_vieses.copy()

    eps = 1e-6

    # 1. Checagem numérica de dW
    dW_num = np.zeros_like(camada.pesos)
    for i in range(d_in):
        for j in range(d_out):
            camada.pesos[i, j] += eps
            L_mais = 0.5 * np.sum(camada.passo_a_frente(X) ** 2)
            camada.pesos[i, j] -= 2 * eps
            L_menos = 0.5 * np.sum(camada.passo_a_frente(X) ** 2)
            camada.pesos[i, j] += eps  # restaura
            dW_num[i, j] = (L_mais - L_menos) / (2 * eps)
    np.testing.assert_allclose(dW_analitico, dW_num, rtol=1e-5, atol=1e-6)

    # 2. Checagem numérica de db
    db_num = np.zeros_like(camada.vieses)
    for j in range(d_out):
        camada.vieses[0, j] += eps
        L_mais = 0.5 * np.sum(camada.passo_a_frente(X) ** 2)
        camada.vieses[0, j] -= 2 * eps
        L_menos = 0.5 * np.sum(camada.passo_a_frente(X) ** 2)
        camada.vieses[0, j] += eps  # restaura
        db_num[0, j] = (L_mais - L_menos) / (2 * eps)
    np.testing.assert_allclose(db_analitico, db_num, rtol=1e-5, atol=1e-6)

    # 3. Checagem numérica de dX
    dX_num = np.zeros_like(X)
    for i in range(n_samples):
        for j in range(d_in):
            X_mais = X.copy()
            X_mais[i, j] += eps
            L_mais = 0.5 * np.sum(camada.passo_a_frente(X_mais) ** 2)
            X_menos = X.copy()
            X_menos[i, j] -= eps
            L_menos = 0.5 * np.sum(camada.passo_a_frente(X_menos) ** 2)
            dX_num[i, j] = (L_mais - L_menos) / (2 * eps)
    np.testing.assert_allclose(dX_analitico, dX_num, rtol=1e-5, atol=1e-6)


def test_densa_atualizar_parametros():
    """Valida se a atualização por SGD subtrai corretamente os gradientes multiplicados pela lr."""
    camada = Densa(2, 2)
    with pytest.raises(RuntimeError):
        camada.atualizar_parametros(taxa_aprendizado=0.1)

    camada.pesos = np.array([[2.0, 2.0], [2.0, 2.0]])
    camada.vieses = np.array([[1.0, 1.0]])

    camada.gradiente_pesos = np.array([[1.0, 0.5], [0.5, 1.0]])
    camada.gradiente_vieses = np.array([[0.2, 0.2]])

    camada.atualizar_parametros(taxa_aprendizado=0.1)

    np.testing.assert_allclose(camada.pesos, [[1.9, 1.95], [1.95, 1.9]])
    np.testing.assert_allclose(camada.vieses, [[0.98, 0.98]])
