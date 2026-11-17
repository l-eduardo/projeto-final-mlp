import numpy as np
import pytest

from mlp.ativacoes import (
    ELU,
    LeakyReLU,
    Sigmoide,
    Softmax,
    Softplus,
    Softsign,
    Swish,
    Tanh,
)
from mlp.camadas import Densa
from mlp.checagem_gradientes import (
    calcular_erro_relativo,
    checar_gradiente_ativacao,
    checar_gradiente_camada_densa,
    checar_gradiente_rede,
)
from mlp.custos import BCE, CCE, MSE
from mlp.rede import RedeNeural


def test_calcular_erro_relativo_identicos():
    """Valida que gradientes analítico e numérico idênticos produzem erro relativo zero."""
    g = np.array([1.0, 2.0, 3.0])
    assert calcular_erro_relativo(g, g) == 0.0


def test_calcular_erro_relativo_vetores_zerados():
    """Valida a estabilidade numérica contra divisão por zero para vetores nulos."""
    g = np.zeros(5)
    assert calcular_erro_relativo(g, g) == 0.0


def test_calcular_erro_relativo_calculo_manual():
    """Valida o cálculo manual do erro relativo ||ga - gn|| / (||ga|| + ||gn||)."""
    ga = np.array([1.0, 0.0])
    gn = np.array([1.0001, 0.0])
    # ||ga - gn|| = 0.0001, ||ga|| + ||gn|| = 2.0001
    esperado = 0.0001 / 2.0001
    assert pytest.approx(calcular_erro_relativo(ga, gn), rel=1e-5) == esperado


@pytest.mark.parametrize(
    "ativacao",
    [
        Sigmoide(),
        LeakyReLU(alpha=0.01),
        Swish(),
        Softmax(),
        Tanh(),
        ELU(alpha=1.0),
        Softplus(),
        Softsign(),
    ],
)
def test_checar_gradiente_todas_ativacoes(ativacao):
    """Valida a aproximação numérica por diferenças finitas para todas as 8 ativações."""
    rng = np.random.RandomState(42)
    # Evita pontos de não-diferenciabilidade exata em z=0 para LeakyReLU e ELU
    entradas = rng.uniform(0.1, 2.0, size=(3, 4))
    if isinstance(ativacao, (LeakyReLU, ELU)):
        entradas[entradas < 0.05] += 0.1
    gradiente_saida = rng.randn(3, 4)

    erro_relativo = checar_gradiente_ativacao(ativacao, entradas, gradiente_saida)
    assert erro_relativo < 1e-6, (
        f"Ativação {type(ativacao).__name__} falhou na checagem numérica com erro {erro_relativo}"
    )


def test_checar_gradiente_camada_densa():
    """Valida as derivadas analíticas de pesos, vieses e entradas da camada Densa."""
    rng = np.random.RandomState(123)
    camada = Densa(3, 2, gerador_aleatorio=rng)
    entradas = rng.randn(4, 3)
    gradiente_saida = rng.randn(4, 2)

    erros = checar_gradiente_camada_densa(camada, entradas, gradiente_saida)

    assert erros["pesos"] < 1e-7, f"Erro relativo em W muito alto: {erros['pesos']}"
    assert erros["vieses"] < 1e-7, f"Erro relativo em b muito alto: {erros['vieses']}"
    assert erros["entradas"] < 1e-7, (
        f"Erro relativo em X muito alto: {erros['entradas']}"
    )


def test_checar_gradiente_rede_sem_compilar_lanca_excecao():
    """Valida que checar_gradiente_rede exige rede previamente compilada."""
    rede = RedeNeural()
    rede.adicionar(Densa(2, 2))
    with pytest.raises(RuntimeError, match="deve ser compilada antes"):
        checar_gradiente_rede(rede, np.ones((2, 2)), np.ones((2, 2)))


def test_checar_gradiente_rede_sem_camadas_treinaveis():
    """Valida comportamento da checagem em rede com apenas ativações."""
    rede = RedeNeural()
    rede.adicionar(Sigmoide())
    rede.compilar(MSE())
    res = checar_gradiente_rede(rede, np.ones((2, 2)), np.ones((2, 2)))
    assert res["camadas"] == []
    assert res["erro_relativo_maximo"] == 0.0
    assert res["passou_tolerancia"] is True


def test_checar_gradiente_rede_ponta_a_ponta_mse():
    """Valida a checagem numérica de ponta a ponta em rede profunda com MSE."""
    rng = np.random.RandomState(42)
    rede = RedeNeural(semente_aleatoria=42)
    rede.adicionar(Densa(3, 4, gerador_aleatorio=rng))
    rede.adicionar(LeakyReLU())
    rede.adicionar(Densa(4, 2, gerador_aleatorio=rng))
    rede.compilar(MSE())

    X = rng.randn(5, 3)
    y = rng.randn(5, 2)

    relatorio = checar_gradiente_rede(rede, X, y, tolerancia=1e-6)

    assert relatorio["passou_tolerancia"] is True
    assert relatorio["erro_relativo_maximo"] < 1e-6
    for c in relatorio["camadas"]:
        assert c["passou_pesos"] is True
        assert c["passou_vieses"] is True


def test_checar_gradiente_rede_ponta_a_ponta_bce():
    """Valida a checagem numérica de ponta a ponta em rede de classificação binária com BCE."""
    rng = np.random.RandomState(42)
    rede = RedeNeural(semente_aleatoria=42)
    rede.adicionar(Densa(3, 5, gerador_aleatorio=rng))
    rede.adicionar(Swish())
    rede.adicionar(Densa(5, 1, gerador_aleatorio=rng))
    rede.adicionar(Sigmoide())
    rede.compilar(BCE())

    X = rng.randn(6, 3)
    y = rng.choice([0.0, 1.0], size=(6, 1))

    relatorio = checar_gradiente_rede(rede, X, y, tolerancia=1e-6)

    assert relatorio["passou_tolerancia"] is True
    assert relatorio["erro_relativo_maximo"] < 1e-6


def test_checar_gradiente_rede_ponta_a_ponta_cce():
    """Valida a checagem numérica de ponta a ponta em rede multiclasse com CCE."""
    rng = np.random.RandomState(42)
    rede = RedeNeural(semente_aleatoria=42)
    rede.adicionar(Densa(4, 5, gerador_aleatorio=rng))
    rede.adicionar(Tanh())
    rede.adicionar(Densa(5, 3, gerador_aleatorio=rng))
    rede.adicionar(Softmax())
    rede.compilar(CCE())

    X = rng.randn(6, 4)
    # One-hot encoding de 3 classes
    labels = rng.choice([0, 1, 2], size=6)
    y = np.zeros((6, 3))
    y[np.arange(6), labels] = 1.0

    relatorio = checar_gradiente_rede(rede, X, y, tolerancia=1e-6)

    assert relatorio["passou_tolerancia"] is True
    assert relatorio["erro_relativo_maximo"] < 1e-6
