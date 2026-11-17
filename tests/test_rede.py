import numpy as np
import pytest

from mlp.ativacoes import Sigmoide
from mlp.camadas import Densa
from mlp.custos import MSE
from mlp.rede import RedeNeural


def test_rede_inicializacao_padrao():
    """Valida o estado inicial padrão de uma nova instância de RedeNeural."""
    rede = RedeNeural()
    assert rede.camadas == []
    assert rede.funcao_custo is None
    assert rede.semente_aleatoria is None
    assert isinstance(rede.gerador_aleatorio, np.random.RandomState)


def test_rede_inicializacao_com_semente():
    """Valida se a semente aleatória é registrada e inicializa o gerador."""
    rede = RedeNeural(semente_aleatoria=42)
    assert rede.semente_aleatoria == 42
    assert rede.gerador_aleatorio is not None


def test_rede_adicionar_camadas_validas():
    """Valida a adição sequencial de camadas densas e funções de ativação."""
    rede = RedeNeural()
    camada_densa = Densa(3, 4)
    ativacao = Sigmoide()

    rede.adicionar(camada_densa)
    rede.adicionar(ativacao)

    assert len(rede.camadas) == 2
    assert rede.camadas[0] is camada_densa
    assert rede.camadas[1] is ativacao


def test_rede_adicionar_objeto_invalido_lanca_excecao():
    """Valida que adicionar objeto que não seja Camada ou Ativacao levanta TypeError."""
    rede = RedeNeural()
    with pytest.raises(TypeError, match="A camada deve ser uma instância"):
        rede.adicionar("camada_invalida")  # type: ignore[arg-type]


def test_rede_compilar_valida():
    """Valida a configuração da função de custo."""
    rede = RedeNeural()
    custo = MSE()
    rede.compilar(custo)
    assert rede.funcao_custo is custo


def test_rede_compilar_objeto_invalido_lanca_excecao():
    """Valida que compilar com objeto não herdado de FuncaoCusto levanta TypeError."""
    rede = RedeNeural()
    with pytest.raises(TypeError, match="A função de custo deve ser uma instância"):
        rede.compilar("custo_invalido")  # type: ignore[arg-type]


def test_rede_passo_a_frente_sem_camadas_lanca_excecao():
    """Valida que chamar passo_a_frente sem camadas levanta RuntimeError."""
    rede = RedeNeural()
    with pytest.raises(RuntimeError, match="Nenhuma camada foi adicionada"):
        rede.passo_a_frente(np.array([[1.0, 2.0]]))


def test_rede_passo_atras_sem_camadas_lanca_excecao():
    """Valida que chamar passo_atras sem camadas levanta RuntimeError."""
    rede = RedeNeural()
    with pytest.raises(RuntimeError, match="Nenhuma camada foi adicionada"):
        rede.passo_atras(np.array([[1.0, 2.0]]))


def test_rede_fluxo_feedforward_e_predizer():
    """Valida se passo_a_frente e predizer produzem os mesmos resultados e formatos esperados."""
    rng = np.random.RandomState(42)
    rede = RedeNeural(semente_aleatoria=42)
    rede.adicionar(Densa(4, 3, gerador_aleatorio=rng))
    rede.adicionar(Sigmoide())
    rede.adicionar(Densa(3, 1, gerador_aleatorio=rng))

    X = rng.randn(5, 4)
    saida_forward = rede.passo_a_frente(X)
    saida_predizer = rede.predizer(X)

    assert saida_forward.shape == (5, 1)
    np.testing.assert_allclose(saida_forward, saida_predizer)


def test_rede_treinar_sem_compilar_lanca_excecao():
    """Valida que treinar sem compilar levanta RuntimeError."""
    rede = RedeNeural()
    rede.adicionar(Densa(2, 2))
    X = np.ones((10, 2))
    y = np.ones((10, 2))

    with pytest.raises(RuntimeError, match="Rede não compilada"):
        rede.treinar(X, y, epocas=5, taxa_aprendizagem=0.01)


def test_rede_treinar_sem_camadas_lanca_excecao():
    """Valida que treinar sem adicionar camadas levanta RuntimeError."""
    rede = RedeNeural()
    rede.compilar(MSE())
    X = np.ones((10, 2))
    y = np.ones((10, 2))

    with pytest.raises(RuntimeError, match="Nenhuma camada foi adicionada"):
        rede.treinar(X, y, epocas=5, taxa_aprendizagem=0.01)


def test_rede_treinar_dimensoes_incompativeis_lanca_excecao():
    """Valida validações de formatos e integridade dos dados de treino e validação."""
    rede = RedeNeural()
    rede.adicionar(Densa(2, 2))
    rede.compilar(MSE())

    X = np.ones((10, 2))
    y_incompativel = np.ones((8, 2))

    with pytest.raises(ValueError, match="Incompatibilidade no número de amostras:"):
        rede.treinar(X, y_incompativel, epocas=5, taxa_aprendizagem=0.01)

    y = np.ones((10, 2))
    X_val = np.ones((5, 2))
    # Apenas X_val fornecido sem y_val
    with pytest.raises(
        ValueError, match="X_validacao e y_validacao devem ser fornecidos"
    ):
        rede.treinar(X, y, epocas=5, taxa_aprendizagem=0.01, X_validacao=X_val)

    # X_val e y_val com tamanhos diferentes
    y_val_incompativel = np.ones((4, 2))
    with pytest.raises(
        ValueError, match="Incompatibilidade no número de amostras de validação"
    ):
        rede.treinar(
            X,
            y,
            epocas=5,
            taxa_aprendizagem=0.01,
            X_validacao=X_val,
            y_validacao=y_val_incompativel,
        )


@pytest.mark.parametrize(
    "epocas, taxa, lote",
    [
        (0, 0.01, 32),
        (-5, 0.01, 32),
        (10, 0.0, 32),
        (10, -0.05, 32),
        (10, 0.01, 0),
        (10, 0.01, -16),
    ],
)
def test_rede_treinar_parametros_invalidos_lanca_excecao(epocas, taxa, lote):
    """Valida que parâmetros numéricos inválidos de treinamento levantam ValueError."""
    rede = RedeNeural()
    rede.adicionar(Densa(2, 2))
    rede.compilar(MSE())

    X = np.ones((10, 2))
    y = np.ones((10, 2))

    with pytest.raises(ValueError):
        rede.treinar(X, y, epocas=epocas, taxa_aprendizagem=taxa, tamanho_lote=lote)


def test_rede_treinar_convergencia_e_reducao_de_custo():
    """Valida que o laço de treinamento com Mini-batch SGD reduz o custo efetivamente."""
    rng = np.random.RandomState(42)
    rede = RedeNeural(semente_aleatoria=42)
    rede.adicionar(Densa(2, 1, gerador_aleatorio=rng))
    rede.compilar(MSE())

    # Dados sintéticos linearmente separáveis/ajustáveis: y = 2*x1 - x2 + 0.5
    X = rng.uniform(-1, 1, size=(60, 2))
    y = 2.0 * X[:, 0:1] - 1.0 * X[:, 1:2] + 0.5

    historico = rede.treinar(
        X,
        y,
        epocas=40,
        taxa_aprendizagem=0.1,
        tamanho_lote=15,
        verbal=False,
    )

    assert "custo_treino" in historico
    assert len(historico["custo_treino"]) == 40
    # O custo final deve ser significativamente menor que o custo inicial
    assert historico["custo_treino"][-1] < historico["custo_treino"][0] * 0.1


def test_rede_treinar_com_validacao_e_verbal(capsys):
    """Valida o registro de custo de validação e a saída de texto quando verbal=True."""
    rng = np.random.RandomState(42)
    rede = RedeNeural(semente_aleatoria=42)
    rede.adicionar(Densa(2, 1, gerador_aleatorio=rng))
    rede.compilar(MSE())

    X_tr = rng.uniform(-1, 1, size=(40, 2))
    y_tr = 2.0 * X_tr[:, 0:1] + 1.0
    X_val = rng.uniform(-1, 1, size=(10, 2))
    y_val = 2.0 * X_val[:, 0:1] + 1.0

    historico = rede.treinar(
        X_tr,
        y_tr,
        epocas=5,
        taxa_aprendizagem=0.05,
        tamanho_lote=10,
        X_validacao=X_val,
        y_validacao=y_val,
        verbal=True,
    )

    assert "custo_validacao" in historico
    assert len(historico["custo_validacao"]) == 5

    captured = capsys.readouterr()
    assert "Época 1/5" in captured.out
    assert "Custo Validação" in captured.out


def test_rede_reprodutibilidade_com_semente():
    """Valida que duas redes com a mesma semente geram histórico de treinamento idêntico."""
    X = np.random.RandomState(0).randn(30, 2)
    y = np.random.RandomState(0).randn(30, 1)

    r1 = RedeNeural(semente_aleatoria=99)
    r1.adicionar(Densa(2, 1, gerador_aleatorio=np.random.RandomState(99)))
    r1.compilar(MSE())
    h1 = r1.treinar(
        X, y, epocas=10, taxa_aprendizagem=0.05, tamanho_lote=10, verbal=False
    )

    r2 = RedeNeural(semente_aleatoria=99)
    r2.adicionar(Densa(2, 1, gerador_aleatorio=np.random.RandomState(99)))
    r2.compilar(MSE())
    h2 = r2.treinar(
        X, y, epocas=10, taxa_aprendizagem=0.05, tamanho_lote=10, verbal=False
    )

    np.testing.assert_allclose(h1["custo_treino"], h2["custo_treino"])
    np.testing.assert_allclose(r1.camadas[0].pesos, r2.camadas[0].pesos)
