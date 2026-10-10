import numpy as np
from mlp.base import Ativacao, Camada, FuncaoCusto


class RedeNeural:
    """Estrutura principal para encadeamento, treinamento e inferência da MLP."""

    def __init__(self, semente_aleatoria: int | None = None):
        self.camadas: list[Camada | Ativacao] = []
        self.funcao_custo: FuncaoCusto | None = None
        self.semente_aleatoria = semente_aleatoria
        self.gerador_aleatorio = np.random.RandomState(semente_aleatoria)

    def adicionar(self, camada: Camada | Ativacao) -> None:
        """Adiciona uma camada ou função de ativação à rede."""
        raise NotImplementedError

    def compilar(self, funcao_custo: FuncaoCusto) -> None:
        """Define a função de custo a ser utilizada no treinamento."""
        raise NotImplementedError

    def passo_a_frente(self, X: np.ndarray) -> np.ndarray:
        """Realiza a propagação direta através de todas as camadas da rede."""
        raise NotImplementedError

    def passo_atras(self, gradiente_saida: np.ndarray) -> None:
        """Realiza a retropropagação do gradiente do fim para o início da rede."""
        raise NotImplementedError

    def treinar(
        self,
        X_treino: np.ndarray,
        y_treino: np.ndarray,
        epocas: int,
        taxa_aprendizagem: float,
        tamanho_lote: int = 32,
        X_validacao: np.ndarray | None = None,
        y_validacao: np.ndarray | None = None,
        verbal: bool = True,
    ) -> dict[str, list[float]]:
        """Executa o laço de treinamento (Mini-batch SGD) e retorna o histórico de custo."""
        raise NotImplementedError

    def predizer(self, X: np.ndarray) -> np.ndarray:
        """Realiza predições para novas observações."""
        raise NotImplementedError