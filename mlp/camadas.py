import numpy as np

from mlp.base import Camada, Inicializador
from mlp.inicializadores import GlorotUniforme, Zeros


class Densa(Camada):
    """Camada totalmente conectada (linear).

    Realiza a transformação afim Z = X * W + b.

    Parâmetros:
        dimensao_entrada (int): Número de características (features) da entrada.
        dimensao_saida (int): Número de neurônios da camada.
        inicializador (Inicializador | None): Estratégia de inicialização para os pesos.
            Caso seja None, utiliza GlorotUniforme por padrão.
        gerador_aleatorio (np.random.RandomState | None): Gerador de números pseudoaleatórios
            para reprodutibilidade. Caso seja None, utiliza uma instância padrão.
    """

    def __init__(
        self,
        dimensao_entrada: int,
        dimensao_saida: int,
        inicializador: Inicializador | None = None,
        gerador_aleatorio: np.random.RandomState | None = None,
    ):
        super().__init__()
        self.dimensao_entrada = int(dimensao_entrada)
        self.dimensao_saida = int(dimensao_saida)

        self.rng = (
            gerador_aleatorio
            if gerador_aleatorio is not None
            else np.random.RandomState()
        )
        self.inicializador = (
            inicializador if inicializador is not None else GlorotUniforme()
        )

        formato_pesos = (self.dimensao_entrada, self.dimensao_saida)
        self.pesos = self.inicializador.inicializar(formato_pesos, self.rng)
        self.vieses = Zeros().inicializar((1, self.dimensao_saida), self.rng)

        self.entradas: np.ndarray | None = None
        self.gradiente_pesos: np.ndarray | None = None
        self.gradiente_vieses: np.ndarray | None = None

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        """Calcula a propagação direta Z = X * W + b.

        Parâmetros:
            entradas (np.ndarray): Matriz de entradas X de formato (N, dimensao_entrada).

        Retorna:
            np.ndarray: Matriz de pré-ativações Z de formato (N, dimensao_saida).
        """
        self.entradas = entradas
        return entradas @ self.pesos + self.vieses

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Calcula a retropropagação analítica através da camada linear.

        Parâmetros:
            gradiente_saida (np.ndarray): dL/dZ recebido da camada posterior, de formato (N, dimensao_saida).

        Retorna:
            np.ndarray: Gradiente em relação às entradas dL/dX = dL/dZ * W^T, de formato (N, dimensao_entrada).
        """
        if self.entradas is None:
            raise RuntimeError(
                "O passo_a_frente deve ser executado antes do passo_atras."
            )

        self.gradiente_pesos = self.entradas.T @ gradiente_saida
        self.gradiente_vieses = np.sum(gradiente_saida, axis=0, keepdims=True)
        return gradiente_saida @ self.pesos.T

    def atualizar_parametros(self, taxa_aprendizado: float) -> None:
        """Atualiza pesos e vieses usando a regra básica da descida do gradiente."""
        if self.gradiente_pesos is None or self.gradiente_vieses is None:
            raise RuntimeError(
                "Gradientes não calculados. Execute passo_atras primeiro."
            )
        self.pesos -= taxa_aprendizado * self.gradiente_pesos
        self.vieses -= taxa_aprendizado * self.gradiente_vieses
