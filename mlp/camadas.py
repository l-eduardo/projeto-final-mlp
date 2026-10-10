import numpy as np
from mlp.base import Camada, Inicializador


class Densa(Camada):
    """Camada totalmente conectada (linear)."""

    def __init__(
        self,
        dimensao_entrada: int,
        dimensao_saida: int,
        inicializador: Inicializador | None = None,
        gerador_aleatorio: np.random.RandomState | None = None,
    ):
        super().__init__()
        raise NotImplementedError

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        raise NotImplementedError