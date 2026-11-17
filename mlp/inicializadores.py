import numpy as np

from mlp.base import Inicializador


class Zeros(Inicializador):
    """Inicializa todos os elementos da matriz com zero.

    Frequentemente utilizado para a inicialização dos vetores de vieses (bias).
    """

    def inicializar(
        self,
        formato: tuple[int, int],
        gerador_aleatorio: np.random.RandomState,
    ) -> np.ndarray:
        """Retorna uma matriz de zeros no formato especificado."""

        zeros = np.zeros(formato, dtype=np.float32)

        return zeros


class GlorotNormal(Inicializador):
    """Inicializador Xavier / Glorot Normal.

    Sorteia valores de uma distribuição normal N(0, desvio_padrao^2), onde:
        desvio_padrao = sqrt(2 / (fan_in + fan_out))
    """

    def inicializar(
        self,
        formato: tuple[int, int],
        gerador_aleatorio: np.random.RandomState,
    ) -> np.ndarray:
        stdDev = np.sqrt(2 / (formato[0] + formato[1]))

        pesos = gerador_aleatorio.normal(loc=0, scale=stdDev, size=formato)

        return pesos


class GlorotUniforme(Inicializador):
    """Inicializador Xavier / Glorot Uniforme.

    Sorteia valores de uma distribuição uniforme U(-limite, limite), onde:
        limite = sqrt(6 / (fan_in + fan_out))
    """

    def inicializar(
        self,
        formato: tuple[int, int],
        gerador_aleatorio: np.random.RandomState,
    ) -> np.ndarray:
        limite = np.sqrt(6 / (formato[0] + formato[1]))

        pesos = gerador_aleatorio.uniform(-limite, limite, formato)

        return pesos


class HeNormal(Inicializador):
    """Inicializador He / Kaiming Normal.

    Sorteia valores de uma distribuição normal N(0, desvio_padrao^2), onde:
        desvio_padrao = sqrt(2 / fan_in)
    """

    def inicializar(
        self,
        formato: tuple[int, int],
        gerador_aleatorio: np.random.RandomState,
    ) -> np.ndarray:
        stdDev = np.sqrt(2 / formato[0])

        pesos = gerador_aleatorio.normal(loc=0, scale=stdDev, size=formato)

        return pesos
