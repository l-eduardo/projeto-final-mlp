from abc import ABC, abstractmethod
import numpy as np


class Camada(ABC):
    """Interface abstrata base para todas as camadas da rede neural."""

    @abstractmethod
    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        """Calcula a propagação direta (forward) da camada."""
        pass

    @abstractmethod
    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Calcula a retropropagação (backward) e os gradientes dos parâmetros."""
        pass


class Ativacao(ABC):
    """Interface abstrata base para funções de ativação."""

    @abstractmethod
    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        """Aplica a função de ativação f(x)."""
        pass

    @abstractmethod
    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Calcula a derivada dE/dX = dE/dY * f'(x)."""
        pass


class FuncaoCusto(ABC):
    """Interface abstrata base para funções de custo/perda."""

    @abstractmethod
    def calcular_custo(
        self, y_verdadeiro: np.ndarray, y_predito: np.ndarray
    ) -> float:
        """Calcula o valor escalar do custo (loss)."""
        pass

    @abstractmethod
    def gradiente(
        self, y_verdadeiro: np.ndarray, y_predito: np.ndarray
    ) -> np.ndarray:
        """Calcula a derivada da função de custo em relação a y_predito (dE/dy_predito)."""
        pass


class Inicializador(ABC):
    """Interface abstrata base para inicializadores de pesos."""

    @abstractmethod
    def inicializar(
        self,
        formato: tuple[int, int],
        gerador_aleatorio: np.random.RandomState,
    ) -> np.ndarray:
        """Retorna uma matriz de pesos com as dimensões especificadas."""
        pass