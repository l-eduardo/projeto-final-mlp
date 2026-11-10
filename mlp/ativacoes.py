import numpy as np

from mlp.base import Ativacao


class Sigmoide(Ativacao):
    """Função de ativação Sigmoide (Logística).

    Fórmula matemática:
        f(z) = 1 / (1 + exp(-z))

    Derivada analítica:
        f'(z) = f(z) * (1 - f(z))

    Nota de estabilidade numérica:
        Para evitar overflow em exp(-z) com valores de z muito negativos,
        recomenda-se tratar os ramos z >= 0 e z < 0 ou aplicar clipping.
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        """Calcula a propagação direta da função Sigmoide.

        Parâmetros:
            entradas (np.ndarray): Matriz de entradas (pré-ativações Z).

        Retorna:
            np.ndarray: Ativações calculadas A = f(Z).
        """
        raise NotImplementedError("A ser implementado na Issue #7")

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Calcula a retropropagação da Sigmoide aplicando a regra da cadeia.

        Parâmetros:
            gradiente_saida (np.ndarray): dL/dA recebido da camada posterior.

        Retorna:
            np.ndarray: Gradiente em relação às entradas dL/dZ = dL/dA * f'(Z).
        """
        raise NotImplementedError("A ser implementado na Issue #7")


class ReLU(Ativacao):
    """Função de ativação ReLU (Rectified Linear Unit).

    Fórmula matemática:
        f(z) = max(0, z)

    Derivada analítica:
        f'(z) = 1 se z > 0, caso contrário 0 (ou subgradiente 0 em z = 0).
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        """Calcula a propagação direta da função ReLU.

        Parâmetros:
            entradas (np.ndarray): Matriz de entradas (pré-ativações Z).

        Retorna:
            np.ndarray: Ativações calculadas A = max(0, Z).
        """
        raise NotImplementedError("A ser implementado na Issue #7")

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Calcula a retropropagação da ReLU aplicando a regra da cadeia.

        Parâmetros:
            gradiente_saida (np.ndarray): dL/dA recebido da camada posterior.

        Retorna:
            np.ndarray: Gradiente em relação às entradas dL/dZ = dL/dA * f'(Z).
        """
        raise NotImplementedError("A ser implementado na Issue #7")


class LeakyReLU(Ativacao):
    """Função de ativação Leaky ReLU.

    Fórmula matemática:
        f(z) = z se z > 0, caso contrário alpha * z

    Derivada analítica:
        f'(z) = 1 se z > 0, caso contrário alpha.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """Inicializa a LeakyReLU com o fator de inclinação alpha para valores negativos.

        Parâmetros:
            alpha (float): Inclinação para z < 0. Padrão: 0.01.
        """
        self.alpha = float(alpha)

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        """Calcula a propagação direta da função LeakyReLU.

        Parâmetros:
            entradas (np.ndarray): Matriz de entradas (pré-ativações Z).

        Retorna:
            np.ndarray: Ativações calculadas A = f(Z).
        """
        raise NotImplementedError("A ser implementado na Issue #7")

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Calcula a retropropagação da LeakyReLU aplicando a regra da cadeia.

        Parâmetros:
            gradiente_saida (np.ndarray): dL/dA recebido da camada posterior.

        Retorna:
            np.ndarray: Gradiente em relação às entradas dL/dZ = dL/dA * f'(Z).
        """
        raise NotImplementedError("A ser implementado na Issue #7")


class Tanh(Ativacao):
    """Função de ativação Tangente Hiperbólica (Tanh) analítica.

    ATENÇÃO - REQUISITO DE NOTA ZERO:
        É terminantemente proibido o uso de `np.tanh`.
        O cálculo deve ser analítico a partir de exponenciais elementares:
        tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        ou equivalentemente:
        tanh(z) = 2 / (1 + exp(-2z)) - 1

    Derivada analítica:
        f'(z) = 1 - (tanh(z))^2
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        """Calcula a propagação direta da função Tanh analítica.

        Parâmetros:
            entradas (np.ndarray): Matriz de entradas (pré-ativações Z).

        Retorna:
            np.ndarray: Ativações calculadas A = tanh(Z).
        """
        raise NotImplementedError("A ser implementado na Issue #7")

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Calcula a retropropagação da Tanh aplicando a regra da cadeia.

        Parâmetros:
            gradiente_saida (np.ndarray): dL/dA recebido da camada posterior.

        Retorna:
            np.ndarray: Gradiente em relação às entradas dL/dZ = dL/dA * f'(Z).
        """
        raise NotImplementedError("A ser implementado na Issue #7")
