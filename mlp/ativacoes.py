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
        cliped_entradas = np.clip(entradas, -500, 500)

        output = 1 / (1 + np.exp(-cliped_entradas))

        self.ativacoes = output  # Armazena as ativações para uso na retropropagação
        return output

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        return gradiente_saida * self.ativacoes * (1 - self.ativacoes)


class LeakyReLU(Ativacao):
    """Função de ativação Leaky ReLU.

    Nota:
        Para utilizar a ReLU padrão (f(z) = max(0, z)), basta instanciar
        com `alpha=0`.

    Fórmula matemática:
        f(z) = z se z > 0, caso contrário alpha * z

    Derivada analítica:
        f'(z) = 1 se z > 0, caso contrário alpha (ou 0 em z = 0 quando alpha = 0).
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """Inicializa a LeakyReLU com o fator de inclinação alpha para valores negativos.

        Parâmetros:
            alpha (float): Inclinação para z <= 0. Use 0.0 para ReLU padrão. Padrão: 0.01.
        """
        self.alpha = float(alpha)

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        """Calcula a propagação direta da função LeakyReLU.

        Parâmetros:
            entradas (np.ndarray): Matriz de entradas (pré-ativações Z).

        Retorna:
            np.ndarray: Ativações calculadas A = f(Z).
        """
        self.entradas = entradas
        return np.where(entradas > 0, entradas, self.alpha * entradas)

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        """Calcula a retropropagação da LeakyReLU aplicando a regra da cadeia.

        Parâmetros:
            gradiente_saida (np.ndarray): dL/dA recebido da camada posterior.

        Retorna:
            np.ndarray: Gradiente em relação às entradas dL/dZ = dL/dA * f'(Z).
        """
        return np.where(
            self.entradas > 0, gradiente_saida, self.alpha * gradiente_saida
        )


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
