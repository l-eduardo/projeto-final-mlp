import numpy as np

from mlp.base import Ativacao


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Calcula a função sigmoide de uma matriz de entradas z de forma numericamente estável.

    Este método foi criado como função utilitária pois a sigmoide é utilizada em
    diversas funções de ativação e suas respectivas derivadas analíticas.
    """
    exp_neg_abs = np.exp(-np.abs(z))
    return np.where(
        z >= 0,
        1 / (1 + exp_neg_abs),  # Para z >= 0: 1 / (1 + exp(-z))
        exp_neg_abs / (1 + exp_neg_abs),  # Para z < 0: exp(z) / (1 + exp(z))
    )


class Sigmoide(Ativacao):
    """Função de ativação Sigmoide (Logística).

    Fórmula matemática:
        f(z) = 1 / (1 + exp(-z))

    Derivada analítica:
        f'(z) = f(z) * (1 - f(z))

    Nota de estabilidade numérica:
        Calculada de forma estável decompondo em z >= 0 e z < 0 sem saturação por overflow.
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        self.ativacoes = sigmoide(entradas)
        return self.ativacoes

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
        self.entradas = entradas
        return np.where(entradas > 0, entradas, self.alpha * entradas)

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        return np.where(
            self.entradas > 0, gradiente_saida, self.alpha * gradiente_saida
        )


class Swish(Ativacao):
    """Função de ativação Swish (SiLU — Sigmoid Linear Unit).

    Fórmula matemática:
        f(z) = z * sigmoide(z), onde sigmoide(z) = 1 / (1 + exp(-z))

    Derivada analítica:
        f'(z) = sigmoide(z) + z * sigmoide(z) * (1 - sigmoide(z))
        ou equivalentemente:
        f'(z) = f(z) + sigmoide(z) * (1 - f(z))

    Nota de estabilidade numérica:
        Calculada com sigmoide estável sem risco de overflow em exp(-z).
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        self.sigmoide = sigmoide(entradas)
        self.ativacoes = entradas * self.sigmoide
        return self.ativacoes

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        derivada = self.ativacoes + self.sigmoide * (1 - self.ativacoes)
        return gradiente_saida * derivada


class Softmax(Ativacao):
    """Função de ativação Softmax para classificação multiclasse.

    Fórmula matemática:
        S(z_i) = exp(z_i - max(z)) / sum_j(exp(z_j - max(z)))

    Derivada analítica (produto vetor-Jacobiana por amostra):
        dL/dz_i = S_i * (dL/dS_i - sum_j(dL/dS_j * S_j))
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        z_estavel = entradas - np.max(entradas, axis=-1, keepdims=True)
        exps = np.exp(z_estavel)
        self.ativacoes = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.ativacoes

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        soma = np.sum(gradiente_saida * self.ativacoes, axis=-1, keepdims=True)
        return self.ativacoes * (gradiente_saida - soma)
