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


class Tanh(Ativacao):
    """Função de ativação Tangente Hiperbólica (Tanh) analítica.

    ATENÇÃO - REQUISITO DE NOTA ZERO:
        É terminantemente proibido o uso de `np.tanh`.
        O cálculo deve ser analítico a partir de formulações elementares:
        tanh(z) = 2 * sigmoide(2z) - 1

    Derivada analítica:
        f'(z) = 1 - (tanh(z))^2
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        self.ativacoes = 2 * sigmoide(2 * entradas) - 1
        return self.ativacoes

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        return gradiente_saida * (1 - self.ativacoes**2)


class ELU(Ativacao):
    """Função de ativação Exponential Linear Unit (ELU).

    Fórmula matemática:
        f(z) = z se z > 0, caso contrário alpha * (exp(z) - 1)

    Derivada analítica:
        f'(z) = 1 se z > 0, caso contrário alpha * exp(z) = f(z) + alpha
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Inicializa a ELU com o parâmetro de escala alpha para valores negativos.

        Parâmetros:
            alpha (float): Fator de escala para z <= 0. Padrão: 1.0.
        """
        self.alpha = float(alpha)

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        self.entradas = entradas
        self.ativacoes = np.where(
            entradas > 0, entradas, self.alpha * np.expm1(np.minimum(entradas, 0))
        )
        return self.ativacoes

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        derivada = np.where(self.entradas > 0, 1.0, self.ativacoes + self.alpha)
        return gradiente_saida * derivada


class Softplus(Ativacao):
    """Função de ativação Softplus.

    Fórmula matemática:
        f(z) = ln(1 + exp(z))

    Derivada analítica:
        f'(z) = sigmoide(z)

    Nota de estabilidade numérica:
        Calculada como max(z, 0) + ln(1 + exp(-|z|)) via log1p para prevenir
        overflow com entradas z >> 0.
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        self.entradas = entradas
        self.ativacoes = np.maximum(entradas, 0) + np.log1p(np.exp(-np.abs(entradas)))
        return self.ativacoes

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        return gradiente_saida * sigmoide(self.entradas)


class Softsign(Ativacao):
    """Função de ativação Softsign.

    Fórmula matemática:
        f(z) = z / (1 + |z|)

    Derivada analítica:
        f'(z) = 1 / (1 + |z|)^2
    """

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        self.entradas = entradas
        self.ativacoes = entradas / (1 + np.abs(entradas))
        return self.ativacoes

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        derivada = 1.0 / ((1 + np.abs(self.entradas)) ** 2)
        return gradiente_saida * derivada
