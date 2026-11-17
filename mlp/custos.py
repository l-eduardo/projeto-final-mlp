import numpy as np

from mlp.base import FuncaoCusto


class MSE(FuncaoCusto):
    """Função de custo Erro Quadrático Médio (Mean Squared Error).

    Utilizada prioritariamente em problemas de regressão linear e não-linear.

    Fórmula matemática:
        L(y, y_hat) = (1 / N) * sum_i ||y_i - y_hat_i||^2

    Gradiente analítico (derivada em relação a y_hat):
        dL / dy_hat = (2 / N) * (y_hat - y)
    """

    def calcular_custo(self, y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> float:
        """Calcula o valor escalar do Erro Quadrático Médio.

        Parâmetros:
            y_verdadeiro (np.ndarray): Valores reais observados (alvos).
            y_predito (np.ndarray): Valores preditos pelo modelo.

        Retorna:
            float: Valor escalar médio da perda quadrática.
        """
        n_amostras = y_predito.shape[0] if y_predito.ndim > 0 else 1
        diferenca = y_predito - y_verdadeiro
        return float(np.sum(diferenca**2) / n_amostras)

    def gradiente(self, y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> np.ndarray:
        """Calcula a derivada analítica do MSE em relação às predições (dL/dy_predito).

        Parâmetros:
            y_verdadeiro (np.ndarray): Valores reais observados (alvos).
            y_predito (np.ndarray): Valores preditos pelo modelo.

        Retorna:
            np.ndarray: Matriz de gradientes com as mesmas dimensões de y_predito.
        """
        n_amostras = y_predito.shape[0] if y_predito.ndim > 0 else 1
        return (2.0 / n_amostras) * (y_predito - y_verdadeiro)


class BCE(FuncaoCusto):
    """Função de custo Entropia Cruzada Binária (Binary Cross-Entropy).

    Utilizada em tarefas de classificação binária, tipicamente combinada com
    uma camada de saída com ativação Sigmoide.

    Fórmula matemática:
        L(y, y_hat) = -(1 / N) * sum_i [ y_i * ln(y_hat_i) + (1 - y_i) * ln(1 - y_hat_i) ]

    Gradiente analítico (derivada em relação a y_hat):
        dL / dy_hat = (1 / N) * [ (y_hat - y) / (y_hat * (1 - y_hat)) ]

    Nota de estabilidade numérica:
        As predições são truncadas no intervalo [eps, 1 - eps] para prevenir
        avaliação de ln(0) e divisões por zero.
    """

    def __init__(self, eps: float = 1e-15) -> None:
        """Inicializa a função de custo com tolerância para estabilidade numérica.

        Parâmetros:
            eps (float): Limite inferior e superior de saturação. Padrão: 1e-15.
        """
        self.eps = float(eps)

    def calcular_custo(self, y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> float:
        """Calcula o valor escalar da perda de entropia cruzada binária.

        Parâmetros:
            y_verdadeiro (np.ndarray): Alvos binários (0 ou 1).
            y_predito (np.ndarray): Probabilidades estimadas no intervalo (0, 1).

        Retorna:
            float: Valor escalar médio da perda BCE.
        """
        n_amostras = y_predito.shape[0] if y_predito.ndim > 0 else 1
        y_hat_estavel = np.clip(y_predito, self.eps, 1.0 - self.eps)
        perda = -(
            y_verdadeiro * np.log(y_hat_estavel)
            + (1.0 - y_verdadeiro) * np.log(1.0 - y_hat_estavel)
        )
        return float(np.sum(perda) / n_amostras)

    def gradiente(self, y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> np.ndarray:
        """Calcula o gradiente da perda BCE em relação às predições (dL/dy_predito).

        Parâmetros:
            y_verdadeiro (np.ndarray): Alvos binários (0 ou 1).
            y_predito (np.ndarray): Probabilidades estimadas no intervalo (0, 1).

        Retorna:
            np.ndarray: Gradiente com o mesmo formato de y_predito.
        """
        n_amostras = y_predito.shape[0] if y_predito.ndim > 0 else 1
        y_hat_estavel = np.clip(y_predito, self.eps, 1.0 - self.eps)
        grad = (y_hat_estavel - y_verdadeiro) / (y_hat_estavel * (1.0 - y_hat_estavel))
        return grad / n_amostras


class CCE(FuncaoCusto):
    """Função de custo Entropia Cruzada Categórica (Categorical Cross-Entropy).

    Utilizada em tarefas de classificação multiclasse (C >= 2 classes),
    tipicamente combinada com uma camada de saída com ativação Softmax.

    Fórmula matemática (com alvos one-hot):
        L(y, y_hat) = -(1 / N) * sum_i sum_c y_{ic} * ln(y_hat_{ic})

    Gradiente analítico (derivada em relação a y_hat):
        dL / dy_hat_{ic} = -(1 / N) * (y_{ic} / y_hat_{ic})

    Nota de estabilidade numérica:
        Predições y_hat são truncadas com limite inferior eps para evitar ln(0).
    """

    def __init__(self, eps: float = 1e-15) -> None:
        """Inicializa a função de custo com tolerância para estabilidade numérica.

        Parâmetros:
            eps (float): Limite inferior para predições de probabilidade. Padrão: 1e-15.
        """
        self.eps = float(eps)

    def _preparar_alvos(self, y_verdadeiro: np.ndarray, n_classes: int) -> np.ndarray:
        """Normaliza os alvos para formato one-hot se fornecidos como vetor de rótulos inteiros."""
        if (
            y_verdadeiro.ndim == 1
            or (y_verdadeiro.ndim == 2 and y_verdadeiro.shape[1] == 1)
        ) and np.issubdtype(y_verdadeiro.dtype, np.integer):
            indices = y_verdadeiro.ravel()
            one_hot = np.zeros((indices.shape[0], n_classes), dtype=np.float64)
            one_hot[np.arange(indices.shape[0]), indices] = 1.0
            return one_hot
        return y_verdadeiro.astype(np.float64)

    def calcular_custo(self, y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> float:
        """Calcula a perda média de entropia cruzada categórica.

        Parâmetros:
            y_verdadeiro (np.ndarray): Rótulos de classe (one-hot ou índices inteiros).
            y_predito (np.ndarray): Distribuição de probabilidade por amostra (Softmax).

        Retorna:
            float: Valor escalar médio da perda CCE.
        """
        n_amostras, n_classes = y_predito.shape
        y_true = self._preparar_alvos(y_verdadeiro, n_classes)
        y_hat_estavel = np.clip(y_predito, self.eps, 1.0)
        perda = -np.sum(y_true * np.log(y_hat_estavel))
        return float(perda / n_amostras)

    def gradiente(self, y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> np.ndarray:
        """Calcula a derivada da CCE em relação às probabilidades estimadas (dL/dy_predito).

        Parâmetros:
            y_verdadeiro (np.ndarray): Rótulos de classe (one-hot ou índices inteiros).
            y_predito (np.ndarray): Distribuição de probabilidade predita pelo modelo.

        Retorna:
            np.ndarray: Gradiente com o mesmo formato de y_predito.
        """
        n_amostras, n_classes = y_predito.shape
        y_true = self._preparar_alvos(y_verdadeiro, n_classes)
        y_hat_estavel = np.clip(y_predito, self.eps, 1.0)
        return -(y_true / y_hat_estavel) / n_amostras
