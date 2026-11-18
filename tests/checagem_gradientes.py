from typing import Any

import numpy as np

from mlp.base import Ativacao
from mlp.camadas import Densa
from mlp.rede import RedeNeural

# IA Based Doc

def calcular_erro_relativo(
    gradiente_analitico: np.ndarray,
    gradiente_numerico: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Calcula o erro relativo L2 entre gradiente analítico e aproximação numérica.

    Fórmula adotada pelo padrão da literatura (CS231n / Stanford):
        Erro Relativo = ||ga - gn||_2 / max(||ga||_2 + ||gn||_2, eps)

    Parâmetros:
        gradiente_analitico (np.ndarray): Tensor de gradientes analíticos.
        gradiente_numerico (np.ndarray): Tensor de gradientes por diferenças finitas.
        eps (float): Termo infinitesimal para estabilidade numérica e evitar divisão por zero.

    Retorna:
        float: Valor escalar do erro relativo adimensional.
    """
    diferenca_norma = float(np.linalg.norm(gradiente_analitico - gradiente_numerico))
    soma_normas = float(
        np.linalg.norm(gradiente_analitico) + np.linalg.norm(gradiente_numerico)
    )
    return diferenca_norma / max(soma_normas, eps)


def checar_gradiente_ativacao(
    ativacao: Ativacao,
    entradas: np.ndarray,
    gradiente_saida: np.ndarray,
    epsilon: float = 1e-7,
) -> float:
    """Valida o gradiente analítico de uma função de ativação contra diferenças finitas.

    Considera uma função objetivo escalar L = sum(A * gradiente_saida), cuja derivada
    em relação a entradas é analiticamente dada por ativacao.passo_atras(gradiente_saida).

    Parâmetros:
        ativacao (Ativacao): Instância da função de ativação a ser testada.
        entradas (np.ndarray): Matriz de entradas (pré-ativações Z) de formato (N, d).
        gradiente_saida (np.ndarray): Matriz de gradientes à montante de formato (N, d).
        epsilon (float): Perturbação infinitesimal para as diferenças finitas centradas.

    Retorna:
        float: Erro relativo entre o gradiente analítico e o numérico.
    """
    ativacao.passo_a_frente(entradas)
    grad_analitico = ativacao.passo_atras(gradiente_saida)

    grad_numerico = np.zeros_like(entradas)
    it = np.nditer(entradas, flags=["multi_index"])

    while not it.finished:
        idx = it.multi_index
        val_original = entradas[idx]

        entradas[idx] = val_original + epsilon
        saida_mais = ativacao.passo_a_frente(entradas)
        loss_mais = float(np.sum(saida_mais * gradiente_saida))

        entradas[idx] = val_original - epsilon
        saida_menos = ativacao.passo_a_frente(entradas)
        loss_menos = float(np.sum(saida_menos * gradiente_saida))

        grad_numerico[idx] = (loss_mais - loss_menos) / (2.0 * epsilon)
        entradas[idx] = val_original
        it.iternext()

    # Restaura o estado da ativação para o valor original
    ativacao.passo_a_frente(entradas)
    return calcular_erro_relativo(grad_analitico, grad_numerico)


def checar_gradiente_camada_densa(
    camada: Densa,
    entradas: np.ndarray,
    gradiente_saida: np.ndarray,
    epsilon: float = 1e-7,
) -> dict[str, float]:
    """Valida os gradientes analíticos de pesos, vieses e entradas de uma camada Densa.

    Considera a perda escalar L = sum(Z * gradiente_saida), onde Z = entradas * W + b.

    Parâmetros:
        camada (Densa): Instância da camada densa a ser testada.
        entradas (np.ndarray): Matriz de entradas X de formato (N, d_in).
        gradiente_saida (np.ndarray): Gradiente recebido dL/dZ de formato (N, d_out).
        epsilon (float): Perturbação infinitesimal para as diferenças finitas.

    Retorna:
        dict[str, float]: Dicionário com os erros relativos para 'pesos', 'vieses' e 'entradas'.
    """
    camada.passo_a_frente(entradas)
    grad_entradas_analitico = camada.passo_atras(gradiente_saida)
    assert camada.gradiente_pesos is not None
    assert camada.gradiente_vieses is not None
    grad_pesos_analitico = camada.gradiente_pesos.copy()
    grad_vieses_analitico = camada.gradiente_vieses.copy()

    # 1. Gradiente numérico dos pesos (W)
    grad_pesos_num = np.zeros_like(camada.pesos)
    it_w = np.nditer(camada.pesos, flags=["multi_index"])
    while not it_w.finished:
        idx = it_w.multi_index
        val_w = camada.pesos[idx]

        camada.pesos[idx] = val_w + epsilon
        z_mais = camada.passo_a_frente(entradas)
        loss_mais = float(np.sum(z_mais * gradiente_saida))

        camada.pesos[idx] = val_w - epsilon
        z_menos = camada.passo_a_frente(entradas)
        loss_menos = float(np.sum(z_menos * gradiente_saida))

        grad_pesos_num[idx] = (loss_mais - loss_menos) / (2.0 * epsilon)
        camada.pesos[idx] = val_w
        it_w.iternext()

    # 2. Gradiente numérico dos vieses (b)
    grad_vieses_num = np.zeros_like(camada.vieses)
    it_b = np.nditer(camada.vieses, flags=["multi_index"])
    while not it_b.finished:
        idx = it_b.multi_index
        val_b = camada.vieses[idx]

        camada.vieses[idx] = val_b + epsilon
        z_mais = camada.passo_a_frente(entradas)
        loss_mais = float(np.sum(z_mais * gradiente_saida))

        camada.vieses[idx] = val_b - epsilon
        z_menos = camada.passo_a_frente(entradas)
        loss_menos = float(np.sum(z_menos * gradiente_saida))

        grad_vieses_num[idx] = (loss_mais - loss_menos) / (2.0 * epsilon)
        camada.vieses[idx] = val_b
        it_b.iternext()

    # 3. Gradiente numérico das entradas (X)
    grad_entradas_num = np.zeros_like(entradas)
    it_x = np.nditer(entradas, flags=["multi_index"])
    while not it_x.finished:
        idx = it_x.multi_index
        val_x = entradas[idx]

        entradas[idx] = val_x + epsilon
        z_mais = camada.passo_a_frente(entradas)
        loss_mais = float(np.sum(z_mais * gradiente_saida))

        entradas[idx] = val_x - epsilon
        z_menos = camada.passo_a_frente(entradas)
        loss_menos = float(np.sum(z_menos * gradiente_saida))

        grad_entradas_num[idx] = (loss_mais - loss_menos) / (2.0 * epsilon)
        entradas[idx] = val_x
        it_x.iternext()

    # Restaura o forward da camada
    camada.passo_a_frente(entradas)
    camada.passo_atras(gradiente_saida)

    return {
        "pesos": calcular_erro_relativo(grad_pesos_analitico, grad_pesos_num),
        "vieses": calcular_erro_relativo(grad_vieses_analitico, grad_vieses_num),
        "entradas": calcular_erro_relativo(grad_entradas_analitico, grad_entradas_num),
    }


def checar_gradiente_rede(
    rede: RedeNeural,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-7,
    tolerancia: float = 1e-6,
) -> dict[str, Any]:
    """Executa a verificação numérica ponta a ponta de todos os parâmetros de uma RedeNeural.

    Avalia a função de custo compilada L(y, Rede(X)) perturbando cada peso e viés
    individualmente através de diferenças finitas centradas, comparando contra
    os gradientes acumulados pelo método passo_atras.

    Parâmetros:
        rede (RedeNeural): Instância compilada e configurada da rede neural.
        X (np.ndarray): Matriz de entradas de formato (N, d_in).
        y (np.ndarray): Matriz de alvos/rótulos reais de formato (N, d_out).
        epsilon (float): Tamanho da perturbação para diferenças finitas centradas.
        tolerancia (float): Limiar máximo aceitável de erro relativo (padrão 1e-6).

    Retorna:
        dict[str, Any]: Relatório estruturado contendo erros relativos de cada camada
            treinável e o status booleano global 'passou_tolerancia'.

    Levanta:
        RuntimeError: Se a rede não estiver compilada com uma função de custo.
    """
    if rede.funcao_custo is None:
        raise RuntimeError("Rede neural deve ser compilada antes da checagem.")

    # 1. Forward e backward analíticos ponta a ponta
    y_pred = rede.passo_a_frente(X)
    grad_custo = rede.funcao_custo.gradiente(y, y_pred)
    rede.passo_atras(grad_custo)

    relatorio_camadas = []
    erros_globais = []

    # 2. Itera por todas as camadas da rede procurando camadas treináveis
    for idx_camada, camada in enumerate(rede.camadas):
        if not isinstance(camada, Densa):
            continue

        assert camada.gradiente_pesos is not None
        assert camada.gradiente_vieses is not None
        grad_w_analitico = camada.gradiente_pesos.copy()
        grad_b_analitico = camada.gradiente_vieses.copy()

        # Checagem de W
        grad_w_num = np.zeros_like(camada.pesos)
        it_w = np.nditer(camada.pesos, flags=["multi_index"])
        while not it_w.finished:
            idx = it_w.multi_index
            val_w = camada.pesos[idx]

            camada.pesos[idx] = val_w + epsilon
            pred_mais = rede.passo_a_frente(X)
            cost_mais = rede.funcao_custo.calcular_custo(y, pred_mais)

            camada.pesos[idx] = val_w - epsilon
            pred_menos = rede.passo_a_frente(X)
            cost_menos = rede.funcao_custo.calcular_custo(y, pred_menos)

            grad_w_num[idx] = (cost_mais - cost_menos) / (2.0 * epsilon)
            camada.pesos[idx] = val_w
            it_w.iternext()

        # Checagem de b
        grad_b_num = np.zeros_like(camada.vieses)
        it_b = np.nditer(camada.vieses, flags=["multi_index"])
        while not it_b.finished:
            idx = it_b.multi_index
            val_b = camada.vieses[idx]

            camada.vieses[idx] = val_b + epsilon
            pred_mais = rede.passo_a_frente(X)
            cost_mais = rede.funcao_custo.calcular_custo(y, pred_mais)

            camada.vieses[idx] = val_b - epsilon
            pred_menos = rede.passo_a_frente(X)
            cost_menos = rede.funcao_custo.calcular_custo(y, pred_menos)

            grad_b_num[idx] = (cost_mais - cost_menos) / (2.0 * epsilon)
            camada.vieses[idx] = val_b
            it_b.iternext()

        err_w = calcular_erro_relativo(grad_w_analitico, grad_w_num)
        err_b = calcular_erro_relativo(grad_b_analitico, grad_b_num)

        erros_globais.extend([err_w, err_b])
        relatorio_camadas.append(
            {
                "indice_camada": idx_camada,
                "erro_relativo_pesos": err_w,
                "erro_relativo_vieses": err_b,
                "passou_pesos": bool(err_w < tolerancia),
                "passou_vieses": bool(err_b < tolerancia),
            }
        )

    # Restaura o estado da rede
    rede.passo_a_frente(X)

    erro_maximo = max(erros_globais) if erros_globais else 0.0
    return {
        "camadas": relatorio_camadas,
        "erro_relativo_maximo": erro_maximo,
        "passou_tolerancia": bool(erro_maximo < tolerancia),
    }
