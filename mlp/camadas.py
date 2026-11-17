import numpy as np

from mlp.base import Camada, Inicializador
from mlp.inicializadores import GlorotUniforme, Zeros


class Densa(Camada):
    """Camada totalmente conectada (linear).

    Parâmetros:
        dimensao_entrada (int): Número de características da entrada.
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
        self.entradas = entradas
        return entradas @ self.pesos + self.vieses

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        if self.entradas is None:
            raise RuntimeError(
                "O passo_a_frente deve ser executado antes do passo_atras."
            )

        # Transpomos a entradas para (dim_entrada, batch)
        # para alinhar a multiplicação com `gradiente_saida`,
        # somando os gradientes ao longo do batch e gerando (dim_entrada, dim_saida),
        # que é exatamente o mesmo formato de `self.pesos`.
        self.gradiente_pesos = self.entradas.T @ gradiente_saida
        self.gradiente_vieses = np.sum(gradiente_saida, axis=0, keepdims=True)
        return gradiente_saida @ self.pesos.T

    # NOTA: Ese método auxiliar NÃO faz parte do contrato original de Camada
    # disponibilizado. Foi adicionado como utilitário para
    # encapsular a atualização SGD dos parâmetros da camada.
    def atualizar_parametros(self, taxa_aprendizado: float) -> None:
        """Atualiza pesos e vieses usando a regra básica da descida do gradiente."""
        if self.gradiente_pesos is None or self.gradiente_vieses is None:
            raise RuntimeError(
                "Gradientes não calculados. Execute passo_atras primeiro."
            )
        self.pesos -= taxa_aprendizado * self.gradiente_pesos
        self.vieses -= taxa_aprendizado * self.gradiente_vieses
