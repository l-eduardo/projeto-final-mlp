import numpy as np

from mlp.base import Ativacao, Camada, FuncaoCusto


class RedeNeural:
    """Estrutura principal para encadeamento, treinamento e inferência da MLP."""

    def __init__(self, semente_aleatoria: int | None = None):
        self.camadas: list[Camada | Ativacao] = []
        self.funcao_custo: FuncaoCusto | None = None
        self.semente_aleatoria = semente_aleatoria
        self.gerador_aleatorio = np.random.RandomState(semente_aleatoria)

    def adicionar(self, camada: Camada | Ativacao) -> None:
        """Adiciona uma camada ou função de ativação à rede."""
        if not isinstance(camada, (Camada, Ativacao)):
            raise TypeError(
                f"Objeto inválido do tipo {type(camada).__name__}. "
                "A camada deve ser uma instância de Camada ou Ativacao."
            )
        self.camadas.append(camada)

    def compilar(self, funcao_custo: FuncaoCusto) -> None:
        """Define a função de custo a ser utilizada no treinamento."""
        if not isinstance(funcao_custo, FuncaoCusto):
            raise TypeError(
                f"Objeto inválido do tipo {type(funcao_custo).__name__}. "
                "A função de custo deve ser uma instância de FuncaoCusto."
            )
        self.funcao_custo = funcao_custo

    def passo_a_frente(self, X: np.ndarray) -> np.ndarray:
        """Realiza a propagação direta através de todas as camadas da rede."""
        if not self.camadas:
            raise RuntimeError("Nenhuma camada foi adicionada à rede neural.")

        saida = X
        for camada in self.camadas:
            saida = camada.passo_a_frente(saida)
        return saida

    def passo_atras(self, gradiente_saida: np.ndarray) -> None:
        """Realiza a retropropagação do gradiente do fim para o início da rede."""
        if not self.camadas:
            raise RuntimeError("Nenhuma camada foi adicionada à rede neural.")

        gradiente = gradiente_saida
        for camada in reversed(self.camadas):
            gradiente = camada.passo_atras(gradiente)

    def predizer(self, X: np.ndarray) -> np.ndarray:
        """Realiza predições para novas observações."""
        return self.passo_a_frente(X)

    def treinar(
        self,
        X_treino: np.ndarray,
        y_treino: np.ndarray,
        epocas: int,
        taxa_aprendizagem: float,
        tamanho_lote: int = 32,
        X_validacao: np.ndarray | None = None,
        y_validacao: np.ndarray | None = None,
        verbal: bool = True,
    ) -> dict[str, list[float]]:
        """Executa o laço de treinamento (Mini-batch SGD) e retorna o histórico de custo.
        
        A cada época, os dados de treinamento são embaralhados aleatoriamente e particionados
        em mini-lotes. Para cada lote, são executados: passo à frente, cálculo do gradiente
        da função de custo, passo atrás e atualização de pesos de todas as camadas treináveis.
        Ao final de cada época, o custo agregado de treino é registrado no histórico.
        
        """
        if self.funcao_custo is None:
            raise RuntimeError(
                "Rede não compilada. Execute compilar(funcao_custo) antes de treinar."
            )
        if not self.camadas:
            raise RuntimeError("Nenhuma camada foi adicionada à rede neural.")

        if X_treino.shape[0] != y_treino.shape[0]:
            raise ValueError(
                f"Incompatibilidade no número de amostras: X_treino possui {X_treino.shape[0]} "
                f"amostras, mas y_treino possui {y_treino.shape[0]}."
            )

        tem_validacao = X_validacao is not None and y_validacao is not None
        if (X_validacao is None) != (y_validacao is None):
            raise ValueError(
                "X_validacao e y_validacao devem ser fornecidos em conjunto."
            )

        if tem_validacao and X_validacao.shape[0] != y_validacao.shape[0]:
            raise ValueError(
                f"Incompatibilidade no número de amostras de validação: X_validacao possui "
                f"{X_validacao.shape[0]}, mas y_validacao possui {y_validacao.shape[0]}."
            )

        if epocas <= 0:
            raise ValueError(
                f"O número de épocas deve ser positivo. Recebido: {epocas}."
            )
        if taxa_aprendizagem <= 0:
            raise ValueError(
                f"A taxa de aprendizagem deve ser positiva. Recebido: {taxa_aprendizagem}."
            )
        if tamanho_lote <= 0:
            raise ValueError(
                f"O tamanho do lote deve ser positivo. Recebido: {tamanho_lote}."
            )

        n_amostras = X_treino.shape[0]
        historico: dict[str, list[float]] = {"custo_treino": []}
        if tem_validacao:
            historico["custo_validacao"] = []

        for epoca in range(1, epocas + 1):
            indices = self.gerador_aleatorio.permutation(n_amostras)
            X_embaralhado = X_treino[indices]
            y_embaralhado = y_treino[indices]

            for inicio_idx in range(0, n_amostras, tamanho_lote):
                fim_idx = min(inicio_idx + tamanho_lote, n_amostras)
                X_lote = X_embaralhado[inicio_idx:fim_idx]
                y_lote = y_embaralhado[inicio_idx:fim_idx]

                y_pred_lote = self.passo_a_frente(X_lote)
                grad_lote = self.funcao_custo.gradiente(y_lote, y_pred_lote)
                self.passo_atras(grad_lote)
                self._atualizar_parametros(taxa_aprendizagem)

            y_pred_treino = self.passo_a_frente(X_treino)
            custo_treino = self.funcao_custo.calcular_custo(y_treino, y_pred_treino)
            historico["custo_treino"].append(float(custo_treino))

            if tem_validacao:
                y_pred_val = self.passo_a_frente(X_validacao)
                custo_val = self.funcao_custo.calcular_custo(y_validacao, y_pred_val)
                historico["custo_validacao"].append(float(custo_val))

            if verbal:
                msg = f"Época {epoca}/{epocas} - Custo Treino: {custo_treino:.4f}"
                if tem_validacao:
                    msg += f" - Custo Validação: {custo_val:.4f}"
                print(msg)

        return historico

    def _atualizar_parametros(self, taxa_aprendizagem: float) -> None:
        """Atualiza pesos e vieses de todas as camadas treináveis."""
        for camada in self.camadas:
            if hasattr(camada, "atualizar_parametros"):
                camada.atualizar_parametros(taxa_aprendizagem)
            elif isinstance(camada, Camada):
                camada.pesos -= taxa_aprendizagem * camada.gradiente_pesos
                camada.vieses -= taxa_aprendizagem * camada.gradiente_vieses
