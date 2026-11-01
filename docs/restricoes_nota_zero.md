# Restrições Invioláveis e Critérios de Nota Zero (INE5664)

Este documento estabelece as diretrizes e regras inegociáveis do Projeto Final de Aprendizado de Máquina (INE5664 - INE/CTC/UFSC).

> [!CAUTION]
> **ATENÇÃO CRÍTICA — RISCO DE NOTA 0,0 (ZERO)**  
> A violação de qualquer uma das restrições descritas neste documento resulta em nota final **0,0 (zero)** para o componente de nota do grupo ou para o integrante individualmente.

---

## 1. Restrições Técnicas Absolutas da Biblioteca (`mlp/`)

1. **Uso Proibido de Bibliotecas de Aprendizado de Máquina e Autograd**:
   - É terminantemente vedado o uso ou importação de bibliotecas prontas como:
     - `scikit-learn` (`sklearn`)
     - `PyTorch` (`torch`)
     - `TensorFlow` / `Keras`
     - `JAX`
     - `autograd`
     - Qualquer framework de diferenciação automática ou otimização pronta.
   - Todo algoritmo dentro do pacote `mlp/` deve ser escrito do zero:
     - Propagação direta (*forward*)
     - Retropropagação (*back-propagation*) analítica via regra da cadeia
     - Funções de ativação e de custo com suas respectivas derivadas analíticas
     - Inicializadores de pesos e vieses
     - Laço de treinamento (*Mini-batch SGD*)

2. **Dependência Única Autorizada**:
   - A biblioteca `mlp` pode ter como dependência externa **apenas e exclusivamente o `numpy`** (para operações elementares de álgebra linear e manipulação de arrays).
   - Nenhuma outra biblioteca de terceiros pode ser adicionada como dependência de execução do pacote `mlp`.

3. **Restrições em Funções de Ativação**:
   - São exigidas pelo menos **8 (oito) funções de ativação distintas**.
   - **Função Identidade (linear)** NÃO contabiliza para a cota de 8.
   - **Funções prontas do NumPy (ex.: `np.tanh`)** NÃO contabilizam para este requisito. Devem ser implementadas analiticamente a partir de formulações matemáticas elementares.

4. **Uso Restrito de Bibliotecas Auxiliares**:
   - `matplotlib`, `pandas` e `scikit-learn` são autorizados **EXCLUSIVAMENTE no notebook do relatório técnico** (`relatorio_tecnico.ipynb`), e unicamente para:
     - Carregamento de dados
     - Divisão de conjuntos em treino, validação e teste
     - Cálculo de métricas de avaliação no teste
     - Geração de tabelas e gráficos
   - O treinamento e inferência dos modelos no notebook devem utilizar **estritamente e unicamente a biblioteca `mlp`** desenvolvida pelo grupo.

---

## 2. Casos Oficiais de Atribuição de Nota Zero (0,0)

### Componente de Nota do Grupo ($N_g = 0,0$):
- Release não informada para avaliação.
- Implementação desrespeitar qualquer uma das restrições estipuladas nos Requisitos Técnicos da Biblioteca (dependências proibidas, autograd, etc.).
- Relatório técnico (`relatorio_tecnico.ipynb`) com resultados fabricados ou não reproduzíveis a partir da release entregue.
- Cópia de código identificada entre grupos.

### Componente de Nota Individual ($N = 0,0$):
- Não comparecer à arguição oral no horário agendado (tolerância de 10 minutos).
- Se retirar antes da conclusão da arguição oral sem autorização ou se recusar a responder às perguntas.
- Apresentar desempenho nulo na arguição oral ($N_a = 0,0$).
- Ter contribuição de 10% ou menos no desenvolvimento do trabalho indicada pelos demais integrantes na avaliação por pares e confirmada na arguição.
- Fraudar a participação no trabalho ou consultar meios não autorizados (Internet, IA, colegas) durante a arguição.
