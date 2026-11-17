# Checklist Final de Auditoria, Conformidade e Entrega do Projeto

**INE5664 — Aprendizado de Máquina (UFSC)**  
**Projeto Final: Biblioteca de Redes Neurais (MLP)**  
**Equipe (Grupo L):**
- Luiz Eduardo da Silva (Matrícula: `22202683` | ID UFSC: `22202683`)
- Marcelo de Oliveira (Matrícula: `21104286` | ID UFSC: `21104286`)
- Gaëtan Pierre Baylou-Lanot (Matrícula: `947765` | ID UFSC: `947765`)

---

## 1. Auditoria dos Requisitos Técnicos Obrigatórios (Risco de Nota Zero)

| Requisito do Enunciado | Verificação Realizada | Status |
| :--- | :--- | :---: |
| **Linguagem Python** | Implementação desenvolvida inteiramente em Python 3.10+ compatível. | ✅ **CONFORME** |
| **Respeito ao Contrato Fixo (`mlp/base.py`)** | As interfaces `Camada`, `Ativacao`, `FuncaoCusto` e `Inicializador` permaneceram inalteradas. | ✅ **CONFORME** |
| **Dependência Única: NumPy no pacote `mlp/`** | Auditoria estática confirma que apenas `numpy`, `abc` e `typing` são importados em `mlp/`. Nenhuma biblioteca externa proibida (`sklearn`, `torch`, `autograd`, `tensorflow`) é importada. | ✅ **CONFORME** |
| **Mínimo de 8 Funções de Ativação Distintas** | Implementadas 8 classes em `mlp/ativacoes.py`: `Sigmoide`, `LeakyReLU`, `Swish`, `Softmax`, `Tanh`, `ELU`, `Softplus`, `Softsign`. Identidade foi excluída. | ✅ **CONFORME** |
| **Proibição Estrita de `np.tanh`** | A função `Tanh` foi implementada analiticamente a partir da sigmoide elementar: $2\sigma(2z) - 1$, sem qualquer chamada a `np.tanh`. | ✅ **CONFORME** |
| **Mínimo de 2 Funções de Custo** | Implementadas 3 funções completas com gradientes analíticos em `mlp/custos.py`: `MSE`, `BCE`, `CCE`. | ✅ **CONFORME** |
| **Mínimo de 3 Inicializadores de Pesos** | Implementadas 4 classes em `mlp/inicializadores.py`: `Zeros`, `GlorotNormal`, `GlorotUniforme`, `HeNormal`. | ✅ **CONFORME** |
| **Algoritmos de Forward e Back-propagation** | Propagação direta e retropropagação analíticas manuais completas implementadas em `Camada` e `Ativacao`. | ✅ **CONFORME** |
| **Otimizador Mini-batch SGD com Shuffle** | Laço de treinamento com embaralhamento determinístico (`permutation`), particionamento em mini-lotes e atualização de parâmetros. | ✅ **CONFORME** |
| **Reprodutibilidade por Semente Aleatória** | Semente configurável em `RedeNeural` e camadas (`semente_aleatoria=42`), garantindo reprodutibilidade exata. | ✅ **CONFORME** |

---

## 2. Auditoria da Suíte de Testes e Qualidade de Código

- [x] **Suíte de Testes Automatizados:** 164 testes unitários executados com 100% de sucesso via `pytest tests/`.
- [x] **Auditoria Matemática (Gradient Checking):** 17 testes de verificação numérica em `tests/test_checagem_gradientes.py`, comprovando erro relativo $< 10^{-6}$ em todas as ativações, camadas densas e redes completas.
- [x] **Linters e Formatação Estática:** 0 advertências no Ruff (`ruff check mlp/ tests/` e `ruff format --check mlp/ tests/`).

---

## 3. Auditoria do Relatório Técnico (`relatorio_tecnico.ipynb`)

- [x] **Identificação da Equipe:** Tabela na Célula 0 preenchida com nomes, matrículas e IDs UFSC do Grupo L.
- [x] **Experimento 1 (Regressão):** Executado sobre *California Housing* com normalização rigorosa, curvas de aprendizado MSE, análise de viés e variância, e métricas no teste ($R^2 = 70.26\%$, $\text{MAE} = 0.4426$).
- [x] **Experimento 2 (Classificação Binária):** Executado sobre *Breast Cancer Wisconsin* com divisão estratificada, curvas de perda BCE, Matriz de Confusão, e discussão do trade-off clínico entre Precisão e Sensibilidade ($\text{Acurácia} = 97.67\%$, $\text{F1} = 96.77\%$).
- [x] **Experimento 3 (Classificação Multiclasse):** Executado sobre *Wine Recognition Dataset* (3 classes) com codificação One-Hot, curvas CCE, Matriz de Confusão 3x3, e métricas Macro ($\text{Acurácia} = 96.30\%$, $\text{F1 Macro} = 96.59\%$).
- [x] **Seção 'Desafios e Aprendizados':** Redigida detalhando instabilidade de ponto flutuante, produto vetor-Jacobiana em Softmax+CCE, inicialização de pesos e condicionamento de gradientes.
- [x] **Estado de Execução:** Caderno entregue **100% executado e reprodutivo**, com todas as saídas, tabelas e gráficos salvos.

---

## 4. Instruções e Procedimentos para Submissão no Moodle

### 4.1. Repositório no GitLab da UFSC (Códigos@UFSC)
Conforme exigência do enunciado do Moodle:
1. O repositório deve estar espelhado na plataforma GitLab da UFSC ([codigos.ufsc.br:luiz.e.silva/projeto-final-mlp](https://codigos.ufsc.br/luiz.e.silva/projeto-final-mlp));
2. A visibilidade do repositório no GitLab UFSC deve ser alterada para **Pública** (*Settings > General > Visibility, project features, permissions > Project visibility > Public*) antes do prazo de entrega;
3. O histórico de commits comprova a contribuição e rastreabilidade dos integrantes.

### 4.2. Release e Tag para Avaliação
1. A release recomendada para submissão é a tag de versão gerada após a consolidação final (ex.: `v1.5.0` ou tag oficial indicada pelo grupo);
2. A submissão na atividade do Moodle (*Entrega do Projeto Final: Release do Repositório Git*) deve conter:
   - **Link do Repositório Git:** `https://codigos.ufsc.br/luiz.e.silva/projeto-final-mlp` (ou link correspondente da equipe no GitLab UFSC);
   - **Release / Tag de Avaliação:** `v1.5.0` (ou commit hash / tag final consolidada).

### 4.3. Avaliação por Pares Obrigatória
- **Atenção:** Cada integrante deve preencher individualmente o formulário de *Avaliação por Pares: Contribuição no Projeto Final* no Moodle até o dia 18/11 às 09:00. O não envio acarreta desconto individual de 0,5 ponto na nota final.

### 4.4. Preparação para a Arguição Oral
- **Data e Horário:** Conforme cronograma oficial do Grupo L (30/11/2026 às 19:35, ou conforme agendamento do professor no Moodle).
- **Duração:** 25 minutos com perguntas individuais para cada membro.
- **Ambiente:** Levar o computador do grupo com o repositório clonado localmente, ambiente virtual configurado e notebook executado. Consultas autorizadas apenas ao próprio repositório (proibido celular, Internet ou IA durante a arguição).