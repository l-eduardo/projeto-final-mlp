# Declaração de Uso de Inteligência Artificial

**INE5664 — Aprendizado de Máquina (UFSC)**  
**Projeto Final: Biblioteca de Redes Neurais (MLP)**  
**Equipe (Grupo L):**
- Luiz Eduardo da Silva (Matrícula: `22202683` | ID UFSC: `22202683`)
- Marcelo de Oliveira (Matrícula: `21104286` | ID UFSC: `21104286`)
- Gaëtan Pierre Baylou-Lanot (Matrícula: `947765` | ID UFSC: `947765`)

**Repositório Oficial:** [github.com/l-eduardo/projeto-final-mlp](https://github.com/l-eduardo/projeto-final-mlp) / [codigos.ufsc.br:luiz.e.silva/projeto-final-mlp](https://codigos.ufsc.br/luiz.e.silva/projeto-final-mlp)

---

## 1. Princípios e Responsabilidade Integral de Autoria

O desenvolvimento deste projeto foi pautado pelas diretrizes éticas e acadêmicas estabelecidas pelo Departamento de Informática e Estatística (INE/CTC/UFSC) para a disciplina INE5664:

1. **Responsabilidade Integral de Autoria:** Todo código, teste, documentação e análise presentes neste repositório é de **100% de responsabilidade dos integrantes da equipe**. Cada membro revisou, executou e compreende profundamente cada dedução matemática e trecho de código, estando plenamente capacitado a explicá-lo, justificá-lo e deduzi-lo na arguição oral individual.
2. **Respeito Estrito às Restrições Acadêmicas:** A Inteligência Artificial atuou como ferramenta auxiliar de engenharia de software e pesquisa, sob supervisão e direcionamento humano constante, com travas explícitas (`docs/restricoes_nota_zero.md` e `CLAUDE.md`) para garantir conformidade inegociável:
   - Pacote `mlp/` dependente **exclusivamente de NumPy**;
   - **Proibição absoluta** de bibliotecas de machine learning ou autograd (`sklearn`, `torch`, `tensorflow`, `autograd`, etc.) dentro de `mlp/`;
   - Proibição de atalhos prontos de ativação do NumPy (como `np.tanh`), exigindo derivação analítica elementar manual;
   - Restrição do uso de `pandas`, `scikit-learn` e `matplotlib` unicamente ao caderno do relatório técnico para carga, divisão e visualização de dados.
3. **Transparência e Rastreabilidade:** Todas as interações com assistentes inteligentes foram documentadas neste arquivo de forma discriminada por etapa do ciclo de vida do projeto.

---

## 2. Detalhamento do Uso de IA por Etapa / Milestone

### 2.1. Milestone 1 — Infraestrutura, Governança, CI/CD e Ambiente
- **Ferramentas Utilizadas:** Google Antigravity CLI / Gemini 3.8 Flash / Claude 3.5 Sonnet.
- **Finalidade:**
  - Estruturação inicial da EAP do projeto e divisão modular das issues com base no enunciado oficial do Moodle;
  - Configuração da automação de integração contínua (GitHub Actions em `.github/workflows/ci.yml`), executando checagem estática de código com Ruff e suíte de testes com Pytest;
  - Configuração do versionamento semântico automatizado com `semantic-release` (`.releaserc.json`);
  - Elaboração das diretrizes de segurança e restrições de nota zero em `docs/restricoes_nota_zero.md` e `CLAUDE.md`.
- **Intervenção e Validação Humana:** Revisão dos scripts de CI, ajuste das versões de dependências em `requirements.txt`, `requirements-dev.txt` e `requirements-notebook.txt` para assegurar o isolamento do NumPy.

### 2.2. Milestone 2 — Componentes Fundamentais da Biblioteca
- **Ferramentas Utilizadas:** Google Antigravity CLI / Gemini 3.8 Flash.
- **Finalidade:**
  - Auxílio na codificação e digitação das 8 funções de ativação e 3 funções de custo conforme as fórmulas deduzidas matematicamente;
  - Apoio na estruturação das classes de inicialização (`Zeros`, `GlorotNormal`, `GlorotUniforme`, `HeNormal`) e da camada linear afim (`Densa`);
  - Geração de casos de teste exaustivos e testes de borda para as funções em `tests/`.
- **Intervenção e Validação Humana:**
  - Rejeição de qualquer sugestão que utilizasse funções prontas externas ou atalhos vetados (como `np.tanh`);
  - Garantia da implementação da fórmula analítica estável $2\sigma(2z) - 1$ para a Tanh;
  - Validação das equações de retropropagação e broadcasting tensorial dos vieses ($\sum_{i=1}^N \delta_{i, :}$).

### 2.3. Milestone 3 — Orquestração da Rede e Auditoria Numérica (Gradient Checking)
- **Ferramentas Utilizadas:** Google Antigravity CLI / Gemini 3.8 Flash.
- **Finalidade:**
  - Estruturação do laço de treinamento Mini-batch SGD na classe `RedeNeural` (`mlp/rede.py`), com embaralhamento aleatório determinístico (*shuffle*), particionamento em lotes e registro de perdas de treino e validação;
  - Implementação do algoritmo de Diferenças Finitas Centradas em `tests/checagem_gradientes.py`, provendo verificação numérica rigorosa com erro de truncamento $\mathcal{O}(\epsilon^2)$ e métrica de erro relativo normalizado de Stanford/CS231n.
- **Intervenção e Validação Humana:**
  - Deducão do produto vetor-Jacobiana na combinação Softmax + CCE;
  - Verificação de que todos os gradientes analíticos da biblioteca apresentavam erro relativo inferior a $10^{-6}$ contra as aproximações numéricas por diferenças finitas.

### 2.4. Milestone 4 — Experimentos Científicos e Relatório Técnico (`relatorio_tecnico.ipynb`)
- **Ferramentas Utilizadas:** Google Antigravity CLI / Gemini 3.8 Flash.
- **Finalidade:**
  - Construção do fluxo de preparação de dados para as três tarefas supervisionadas (Regressão com California Housing, Classificação Binária com Breast Cancer e Classificação Multiclasse com Wine);
  - Geração de gráficos com Matplotlib para curvas de aprendizado, matrizes de confusão e resíduos;
  - Rascunho inicial da estruturação dos tópicos da seção reflexiva "Desafios Enfrentados e Aprendizados".
- **Intervenção e Validação Humana:**
  - Ajuste e validação dos hiperparâmetros (taxa de aprendizado $\eta$, tamanho de lote $B$, épocas e topologia das camadas) para garantir convergência assintótica estável e ausência de sobreajuste;
  - Redação e aprofundamento das análises críticas e interpretações estatísticas/clínicas dos resultados;
  - Execução integral e verificação de reprodutibilidade de todas as células do notebook sob a semente 42.

### 2.5. Milestone 5 — Auditoria Final, Documentação e Preparação para Entrega
- **Ferramentas Utilizadas:** Google Antigravity CLI / Gemini 3.8 Flash.
- **Finalidade:**
  - Atualização do `README.md` com instruções detalhadas de instalação, testes e arquitetura;
  - Consolidação desta declaração de transparência em `AI_USAGE.md`;
  - Elaboração do checklist de entrega e verificação de requisitos de nota zero.
- **Intervenção e Validação Humana:**
  - Auditoria completa dos critérios do Moodle, testes de execução limpa em ambiente isolado e preparação individual para a arguição oral.

---

## 3. Síntese de Conformidade

| Requisito do Enunciado | Status | Evidência no Repositório |
| :--- | :---: | :--- |
| Pacote `mlp/` dependente exclusivamente de NumPy | **100% Conforme** | `mlp/` não possui imports além de `numpy`, `typing`, `abc` e módulos internos. |
| Mínimo de 8 funções de ativação distintas | **100% Conforme** | 8 ativações analíticas implementadas em `mlp/ativacoes.py`. |
| Proibição de `np.tanh` | **100% Conforme** | `Tanh` calculada analiticamente via $2\sigma(2z) - 1$. |
| Mínimo de 2 funções de custo | **100% Conforme** | 3 funções implementadas (`MSE`, `BCE`, `CCE`) em `mlp/custos.py`. |
| Mínimo de 3 inicializadores | **100% Conforme** | 4 inicializadores em `mlp/inicializadores.py`. |
| Forward e Backward analíticos manuais | **100% Conforme** | Implementação manual vetorial auditada com gradcheck ($< 10^{-6}$). |
| Mini-batch SGD com shuffle e épocas | **100% Conforme** | Implementado no método `treinar` de `mlp/rede.py`. |
| Relatório técnico executado nos 3 problemas | **100% Conforme** | `relatorio_tecnico.ipynb` completamente executado e reproduzível. |
| Domínio e defesa na arguição oral | **100% Assegurado** | Todos os integrantes estudaram e dominam o código e os conceitos teóricos. |