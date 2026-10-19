# Declaração de Uso de Inteligência Artificial (AI Usage Disclosure)

**INE5664 — Aprendizado de Máquina (UFSC)**  
**Projeto Final: Biblioteca de Redes Neurais (MLP)**  
**Repositório:** [codigos.ufsc.br:luiz.e.silva/projeto-final-mlp.git](https://codigos.ufsc.br/luiz.e.silva/projeto-final-mlp)

---

## 1. Princípios de Transparência e Integridade Acadêmica

Em conformidade com as diretrizes de ética, integridade acadêmica da Universidade Federal de Santa Catarina (UFSC) e os requisitos estipulados para o projeto final da disciplina INE5664, este documento fornece uma divulgação completa, transparente e detalhada de como ferramentas e modelos de Inteligência Artificial Generativa foram utilizados no projeto.

Adota-se como princípio inegociável que a Inteligência Artificial atua como ferramenta de produtividade e engenharia de software assistida, **sem jamais substituir a responsabilidade humana, o rigor conceitual ou as regras de negócio acadêmicas impostas pelo enunciado**. Especificamente:
- **Exclusividade NumPy no Core:** A biblioteca `mlp/` é desenvolvida do zero sem o uso de caixas-pretas de aprendizado de máquina (`torch`, `sklearn`, `scipy`, `autograd`, `tensorflow`, `jax`, etc.).
- **Proibição Estrita de `np.tanh`:** Funções de ativação são derivadas e computadas por aritmética elementar de matrizes e exponenciais, garantindo compreensão matemática integral.
- **Imutabilidade das Interfaces Base:** Os contratos arquiteturais definidos em `mlp/base.py` permanecem inalterados.
- **Higienização de Repositório:** Nenhum metadado interno, log de agente ou artefato de planejamento de IA (`.agents/`, `.gemini/`, `.claude/`) é versionado no repositório final de código.

---

## 2. Detalhamento do Uso de IA na Milestone 1

Na Milestone 1, dedicada à **Infraestrutura, Governança, Empacotamento e Auditoria AST**, a Inteligência Artificial foi empregada nas seguintes áreas específicas:

### 2.1. Planejamento Arquitetural e Decomposição de Tarefas
* **Modelos Utilizados:** Modelos de raciocínio e síntese de código baseados em LLMs avançados via arquitetura de subagentes cooperativos.
* **Escopo de Atuação:**
  - Decomposição das exigências do enunciado e dos requisitos do usuário em cinco frentes paralelas de trabalho (Issues #1 a #5).
  - Estruturação do fluxo de trabalho baseado em múltiplos Git worktrees isolados (`governance`, `packaging`, `ast-audit`, `ci-cd`), permitindo desenvolvimento paralelo sem colisões de branch.
  - Definição formal de contratos de interface entre os componentes (CLI do auditor AST, especificações do `pyproject.toml`, estágios do `.gitlab-ci.yml`).

### 2.2. Construção da Infraestrutura de CI/CD e Automação de Releases
* **Escopo de Atuação:**
  - Geração da configuração da esteira declarativa do GitLab CI (`.gitlab-ci.yml`), contemplando os estágios de `lint`, `audit`, `test` e `release`.
  - Configuração do ambiente hermético baseado na imagem oficial `python:3.10-slim`, com caching otimizado de pacotes pip (`$CI_PROJECT_DIR/.cache/pip`).
  - Implementação do script de automação de releases (`scripts/release_helper.py`), responsável por inspecionar o histórico de commits, calcular o próximo incremento SemVer (major/minor/patch), extrair o changelog descritivo e disparar a criação de tags Git e GitLab Releases oficiais de forma automatizada no merge para a branch `main`.

### 2.3. Desenvolvimento do Auditor de Conformidade AST (`scripts/audit_ast.py`)
* **Escopo de Atuação:**
  - Projeto e codificação de um analisador estático baseado na Árvore Sintática Abstrata do Python (`ast.NodeVisitor`), sem qualquer dependência externa além da biblioteca padrão do Python 3.10.
  - Modelagem das regras de inspeção para:
    1. Interrupção imediata da esteira caso sejam detectadas instruções `import` ou `from ... import` envolvendo frameworks de ML e diferenciação automática proibidos.
    2. Detecção de acessos diretos a atributos ou chamadas de função associadas à tangente hiperbólica da biblioteca NumPy (`np.tanh`, `numpy.tanh`), forçando a implementação manual de `tanh`.
    3. Bloqueio preventivo de primitivas de metaprogramação dinâmica que poderiam contornar a análise estática (`eval`, `exec`, `compile`, `__import__`).
  - Criação de suíte de testes unitários dedicada (`tests/test_ast_audit.py`) para validar se o auditor rejeita códigos transgressores e aprova códigos conformes.

### 2.4. Empacotamento e Isolamento de Dependências
* **Escopo de Atuação:**
  - Estruturação do `pyproject.toml` segundo os padrões modernos da comunidade (PEP 517 / PEP 518 / PEP 621) utilizando `setuptools.build_meta`.
  - Separação estrita de dependências:
    - Core: única e exclusivamente `numpy>=1.21.0`.
    - Grupo `dev`: ferramentas de testes e análise estática (`pytest`, `pytest-cov`, `ruff`).
    - Grupo `notebook`: dependências analíticas e de visualização reservadas ao relatório técnico (`matplotlib`, `pandas`, `scikit-learn`, `jupyter`, `notebook`).
  - Criação da suíte inicial de testes de fumaça (*smoke tests* em `tests/test_smoke.py`), validando importação limpa, hierarquia de classes abstratas e isolamento em tempo de execução.

### 2.5. Governança, Templates de Contribuição e Documentação
* **Escopo de Atuação:**
  - Redação do template de Merge Request (`.gitlab/merge_request_templates/Default.md`) contendo seções para descrição de alterações, rastreamento de issues correlatas e o **Checklist de Conformidade Acadêmica** obrigatório.
  - Atualização abrangente do `README.md`, corrigindo referências de documentação (apontando para o arquivo correto `relatorio_tecnico.ipynb`), documentando os passos de criação e ativação do ambiente virtual (`.venv`), os comandos de validação local e as diretrizes de governança de branches protegidas.
  - Configuração do `.gitignore` garantindo a exclusão peremptória de metadados de ferramentas de IA (`.agents/`, `.gemini/`, `.claude/`).

---

## 3. Metodologia de Supervisão Humana e Auditoria Cruzada

A governança do projeto estabeleceu um ciclo rigoroso de revisão e validação:
1. **Revisão e Auditoria Cruzada:** Toda contribuição e artefato produzido por um agente de implementação é submetido à revisão independente por um agente auditor dedicado, que verifica a ausência de atalhos, hardcodes ou violações de regras.
2. **Execução Local e Remota:** Nenhuma alteração é considerada concluída antes da execução e passagem bem-sucedida de todos os comandos de validação (`ruff check .`, `python3 scripts/audit_ast.py` e `pytest tests/`) tanto localmente quanto nos contêineres do GitLab CI.
3. **Supervisão Humana Contínua:** Todas as decisões arquiteturais, formulações matemáticas e validações finais são supervisionadas, compreendidas e assumidas integralmente pelo autor humano do repositório.

---

## 4. Registro de Atualizações

| Versão | Data | Autor / Agente | Escopo das Modificações |
|---|---|---|---|
| 1.0.0 | 2026-10-19 | Worker M1.5 / Equipe | Documentação inicial abrangente do uso de IA na Milestone 1 (Infraestrutura, Governança, CI/CD e Auditoria AST). |
