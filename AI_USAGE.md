# Declaração de Uso de Inteligência Artificial

**INE5664 — Aprendizado de Máquina (UFSC)**  
**Projeto Final: Biblioteca de Redes Neurais (MLP)**  
**Repositório:** [codigos.ufsc.br:luiz.e.silva/projeto-final-mlp.git](https://codigos.ufsc.br/luiz.e.silva/projeto-final-mlp)

---

## 1. Máximas e Responsabilidade de Autoria

O uso de assistentes de IA neste projeto segue princípios inegociáveis de responsabilidade e integridade acadêmica:

1. **Responsabilidade integral de autoria:** Qualquer linha de código ou configuração integrada ao repositório, mesmo quando gerada integralmente por agentes de IA, é de **100% de responsabilidade do autor do Pull Request/Merge Request e dos integrantes que o revisaram**.
2. **Revisão e compreensão obrigatórias:** Nenhuma implementação pode ser aprovada sem que tenha sido minuciosamente revisada, validada e plenamente compreendida pelos integrantes do grupo. Cada membro deve ser capaz de explicar, deduzir matematicamente e defender cada trecho do projeto na arguição oral.
3. **Respeito irrestrito às restrições do enunciado:** A IA atua como ferramenta de produtividade e apoio de engenharia de software, sem autorização para contornar ou relaxar as exigências da disciplina:
   - Pacote `mlp/` dependente exclusivamente de NumPy.
   - Proibição de frameworks de machine learning ou bibliotecas de diferenciação automática (`torch`, `sklearn`, `scipy`, `autograd`, `tensorflow`, `jax`, etc.).
   - Proibição de funções prontas do NumPy como `np.tanh` nas ativações da biblioteca.
   - Nenhuma inserção de arquivos internos ou metadados de IA no histórico do repositório Git.

---

## 2. Uso de IA na Milestone 1

Na etapa de **Configuração, Governança e CI/CD**, a Inteligência Artificial foi utilizada com as seguintes finalidades:

### 2.1. Planejamento e Estruturação de Tarefas
- Estruturação da EAP e levantamento dos requisitos do enunciado.
- Cadastro e organização das milestones e issues no GitLab com links de estudo e critérios de aceitação.

### 2.2. Automação de CI/CD e Governança
- Configuração do pipeline do GitLab CI (`.gitlab-ci.yml`) para executar linters (`ruff`) e suíte de testes (`pytest`).
- Configuração do `semantic-release` (`.releaserc.json`) para geração de releases e tags automáticas no GitLab após merges na branch `main`.
- Elaboração do template padrão de Merge Request (`.gitlab/merge_request_templates/Default.md`) com checklist de conformidade acadêmica e seção para referências de estudo.
- Criação dos arquivos de diretrizes de IA (`CLAUDE.md`, `.gemini/rules.md` e `docs/restricoes_nota_zero.md`) para orientar assistentes a nunca violarem as regras do trabalho.

### 2.3. Gestão de Dependências e Ambiente
- Isolamento das dependências através de arquivos de requisitos (`requirements.txt`, `requirements-dev.txt`), mantendo `numpy` como dependência exclusiva para a biblioteca `mlp` e separando ferramentas de desenvolvimento (`pytest`, `ruff`) e pacotes para análise no relatório técnico.
