# Projeto Final: Biblioteca de Redes Neurais (MLP)

**INE5664 — Aprendizado de Máquina (UFSC)**  
Repositório Oficial: [codigos.ufsc.br:luiz.e.silva/projeto-final-mlp.git](https://codigos.ufsc.br/luiz.e.silva/projeto-final-mlp)

Este repositório fornece a estrutura base (*starter kit*) e a esteira de engenharia para o desenvolvimento de uma biblioteca pura de Perceptron Multicamadas (Multi-Layer Perceptron — MLP) em Python/NumPy, bem como os experimentos comparativos para o trabalho prático da disciplina.

---

## 1. Regras do Código-Fonte (`mlp/`)

* **Contrato Fixo (`mlp/base.py`):** **Não altere** as interfaces e classes abstratas deste arquivo. As assinaturas e contratos públicos são imutáveis e consumidos pelas suítes automatizadas de teste e avaliação.
* **Preenchimento dos Esqueletos:** Substitua as exceções `raise NotImplementedError` pelas implementações genuínas nas seguintes classes e módulos:
  * `mlp/rede.py`: Gerenciamento do fluxo de camadas, propagação direta (*forward pass*), retropropagação de gradientes (*backward pass*) e laço de treinamento com Mini-batch SGD.
  * `mlp/camadas.py`: Combinador linear afim, cálculo de gradientes analíticos dos parâmetros ($W, b$) e propagação do erro ($\delta$).
  * `mlp/ativacoes.py`: Implementação manual das funções de ativação e de suas derivadas exatas (mínimo de 8 funções).
  * `mlp/custos.py`: Cálculo do erro escalar e respectivo gradiente analítico em relação à saída da rede (mínimo de 2 funções).
  * `mlp/inicializadores.py`: Estratégias determinísticas e estocásticas de inicialização de pesos e bias (mínimo de 3 estratégias).
* **Restrição Estrita de Bibliotecas:** O pacote `mlp/` deve conter **exclusivamente NumPy** como dependência externa de tempo de execução. O uso de frameworks de aprendizado de máquina ou diferenciação automática (`torch`, `sklearn`, `scipy`, `autograd`, `tensorflow`, `jax`, `keras`) no diretório `mlp/` é estritamente proibido.
* **Proibição de `np.tanh`:** O uso da primitiva `np.tanh` ou `numpy.tanh` é estritamente vedado pelas regras acadêmicas da disciplina. A tangente hiperbólica deve ser implementada manualmente por meio de operações aritméticas de exponenciais (`np.exp`).

---

## 2. Relatório Técnico e Experimentos (`relatorio_tecnico.ipynb`)

* O relatório técnico e os experimentos práticos da disciplina devem ser desenvolvidos obrigatoriamente no arquivo [`relatorio_tecnico.ipynb`](relatorio_tecnico.ipynb).
* O arquivo deve ser entregue **totalmente executado e reproduzível**, preservando todas as células de saída, tabelas comparativas e gráficos gerados.
* O uso de bibliotecas analíticas e de visualização (`pandas`, `scikit-learn`, `matplotlib`, `jupyter`) é permitido **exclusivamente no notebook** para carga e pré-processamento de datasets, divisão de partições (treino/validação/teste), métricas auxiliares de avaliação e plotagem de curvas de convergência.

---

## 3. Configuração do Ambiente e Instalação

O projeto adota o padrão moderno de empacotamento Python (PEP 517, PEP 518 e PEP 621) definido no `pyproject.toml`, garantindo o isolamento rigoroso entre as dependências de produção, desenvolvimento e análise.

### 3.1. Criação do Ambiente Virtual

Recomenda-se utilizar o Python 3.10 ou superior:

```bash
# Cria o ambiente virtual isolado (.venv)
python3 -m venv .venv

# Ativa o ambiente virtual
# No Linux / macOS:
source .venv/bin/activate

# No Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Atualiza as ferramentas básicas de empacotamento
pip install --upgrade pip setuptools wheel
```

### 3.2. Instalação com Dependências Isoladas

Você pode instalar o pacote em modo editável (`-e`) escolhendo o conjunto exato de dependências necessário:

1. **Apenas a Biblioteca Core (`mlp`):**
   Instala exclusivamente o pacote com sua dependência mínima (`numpy>=1.21.0`):
   ```bash
   pip install -e .
   ```

2. **Ambiente de Desenvolvimento e Testes (`dev`):**
   Adiciona ferramentas de qualidade de código, linters e frameworks de teste (`pytest`, `pytest-cov`, `ruff`):
   ```bash
   pip install -e .[dev]
   ```

3. **Ambiente de Execução do Notebook (`notebook`):**
   Adiciona os pacotes analíticos necessários para rodar o `relatorio_tecnico.ipynb` (`matplotlib`, `pandas`, `scikit-learn`, `jupyter`, `notebook`):
   ```bash
   pip install -e .[notebook]
   ```

4. **Instalação Completa (Desenvolvimento + Notebook):**
   Para desenvolvedores que desejam executar testes e o relatório na mesma máquina:
   ```bash
   pip install -e .[dev,notebook]
   ```

---

## 4. Validação Local de Qualidade e Conformidade

Antes de submeter qualquer contribuição ou abrir um Merge Request, execute a sequência de validações locais para garantir que a esteira de integração contínua (CI) passará sem falhas:

### 4.1. Verificação de Estilo e Linting (Ruff)
Garante conformidade com as diretrizes de código e ausência de problemas sintáticos:
```bash
ruff check .
```

### 4.2. Auditoria Estática de AST (`scripts/audit_ast.py`)
Executa uma análise estática formal baseada na Árvore Sintática Abstrata (AST) do Python sobre todos os arquivos dentro de `mlp/`. O auditor verifica e bloqueia:
- Imports não autorizados de ML ou autograd (`torch`, `sklearn`, `scipy`, `autograd`, `tensorflow`, `jax`, etc.).
- Uso proibido de chamadas diretas ou indiretas a `np.tanh` / `numpy.tanh`.
- Chamadas a primitivas perigosas de reflexão/execução dinâmica (`eval`, `exec`, `compile`, `__import__`).

Comando:
```bash
python3 scripts/audit_ast.py
```
*Código de retorno esperado: `0` (passou).*

### 4.3. Execução da Suíte de Testes com Cobertura
Executa os testes automatizados com relatório detalhado de cobertura de código:
```bash
pytest tests/
# Ou com detalhamento de cobertura:
pytest --cov=mlp --cov-report=term-missing tests/
```

---

## 5. Governança de Branches e Arquitetura de CI/CD

O projeto adota práticas rigorosas de engenharia de software para assegurar a estabilidade do repositório e conformidade acadêmica:

### 5.1. Regras de Branches e Proteção da `main`
- A branch `main` no GitLab UFSC é **estritamente protegida**: pushes diretos são bloqueados.
- Todo código deve ser desenvolvido em branches de funcionalidade (`feat/issue-<num>-<descricao>`) e integrado exclusivamente por meio de **Merge Requests (MR)**.
- O Merge Request exige aprovação e pipeline de CI 100% verde antes da fusão.
- Todo Merge Request deve preencher obrigatoriamente o template padrão em `.gitlab/merge_request_templates/Default.md`, incluindo o **Checklist de Conformidade Acadêmica**.

### 5.2. Pipeline de CI/CD (`.gitlab-ci.yml`)
A automação no GitLab CI executa os seguintes estágios:
1. **`lint`:** Executa `ruff check .` garantindo padrões de código.
2. **`audit`:** Executa a verificação AST independente (`python3 scripts/audit_ast.py`) com Python 3.10 padrão sem tolerância a violações.
3. **`test`:** Roda a suíte completa com `pytest` e exporta relatórios de cobertura (terminal e XML).
4. **`release`:** Estágio de CD disparado automaticamente apenas após a fusão bem-sucedida de um MR na branch `main`. O script de release calcula a próxima versão semântica (SemVer), gera a tag Git anotada e publica uma GitLab Release oficial com o changelog automatizado.

---

## 6. Transparência no Uso de Inteligência Artificial

Em consonância com as boas práticas de integridade acadêmica e transparência, o uso de ferramentas de Inteligência Artificial ao longo de todas as etapas deste projeto está documentado de forma minuciosa em [`AI_USAGE.md`](AI_USAGE.md). Consulte esse documento para verificar escopo de uso, decisões de engenharia assistidas e metodologia de supervisão humana.
