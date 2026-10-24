# Projeto Final: Biblioteca de Redes Neurais (MLP)

**INE5664 — Aprendizado de Máquina (UFSC)**  
Repositório Oficial: [codigos.ufsc.br:luiz.e.silva/projeto-final-mlp.git](https://codigos.ufsc.br/luiz.e.silva/projeto-final-mlp)

Este repositório fornece a estrutura base (*starter kit*) e a esteira de engenharia para o desenvolvimento de uma biblioteca de Perceptron Multicamadas (Multi-Layer Perceptron — MLP) em Python/NumPy, bem como os experimentos para o relatório técnico da disciplina.

---

## 1. Diretrizes do Código-Fonte (`mlp/`)

* **Contrato Fixo (`mlp/base.py`):** **Não altere** as interfaces e classes abstratas deste arquivo. As assinaturas e contratos públicos são imutáveis e consumidos pelas suítes automatizadas de avaliação.
* **Preenchimento dos Módulos:**
  * `mlp/rede.py`: Gerenciamento do fluxo de camadas, propagação direta (*forward pass*), retropropagação de gradientes (*backward pass*) e laço de treinamento com Mini-batch SGD.
  * `mlp/camadas.py`: Camada linear afim, cálculo de gradientes analíticos dos parâmetros ($W, b$) e retropropagação do erro ($\delta$).
  * `mlp/ativacoes.py`: Implementação manual das funções de ativação e de suas derivadas analíticas (mínimo de 8 funções).
  * `mlp/custos.py`: Cálculo do erro escalar e respectivo gradiente analítico em relação à saída da rede (mínimo de 2 funções).
  * `mlp/inicializadores.py`: Estratégias de inicialização de pesos e bias (mínimo de 3 estratégias).
* **Restrições Invioláveis:**
  * Para detalhes sobre proibições e critérios oficiais de nota zero, consulte [`docs/restricoes_nota_zero.md`](docs/restricoes_nota_zero.md).

---

## 2. Relatório Técnico e Experimentos (`relatorio_tecnico.ipynb`)

* O relatório técnico e os experimentos práticos da disciplina devem ser desenvolvidos no arquivo [`relatorio_tecnico.ipynb`](relatorio_tecnico.ipynb).
* O arquivo deve ser entregue **totalmente executado e reproduzível**, preservando todas as células de saída, tabelas comparativas e gráficos gerados.
* Bibliotecas analíticas e de visualização (`pandas`, `scikit-learn`, `matplotlib`, `jupyter`) são permitidas **exclusivamente no notebook** para carga e pré-processamento de datasets, divisão em treino/validação/teste, métricas de avaliação e plotagem de curvas.
* O treinamento dos modelos no notebook deve utilizar exclusivamente a biblioteca `mlp`.

---

## 3. Configuração do Ambiente e Instalação

O projeto adota o padrão moderno de empacotamento Python (PEP 517, PEP 518 e PEP 621) definido no `pyproject.toml`, garantindo o isolamento entre dependências de runtime, desenvolvimento e análise.

### 3.1. Criação do Ambiente Virtual

```bash
# Cria o ambiente virtual isolado (.venv)
python3 -m venv .venv

# Ativa o ambiente virtual
# No Linux / macOS:
source .venv/bin/activate

# No Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Atualiza ferramentas de empacotamento
pip install --upgrade pip setuptools wheel
```

### 3.2. Instalação com Dependências Isoladas

1. **Apenas a Biblioteca Core (`mlp`):**
   ```bash
   pip install -e .
   ```

2. **Ambiente de Desenvolvimento e Testes (`dev`):**
   ```bash
   pip install -e .[dev]
   ```

3. **Ambiente de Execução do Notebook (`notebook`):**
   ```bash
   pip install -e .[notebook]
   ```

4. **Instalação Completa (Desenvolvimento + Notebook):**
   ```bash
   pip install -e .[dev,notebook]
   ```

---

## 4. Validação Local de Qualidade

Antes de submeter contribuições ou abrir um Merge Request:

### 4.1. Verificação de Estilo e Linting (Ruff)
```bash
ruff check .
ruff format --check .
```

### 4.2. Execução dos Testes da Biblioteca (Pytest)
```bash
pytest tests/
```

---

## 5. Governança de Branches e CI/CD

- A branch `main` no GitLab UFSC é **estritamente protegida** contra push direto.
- Todo desenvolvimento ocorre em branches de funcionalidade integradas via **Merge Requests (MR)**.
- O template padrão de MR está disponível em `.gitlab/merge_request_templates/Default.md`.
- A esteira de CI/CD do GitLab (`.gitlab-ci.yml`) executa os estágios de `lint` e `test`. Após o merge na branch `main`, o job de `release` aciona o `semantic-release` para gerar automaticamente a tag e notas da release.

---

## 6. Transparência no Uso de Inteligência Artificial

O uso de ferramentas de Inteligência Artificial está documentado em [`AI_USAGE.md`](AI_USAGE.md).
