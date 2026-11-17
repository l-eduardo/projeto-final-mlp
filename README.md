# Projeto Final: Biblioteca de Redes Neurais (MLP)

**INE5664 — Aprendizado de Máquina**  
**Departamento de Informática e Estatística (INE/CTC/UFSC)**  
**Professor:** Prof. Dr. Eduardo Camilo Inacio  
**Equipe (Grupo L):**
- Luiz Eduardo da Silva (Matrícula: `22202683` | ID UFSC: `22202683`)
- Marcelo de Oliveira (Matrícula: `21104286` | ID UFSC: `21104286`)
- Gaëtan Pierre Baylou-Lanot (Matrícula: `947765` | ID UFSC: `947765`)

---

## 1. Visão Geral do Projeto

Este repositório contém a implementação completa, modular e extensível de uma biblioteca de Redes Neurais Artificiais do tipo Perceptron Multicamadas (*Multilayer Perceptron — MLP*) desenvolvida a partir de primeiros princípios em **Python**, utilizando **exclusivamente NumPy** para álgebra linear e manipulação tensorial.

O projeto atende a todos os requisitos do enunciado da disciplina INE5664 e acompanha uma suíte exaustiva de testes unitários com 100% de aprovação, módulo de verificação numérica de gradientes (*Gradient Checking*), cadernos didáticos de estudo e um relatório técnico experimental reproduzível (`relatorio_tecnico.ipynb`).

---

## 2. Estrutura do Repositório

```text
projeto-final-mlp/
├── mlp/                          # Pacote principal da biblioteca (NumPy puro)
│   ├── __init__.py               # Exportação dos componentes da biblioteca
│   ├── base.py                   # Interfaces abstratas fixas (Camada, Ativacao, FuncaoCusto, Inicializador)
│   ├── inicializadores.py        # Estratégias de inicialização de pesos e vieses
│   ├── ativacoes.py              # 8 funções de ativação e derivadas analíticas estáveis
│   ├── custos.py                 # Funções de perda (MSE, BCE, CCE) e gradientes
│   ├── camadas.py                # Camada Densa linear afim (forward e backward)
│   └── rede.py                   # Classe RedeNeural e laço Mini-batch SGD
├── tests/                        # Suíte de testes automatizados com Pytest (164 testes)
│   ├── checagem_gradientes.py    # Utilitário de auditoria numérica de gradientes por diferenças finitas
│   ├── test_inicializadores.py   # Testes estatísticos e dimensionais dos inicializadores
│   ├── test_ativacoes.py         # Testes de forward e backward das 8 funções de ativação
│   ├── test_custos.py            # Testes de perdas e gradientes analíticos
│   ├── test_camadas.py           # Testes da camada Densa (pesos, vieses, entradas)
│   ├── test_rede.py              # Testes do laço SGD, validação e reprodutibilidade
│   └── test_checagem_gradientes.py # Validação ponta a ponta do gradcheck
├── parciais/                     # Cadernos Jupyter de estudo e simulação por módulo
│   ├── inicializadores.ipynb     # Estudo das distribuições estatísticas
│   ├── ativacoes.ipynb           # Análise de gradientes e saturação das 8 ativações
│   ├── custos.ipynb              # Superfícies de perda e gradientes
│   ├── camadas.ipynb             # Fluxo dimensional e propagação afim
│   ├── rede.ipynb                # Simulações de convergência e fronteiras de decisão
│   ├── checagem_gradientes.ipynb # Demonstração de truncamento e diferenças finitas
│   ├── experimento_regressao.ipynb # Estudo dedicado de regressão (California Housing)
│   ├── experimento_classificacao_binaria.ipynb # Estudo dedicado de classificação binária (Breast Cancer)
│   └── experimento_classificacao_multiclasse.ipynb # Estudo dedicado multiclasse (Wine)
├── docs/                         # Documentações de governança e conformidade
│   └── restricoes_nota_zero.md   # Diretrizes críticas e critérios de nota zero
├── relatorio_tecnico.ipynb       # Relatório técnico oficial executado e reproduzível
├── requirements.txt              # Dependência de execução da biblioteca (apenas numpy)
├── requirements-dev.txt          # Ferramentas de desenvolvimento e testes (pytest, ruff)
├── requirements-notebook.txt     # Pacotes auxiliares para o relatório técnico
├── ruff.toml                     # Configuração de linters e formatação estática
├── pytest.ini                    # Configuração da suíte de testes
├── README.md                     # Documentação geral do repositório
└── AI_USAGE.md                   # Declaração transparente de uso de IA generativa
```

---

## 3. Componentes Implementados na Biblioteca (`mlp/`)

### 3.1. Estratégias de Inicialização de Pesos (`mlp/inicializadores.py`)
- **Zeros:** Inicialização nula, utilizada para vieses.
- **GlorotUniforme (Xavier Uniform):** $U(-\sqrt{6/(f_{\text{in}} + f_{\text{out}})}, \sqrt{6/(f_{\text{in}} + f_{\text{out}})})$.
- **GlorotNormal (Xavier Normal):** $\mathcal{N}(0, 2/(f_{\text{in}} + f_{\text{out}}))$.
- **HeNormal (Kaiming Normal):** $\mathcal{N}(0, 2/f_{\text{in}})$, ideal para retificadores (`ReLU`, `LeakyReLU`, `Swish`).

### 3.2. Funções de Ativação Analíticas (`mlp/ativacoes.py`)
Todas as funções contam com formulações numericamente estáveis para evitar *overflow* ou *underflow* e derivadas analíticas manuais:
1. **Sigmoide:** $\sigma(z) = \frac{1}{1 + e^{-|z|}}$ com derivada $\sigma(z)(1 - \sigma(z))$.
2. **LeakyReLU:** $z$ se $z > 0$, senão $\alpha z$ ($\alpha = 0.01$ por padrão).
3. **Swish (SiLU):** $z \cdot \sigma(z)$ com derivada $f(z) + \sigma(z)(1 - f(z))$.
4. **Softmax:** Exponencial estabilizada por $\max(z)$ e produto vetor-Jacobiana analítico.
5. **Tanh:** Implementação analítica elementar $2\sigma(2z) - 1$ (**estritamente sem `np.tanh`**).
6. **ELU:** $z$ se $z > 0$, senão $\alpha(e^z - 1)$ via `np.expm1`.
7. **Softplus:** $\ln(1 + e^z)$ calculado via `log1p` e derivada $\sigma(z)$.
8. **Softsign:** $\frac{z}{1 + |z|}$ com derivada $\frac{1}{(1 + |z|)^2}$.

### 3.3. Funções de Custo (`mlp/custos.py`)
- **MSE (Erro Quadrático Médio):** Perda quadrática contínua e gradiente $\frac{2}{N}(\hat{y} - y)$.
- **BCE (Entropia Cruzada Binária):** Perda logarítmica com truncamento estável $[\epsilon, 1 - \epsilon]$.
- **CCE (Entropia Cruzada Categórica):** Perda logarítmica multinomial compatível com One-Hot ou índices inteiros.

### 3.4. Camada Linear Afim (`mlp/camadas.py`)
- **Densa:** Transformação afim $Z = XW + b$, armazenamento de contexto de ativações e cálculo analítico exato dos gradientes $\frac{\partial L}{\partial W} = X^T \delta$, $\frac{\partial L}{\partial b} = \sum \delta$ e $\frac{\partial L}{\partial X} = \delta W^T$.

### 3.5. Orquestração e Otimizador (`mlp/rede.py`)
- **RedeNeural:** Encadeamento modular flexível, compilação de perda e treinamento via **Mini-batch Stochastic Gradient Descent (SGD)** com embaralhamento aleatório por época (*shuffle*), cálculo de perda em validação e suporte a semente determinística para reprodutibilidade.

### 3.6. Auditoria Matemática de Gradientes (`tests/checagem_gradientes.py`)
- Verificação numérica rigorosa de gradientes via **Diferenças Finitas Centradas** com erro de truncamento de segunda ordem $\mathcal{O}(\epsilon^2)$ e cálculo de erro relativo euclidiano $L_2$ normalizado pelo padrão de Stanford/CS231n ($\text{tolerância} < 10^{-6}$).

---

## 4. Instruções de Instalação e Execução

### 4.1. Configuração do Ambiente Virtual

Recomenda-se utilizar Python 3.10 ou superior:

```bash
# Criação do ambiente virtual
python3 -m venv .venv

# Ativação do ambiente virtual
source .venv/bin/activate  # No Linux/macOS
# .venv\Scripts\activate   # No Windows
```

### 4.2. Instalação das Dependências

O projeto divide rigorosamente suas dependências:

```bash
# Para utilizar apenas a biblioteca mlp (NumPy puro):
pip install -r requirements.txt

# Para desenvolvimento e execução dos testes (pytest, ruff):
pip install -r requirements-dev.txt

# Para executar o caderno do relatório técnico (matplotlib, pandas, scikit-learn):
pip install -r requirements-notebook.txt
```

---

## 5. Execução da Suíte de Testes e Linters

### 5.1. Testes Unitários Automatizados

Para executar os 164 testes unitários com cobertura de código:

```bash
pytest tests/ -v
```

Para gerar relatório detalhado de cobertura:

```bash
pytest tests/ --cov=mlp --cov-report=term-missing
```

### 5.2. Verificação de Código e Formatação com Ruff

```bash
# Verificação estática de código e boas práticas
ruff check mlp/ tests/

# Checagem de formatação
ruff format --check mlp/ tests/
```

---

## 6. Relatório Técnico Experimental (`relatorio_tecnico.ipynb`)

O arquivo [`relatorio_tecnico.ipynb`](file:///home/eduardo-homelab/projects/projeto-final-mlp/relatorio_tecnico.ipynb) contém a demonstração experimental completa da biblioteca nos três domínios supervisionados obrigatórios:

| Experimento | Conjunto de Dados | Arquitetura MLP | Função de Custo | Métrica Principal no Teste |
| :--- | :--- | :--- | :--- | :--- |
| **1. Regressão** | California Housing | $8 \to 64 \to 32 \to 1$ (Swish, LeakyReLU, Linear) | MSE | $R^2 = 70.26\%$, $\text{MAE} = 0.4426$ |
| **2. Classificação Binária** | Breast Cancer Wisconsin | $30 \to 32 \to 16 \to 1$ (LeakyReLU, Swish, Sigmoide) | BCE | $\text{Acurácia} = 97.67\%$, $\text{F1} = 96.77\%$ |
| **3. Classificação Multiclasse** | Wine Recognition (3 classes) | $13 \to 24 \to 16 \to 3$ (LeakyReLU, Tanh, Softmax) | CCE | $\text{Acurácia} = 96.30\%$, $\text{F1 Macro} = 96.59\%$ |

Todas as células encontram-se executadas e salvas com curvas de aprendizado, matrizes de confusão, gráficos de resíduos e a seção reflexiva **"Desafios Enfrentados e Aprendizados"**.

---

## 7. Conformidade Acadêmica e Uso de IA

- **Restrições da Disciplina:** A biblioteca `mlp/` respeita 100% das regras acadêmicas, não importando nem utilizando `sklearn`, `torch`, `autograd`, `tensorflow` ou similares. A função `Tanh` foi implementada analiticamente sem utilizar `np.tanh`.
- **Declaração de Uso de IA:** O detalhamento do uso de assistentes de IA (AGY / Gemini / Claude) em planejamento, infraestrutura e revisão, acompanhado da declaração de autoria humana e responsabilidade de defesa na arguição oral, encontra-se documentado em [`AI_USAGE.md`](file:///home/eduardo-homelab/projects/projeto-final-mlp/AI_USAGE.md).