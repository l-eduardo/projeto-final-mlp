# Projeto Final: Biblioteca de Redes Neurais (MLP)

**INE5664 — Aprendizado de Máquina (UFSC)**

Este repositório fornece a estrutura inicial (*starter kit*) para o desenvolvimento da biblioteca. O objetivo da equipe é preencher os esqueletos da biblioteca e elaborar os experimentos no Jupyter Notebook.

## Regras do Código-Fonte (`mlp/`)

* **Contrato Fixo (`mlp/base.py`):** **Não altere** as interfaces abstratas deste arquivo. As assinaturas dos métodos são utilizadas pelos testes automatizados de avaliação.
* **Preenchimento dos Esqueletos:** Substitua as exceções `raise NotImplementedError` pelas suas implementações nas seguintes classes:
  * `mlp/rede.py`: Gerenciamento das camadas, propagação direta, retropropagação e laço de treino (Mini-batch SGD).
  * `mlp/camadas.py`: Combinador linear, cálculo de gradientes dos parâmetros e propagação do erro.
  * `mlp/ativacoes.py`: Implementação das funções de ativação e suas derivadas (mínimo de 8 funções).
  * `mlp/custos.py`: Cálculo do erro escalar e gradiente em relação à saída (mínimo de 2 funções).
  * `mlp/inicializadores.py`: Estratégias de inicialização de matrizes de pesos (mínimo de 3 estratégias).
* **Restrição de Bibliotecas:** O pacote `mlp/` pode utilizar **exclusivamente NumPy**. O uso de qualquer framework de ML ou diferenciação automática dentro da pasta `mlp/` é estritamente proibido.

## Relatório Técnico (`notebook.ipynb`)

* Preencha as seções obrigatórias e entregue o arquivo **executado e reproduzível**, mantendo todas as células de saída e gráficos salvos.
* O uso de `pandas`, `scikit-learn` e `matplotlib` é permitido **exclusivamente no notebook** para carga de dados, divisão de conjuntos, métricas auxiliares e gráficos.
