# Instruções e Restrições do Projeto para Assistentes de IA (AGY / Claude / Gemini)

> [!CAUTION]
> **DIRETIVA CRÍTICA DE DESENVOLVIMENTO — RISCO DE NOTA 0,0**
> 
> Para ver em detalhes o que **NUNCA PODE SER FEITO** sob risco de nota zero imediata, confira o arquivo:
> 👉 [`docs/restricoes_nota_zero.md`](file:///home/luizeduardo/_dev/faculdade/aprendizado_maquinas/projeto-final-mlp/docs/restricoes_nota_zero.md)

---

## Regras Obrigatórias para qualquer Agente de IA

1. **NUNCA violar as restrições acadêmicas do projeto**:
   - Dentro de `mlp/`: **apenas NumPy** é permitido. Nunca sugira, adicione imports ou utilize `sklearn`, `torch`, `scipy`, `tensorflow`, `jax`, `autograd` ou similares.
   - Todo forward e backward deve ser analítico e manual.
   - NUNCA use `np.tanh` nas ativações da biblioteca (vedado pelo enunciado).
   - Não adicione bibliotecas externas além de `numpy` nas dependências da biblioteca `mlp` (em `requirements.txt`).

2. **Protocolo em caso de risco ou esbarrar em violação**:
   - **PARE IMEDIATAMENTE**.
   - **NÃO execute** o comando nem escreva o código.
   - **AVISE O DEV** com clareza em caixa alta sobre qual restrição seria violada e o risco iminente de nota zero, pedindo instruções antes de prosseguir.

3. **Autonomia nas decisões técnicas**:
   - As issues e tarefas do projeto estabelecem os requisitos e objetivos (ex.: "implementar 3 inicializadores", "implementar lote de ativações", "implementar funções de custo"), acompanhadas de referências para estudo.
   - O desenvolvedor/agente responsável pela issue tem autonomia para decidir quais abordagens específicas utilizar, respeitando as restrições gerais e o contrato de interfaces em `mlp/base.py`.
