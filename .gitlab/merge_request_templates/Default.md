## Descrição das Alterações

<!-- Resumo claro do que foi implementado ou alterado. -->

## Motivação e Contexto

<!-- Qual problema esta mudança resolve ou qual funcionalidade adiciona? -->

## Issues Relacionadas

<!-- Exemplo: Closes #6, Relacionada a #7 -->
- Closes #

## Referências para Estudo e Revisão (Opcional)

<!-- Links para artigos, documentação ou notas de aula que auxiliem os revisores na compreensão da técnica implementada. -->
- 

## Tipo de Mudança

- [ ] `feat`: Nova funcionalidade ou módulo
- [ ] `fix`: Correção de bug
- [ ] `refactor`: Refatoração sem alteração de comportamento
- [ ] `test`: Adição ou melhoria de testes
- [ ] `docs`: Documentação
- [ ] `ci`: Pipeline de CI/CD ou automação
- [ ] `chore`: Manutenção ou governança

---

## Checklist de Conformidade Acadêmica

- [ ] **Exclusividade NumPy no core `mlp/`:** Zero imports de bibliotecas de ML ou autograd (`torch`, `sklearn`, `scipy`, `autograd`, `tensorflow`, `jax`, etc.).
- [ ] **Proibição de `np.tanh`:** Funções de ativação calculadas a partir de operações aritméticas elementares.
- [ ] **Contratos de `mlp/base.py` preservados:** Nenhuma interface abstrata foi alterada.
- [ ] **Isolamento de dependências:** Dependências auxiliares mantidas exclusivamente nos grupos opcionais do `pyproject.toml`.
- [ ] **Pipeline de CI aprovada:** Testes e linters passam sem erros.
- [ ] **Compreensão integral do código:** Todo o código submetido foi revisado e é compreendido pelos integrantes do grupo.

---

## Verificação Local

Comandos executados para validar as alterações antes de abrir o MR:

```bash
ruff check .
ruff format --check .
pytest tests/
```
