## Descrição das Alterações

<!-- Forneça um resumo claro e conciso das alterações implementadas neste Merge Request. -->

## Motivação e Contexto

<!-- Explique por que esta mudança é necessária. Qual problema ela resolve ou qual funcionalidade adiciona? -->

## Issues Relacionadas

<!-- Referencie as issues aplicáveis (ex.: Closes #1, Relacionada a #2, #3, #4, #5) -->
- Closes #
- Relacionada a: #1, #2, #3, #4, #5

## Tipo de Mudança

- [ ] `feat`: Nova funcionalidade ou módulo
- [ ] `fix`: Correção de defeito/bug
- [ ] `refactor`: Refatoração sem alteração de comportamento
- [ ] `test`: Adição ou melhoria de testes
- [ ] `docs`: Atualização de documentação
- [ ] `ci`: Alterações nos pipelines de CI/CD ou scripts de automação
- [ ] `chore`: Tarefas de manutenção ou governança

---

## Checklist de Conformidade Acadêmica (MANDATÓRIO)

Todos os itens abaixo são obrigatórios para a aprovação do Merge Request e integração na branch `main`:

- [ ] **Exclusividade NumPy no core `mlp/`:** Zero imports de bibliotecas de ML ou diferenciação automática (`torch`, `sklearn`, `scipy`, `autograd`, `tensorflow`, `jax`, `keras`).
- [ ] **Proibição estrita de `np.tanh` / `numpy.tanh`:** Funções de ativação (incluindo tangente hiperbólica) implementadas manualmente a partir de aritmética elementar de exponenciais.
- [ ] **Imutabilidade dos contratos abstratos de `mlp/base.py`:** Nenhuma interface ou assinatura de método público em `mlp/base.py` foi alterada.
- [ ] **Isolamento de dependências:** Dependências de `dev` e `notebook` estritamente segregadas no `pyproject.toml`, mantendo `numpy` como dependência única do core da biblioteca.
- [ ] **Pipeline CI verde:** Todos os estágios da esteira (`lint`, `audit` AST e `test` com pytest) passam com sucesso em ambiente limpo.
- [ ] **Reprodutibilidade do relatório técnico:** O arquivo `relatorio_tecnico.ipynb` executa de ponta a ponta sem erros e mantém células de saída/gráficos salvas.
- [ ] **Transparência do uso de IA documentada:** Todas as interações com ferramentas de Inteligência Artificial estão registradas de forma detalhada e honesta em `AI_USAGE.md`.

---

## Verificação e Testes Locais

Descreva os comandos executados para validar localmente as alterações antes da abertura do MR:

```bash
# 1. Verificação de formatação e linting
ruff check .

# 2. Auditoria estática de AST (conformidade do código mlp/)
python3 scripts/audit_ast.py

# 3. Execução da suíte de testes com cobertura
pytest tests/
```

- [ ] Linting executado e sem alertas (`ruff check .`)
- [ ] Auditoria de AST executada com código de saída 0 (`python3 scripts/audit_ast.py`)
- [ ] Testes unitários/integração executados com 100% de sucesso (`pytest tests/`)

## Notas Adicionais e Riscos

<!-- Descreva qualquer consideração especial, dependências externas ou impactos colaterais esperados. -->
