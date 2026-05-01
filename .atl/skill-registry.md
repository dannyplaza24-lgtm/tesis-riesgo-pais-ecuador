# Skill Registry — tesis-riesgo-pais-ecuador

Generated: 2026-04-05
Persistence: openspec (file-based)

## User Skills

| Skill | Trigger | Path |
|-------|---------|------|
| `sdd-init` | Inicializar SDD en el proyecto | `~/.claude/skills/sdd-init/SKILL.md` |
| `sdd-explore` | Investigar el codebase, explorar notebooks, clarificar requerimientos | `~/.claude/skills/sdd-explore/SKILL.md` |
| `sdd-propose` | Crear o actualizar propuesta de cambio con intención y alcance | `~/.claude/skills/sdd-propose/SKILL.md` |
| `sdd-spec` | Escribir especificaciones con requerimientos y escenarios | `~/.claude/skills/sdd-spec/SKILL.md` |
| `sdd-design` | Escribir diseño técnico con decisiones de arquitectura | `~/.claude/skills/sdd-design/SKILL.md` |
| `sdd-tasks` | Descomponer un cambio en checklist de tareas de implementación | `~/.claude/skills/sdd-tasks/SKILL.md` |
| `sdd-apply` | Implementar tareas siguiendo specs y diseño | `~/.claude/skills/sdd-apply/SKILL.md` |
| `sdd-verify` | Validar que la implementación cumple specs y tareas | `~/.claude/skills/sdd-verify/SKILL.md` |
| `sdd-archive` | Archivar un cambio completado | `~/.claude/skills/sdd-archive/SKILL.md` |

## Project Conventions

No AGENTS.md, CLAUDE.md (project-level), or .cursorrules found in project root.

## Compact Rules

### Python / Jupyter Notebooks

- NEVER use random splits for time series — always chronological (train < val < test by date)
- ALWAYS save intermediate datasets as .pkl at the end of each notebook
- Column naming convention: `target_`, `macro_`, `nlp_`, `oil_`, `us_`, `gold_`, `volatility_`, `index_`, `etf_`
- Forward-fill (ffill) is the approved imputation strategy for monthly/quarterly variables
- Dataset date range: 2013-01-01 to 2025-12-31; trading days only (3047 rows × 22 cols)
- Baseline to beat: ARIMA rolling RMSE 107.48 pb / MAE ~70 pb on test set
- Data leakage rule: test set starts after 2025-01-xx — never let test dates bleed into train
- NLP features come from GDELT v2 (10 columns with `nlp_` prefix); relationship to EMBI is contemporaneous (no Granger lag detected)
- Variables excluded: `macro_remesas`, `macro_prod_petroleo` (temporal coverage gap)
- Split ratio: 80% train / 10% val / 10% test (chronological index slicing)
