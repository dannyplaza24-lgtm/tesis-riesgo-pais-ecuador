# Proposal: Análisis de Regímenes del EMBI

## Intent

Responder la observación del tutor sobre exogeneidad del EMBI ecuatoriano mediante análisis formal de quiebres estructurales y comparación de importancia de variables por régimen temporal. El modelo actual trata 2013-2025 como un régimen homogéneo, ignorando que COVID y la reestructuración 2020 pueden haber cambiado fundamentalmente qué variables determinan el riesgo país.

## Scope

### In Scope
- Test de Chow en fechas clave: 2020-03-01 (COVID) y 2022-01-01 (post-reestructuración)
- CUSUM / Quandt-Andrews para detección de quiebre óptimo (sin fecha preimpuesta)
- SHAP values segmentados: aplicar modelo entrenado sobre subconjuntos pre-COVID (2013-2019) y post-COVID (2022-2025) y comparar top-10 variables por régimen
- Tabla comparativa de dominancia de variables externas vs locales por período
- Narrative sobre el default Correa 2008 como contexto en la discusión (sin modelar)

### Out of Scope
- Extender dataset a GDELT v1 (2004-2012) para incluir el default 2008
- Reentrenar modelos separados por régimen (modelo ya entrenado se reutiliza para SHAP)
- Modificar el pipeline de feature engineering (NB03)

## Approach

Nuevo `notebooks/07_Regimenes_Estructurales.ipynb`:
1. Cargar `dataset_tesis_final.pkl` + modelo entrenado (NB04)
2. Test de Chow sobre residuos del ARIMA/modelo en fechas candidatas
3. SHAP sobre subsets cronológicos usando el modelo XGBoost ya entrenado
4. Visualización: heatmap de importancia por régimen + tabla de cambio de ranking

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `notebooks/07_Regimenes_Estructurales.ipynb` | New | Notebook de análisis de quiebres |
| `notebooks/dataset_tesis_final.pkl` | Read-only | Datos base sin modificar |
| `notebooks/` (modelos .pkl de NB04) | Read-only | Modelo XGBoost reutilizado para SHAP |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Test de Chow requiere residuos del modelo — modelo no serializado explícitamente | Med | Cargar weights del mejor modelo desde NB04 o re-fitear rápido con mismos hiperparámetros |
| Subperiodo post-COVID tiene pocos datos (2022-2025 ≈ 700 días) — SHAP menos estable | Med | Reportar n por subperiodo; usar beeswarm + tabla de medias en lugar de valores individuales |
| El análisis puede mostrar que el NLP no aporta en ningún régimen | Low | Resultado válido y publicable; refuerza la narrativa de exogeneidad |

## Rollback Plan

Notebook independiente — no modifica ningún artefacto existente. Si el análisis no aporta, simplemente no se incluye en la tesis.

## Dependencies

- Modelo XGBoost serializado de NB04 (o re-fit con mismos hiperparámetros)
- `train_feature_engineered.pkl`, `val_feature_engineered.pkl`, `test_feature_engineered.pkl`
- `dataset_tesis_final.pkl`

## Success Criteria

- [ ] Test de Chow ejecutado con p-value reportado para 2020-03 y 2022-01
- [ ] SHAP comparativo generado para al menos 2 subperiodos (pre/post COVID)
- [ ] Tabla o gráfico que muestre si variables externas dominan más que locales en algún régimen
- [ ] Sección "Discusión del default 2008" redactada en el notebook (markdown cell)
- [ ] RMSE del modelo en cada subperiodo reportado (¿el modelo funciona igual en todos los regímenes?)
