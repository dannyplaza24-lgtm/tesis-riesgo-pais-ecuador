# Tasks: Análisis de Regímenes del EMBI

## NB07 — `notebooks/07_Regimenes_Estructurales.ipynb`

### 1. Setup
- [ ] 1.1 Imports (statsmodels, shap, xgboost, sklearn, matplotlib, seaborn, pickle)
- [ ] 1.2 Cargar `dataset_tesis_final.pkl`, splits `.pkl`, `feature_selector_metadata.pkl`
- [ ] 1.3 Definir SUBPERIODOS y BREAK_DATES como constantes configurables

### 2. Test de Chow
- [ ] 2.1 Implementar función `chow_test(series, break_date)` → (F_stat, p_value)
- [ ] 2.2 Correr test para 2020-03-01 y 2022-01-01
- [ ] 2.3 Imprimir tabla: fecha | F_stat | p-value | ¿Significativo?

### 3. CUSUM y Quandt-Andrews
- [ ] 3.1 Ajustar OLS AR(3) sobre `target_embi` completo
- [ ] 3.2 Calcular CUSUM sobre residuos (`breaks_cusumolsresid`)
- [ ] 3.3 Implementar iteración Quandt-Andrews (15%-85%) → fecha óptima de quiebre
- [ ] 3.4 Visualizar CUSUM con bandas de confianza al 95% → guardar `chow_cusum_analysis.png`

### 4. SHAP por Subperiodo
- [ ] 4.1 Re-fitear XGBoost (hiperparámetros NB04) sobre train subset por subperiodo
- [ ] 4.2 Calcular SHAP con `TreeExplainer` para Pre-COVID, COVID, Post-COVID
- [ ] 4.3 Calcular `mean(|SHAP|)` por feature y subperiodo
- [ ] 4.4 Generar heatmap de SHAP por régimen (top-15 global) → `shap_regimenes_heatmap.png`

### 5. Tabla de Dominancia Externa vs Local
- [ ] 5.1 Clasificar variables: Externas / Macro local / NLP / AR
- [ ] 5.2 Calcular % de importancia SHAP por categoría para cada subperiodo
- [ ] 5.3 Imprimir tabla comparativa de ranking

### 6. RMSE por Subperiodo
- [ ] 6.1 Generar predicciones del modelo completo sobre cada subperiodo
- [ ] 6.2 Calcular RMSE y MAE por subperiodo
- [ ] 6.3 Bar chart comparativo → `rmse_por_regimen.png`

### 7. Markdown de Discusión
- [ ] 7.1 Celda markdown: contexto del default Correa 2008 y por qué el dataset empieza en 2013
- [ ] 7.2 Celda markdown: conclusiones del análisis de regímenes para la tesis
