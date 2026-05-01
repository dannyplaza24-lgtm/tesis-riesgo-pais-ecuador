# Spec: Análisis de Regímenes del EMBI

## REQ-1: Test de Chow de Quiebre Estructural

El sistema SHALL ejecutar un test de Chow sobre los residuos del modelo de referencia en fechas candidatas configurables.

**Fechas candidatas obligatorias:** 2020-03-01 (inicio COVID), 2022-01-01 (post-reestructuración).

#### Scenario: Quiebre detectado en COVID
- GIVEN el dataset `dataset_tesis_final.pkl` con serie `target_embi` continua 2013-2025
- WHEN se ejecuta el test de Chow con fecha de corte 2020-03-01
- THEN se reporta un F-statistic y p-value
- AND si p < 0.05 se concluye quiebre estructural significativo
- AND si p >= 0.05 se reporta ausencia de quiebre con ese umbral

#### Scenario: Detección de quiebre óptimo (Quandt-Andrews)
- GIVEN la serie `target_embi`
- WHEN se ejecuta QLRTest / Quandt-Andrews sobre el 15%-85% central del período
- THEN se identifica la fecha con mayor estadístico F
- AND se reporta si esa fecha coincide con eventos conocidos (COVID, elecciones, default)

---

## REQ-2: SHAP por Subperiodo

El sistema SHALL generar SHAP values usando el modelo XGBoost entrenado en NB04, aplicado sobre subconjuntos cronológicos del dataset.

**Subperiodos obligatorios:**
- Pre-COVID: 2013-01-01 → 2020-02-28
- COVID: 2020-03-01 → 2021-12-31
- Post-COVID: 2022-01-01 → 2025-12-31

#### Scenario: SHAP sobre subperiodo pre-COVID
- GIVEN modelo XGBoost con hiperparámetros fijos de NB04
- WHEN se re-fitea sobre el split train que cae en pre-COVID y se aplica SHAP al subset pre-COVID del test
- THEN se obtiene un DataFrame de SHAP values con shape (n_dias_subperiodo, n_features)
- AND se genera un beeswarm o bar plot del top-10 variables

#### Scenario: Comparación de dominancia externa vs local por régimen
- GIVEN SHAP values para al menos 2 subperiodos
- WHEN se calcula el SHAP medio absoluto por variable y por régimen
- THEN se produce una tabla con ranking de variables en cada régimen
- AND se puede identificar si variables externas (VIX, EMB, treasury) suben de ranking en crisis vs variables locales (macro_*)

---

## REQ-3: RMSE por Subperiodo

El sistema SHALL reportar RMSE y MAE del modelo XGBoost para cada subperiodo definido en REQ-2.

#### Scenario: Performance del modelo por régimen
- GIVEN predicciones del modelo sobre el dataset completo
- WHEN se segmentan por subperiodo
- THEN se calcula RMSE y MAE para cada uno
- AND se incluye en la tabla de resultados junto al RMSE global (49.01 pb)

---

## REQ-4: Narrative del Default Correa 2008

El sistema SHALL incluir una celda markdown con contextualización del default soberano de 2008 como antecedente histórico relevante para la hipótesis de exogeneidad.

#### Scenario: Contexto narrativo documentado
- GIVEN el análisis de regímenes completado
- WHEN se redacta la sección de discusión
- THEN se menciona el default de 2008 como precedente de disrupción exógena
- AND se aclara explícitamente que el dataset GDELT v2 inicia en 2013 y no cubre ese evento

---

## Constraints

- MUST NOT modificar ningún notebook o artefacto existente (NB01–NB06)
- MUST NOT usar datos de test para entrenar el modelo base
- SHOULD reutilizar el mismo objeto XGBoost con hiperparámetros de NB04 sin re-búsqueda
- MAY guardar figuras como `.png` en `notebooks/`
