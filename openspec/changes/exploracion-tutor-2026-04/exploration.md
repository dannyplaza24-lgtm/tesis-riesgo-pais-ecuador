# Exploration: Cuestiones planteadas por el tutor

**Fecha:** 2026-04-05
**Estado:** Exploración — sin cambios al código

---

## Estado Actual del Proyecto

- **Pipeline:** 6 notebooks secuenciales (01→06), completos
- **Dataset:** 3047 filas × 22 features, 2013-2025, días de trading
- **Split:** 80/10/10 cronológico (train: 2013-2024, val: 2024, test: 2024-11 → 2025-12)
- **Modelo principal:** XGBoost + RandomForest con RandomizedSearchCV, TimeSeriesSplit 10 folds
- **Mejor RMSE test:** 49.01 pb (Grupo C: AR + NLP) — mejora del 45.3% sobre ARIMA baseline (89.64 pb)
- **NLP seleccionado:** 4 variables de volumen de cobertura (rolling 30d): `nlp_event_count`, `nlp_total_articles`, `nlp_events_gov`, `nlp_events_biz`
- **Granger:** Sin causalidad NLP→EMBI con rezago; relación contemporánea confirmada
- **NB03:** Feature engineering con lags temporales del EMBI + rolling means; 60 features totales seleccionadas

---

## Cuestión 1 — Análisis de Regímenes: ¿El EMBI depende de factores exógenos? (Pre/Post COVID + Default 2008)

### Pregunta del tutor
El tutor plantea que el riesgo país de Ecuador podría estar determinado mayoritariamente por factores externos (precio del petróleo, tasas Fed, apetito de riesgo global) más que por fundamentos propios. Propone analizar si existen quiebres estructurales entre regímenes: pre-COVID, COVID, post-COVID y el evento del default de Correa en 2008.

### Diagnóstico sobre el modelo actual

**Limitación clave:** El modelo actual trata todo el período 2013-2025 como un régimen homogéneo. No hay ningún mecanismo de detección de cambios estructurales. El split 80/10/10 simplemente corta cronológicamente sin considerar eventos discretos.

**Evidencia relevante ya en los notebooks:**
- El EMBI Ecuador tiene picos muy marcados: ~2015 (caída del petróleo), 2020 (COVID + default técnico), 2022-2023 (reestructuración + elecciones)
- La ablación muestra que "AR + Macro" (RMSE 60.07) es peor que "Solo AR" (51.31) — sugiere que las variables macro locales generan **ruido** más que señal en el modelo global. Esto es consistente con la hipótesis del tutor: el EMBI ecuatoriano puede estar más ligado a factores sistémicos externos que a fundamentales locales.
- Las variables con mayor peso probable en SHAP son financieras externas (VIX, EMB ETF, US Treasury)

**Sobre el default 2008:** La ventana de datos actual empieza en 2013 (por disponibilidad GDELT v2). El evento Correa (2008-2009) está fuera del dataset. Incluirlo requeriría extender a GDELT v1 (2004-2012), que ya está disponible como `gdelt_20040101_20150218.csv`, pero con dimensiones diferentes (menos columnas NLP).

### Enfoques posibles

| Enfoque | Descripción | Pros | Contras | Esfuerzo |
|---------|-------------|------|---------|----------|
| **A — Test de Chow / CUSUM** | Test estadístico de quiebre estructural en fechas puntuales (2020-03, 2022-01) | Rigoroso, publicable, responde directamente al tutor | Solo dice si hay quiebre, no cómo cambia el modelo | Bajo |
| **B — Modelos por régimen (subsample)** | Entrenar modelos separados para pre-COVID (2013-2019) y post-COVID (2021-2025) y comparar feature importance | Muestra qué variables dominan en cada época | Dataset se fragmenta, menos datos por modelo | Medio |
| **C — Variable de régimen como feature** | Agregar dummies (pre_covid, covid, post_covid) o indicador continuo de volatilidad global | Sin fragmentar datos, el modelo aprende transiciones | No responde directamente a la pregunta de exogeneidad | Bajo |
| **D — Extender a 2008 con GDELT v1** | Usar `gdelt_20040101_20150218.csv` para incluir el default Correa | Enriquece el análisis histórico y es relevante para la tesis | GDELT v1 tiene menos columnas NLP; requiere reconciliar dos versiones | Alto |

### Recomendación Cuestión 1

**Aplicable y recomendado: Enfoque A + B** (combinado, bajo-medio esfuerzo)

1. **Test de Chow en NB02 o NB07 nuevo:** Verificar quiebre estructural en marzo 2020 y enero 2022. Si el test es significativo, justifica el análisis por régimen y fortalece el argumento metodológico de la tesis.
2. **Comparar feature importance SHAP entre subperiodos** (sin reentrenar, solo aplicar el modelo al subset pre-COVID vs post-COVID del test set). Esto muestra si las variables que explican el EMBI cambian entre regímenes — responde directamente al tutor.
3. **El default 2008 como contexto narrativo** en la introducción/discusión, sin necesariamente modelarlo (el dataset GDELT v2 que garantiza homogeneidad empieza en 2013).

**No recomendado para cierre de tesis:** Enfoque D (extender a GDELT v1) — complejidad alta por reconciliar versiones, riesgo de abrir una nueva línea antes del cierre.

---

## Cuestión 2 — Arquitectura NLP: Reducción de Dimensionalidad con Redes Neuronales

### Pregunta del tutor
El tutor propone usar capas de compresión neuronal (autoencoder o embedding) sobre las variables NLP para capturar relaciones no lineales entre ellas antes de alimentar al modelo principal.

### Diagnóstico sobre el modelo actual

**NLP actual:** 10 variables originales GDELT → 4 seleccionadas por feature selection (rolling 30d de volumen) → entran directamente a XGBoost.

**Problema con la propuesta tal como está:** XGBoost ya captura interacciones no lineales entre features. La reducción de dimensionalidad con autoencoder sobre solo 4-10 variables es de **beneficio marginal** — el espacio original es demasiado pequeño para que la compresión aporte.

**Dónde la propuesta tiene mérito real:** Si se trabaja con los embeddings crudos de texto (ej. BERT sobre títulos de noticias GDELT) o con las 10 variables originales sin feature selection previa, un autoencoder podría condensar señales correlacionadas en dimensiones latentes más informativas.

**Contexto del dataset actual:** Las 10 columnas NLP de GDELT v2 son **métricas agregadas diarias** (event_count, avg_tone, goldstein, etc.), no embeddings de texto. La correlación entre ellas es moderada-alta (event_count y total_articles están fuertemente correlacionados). Un autoencoder sobre estas 10 → 2-3 dims capturaría la estructura de correlación, pero el feature selector ya hizo algo similar.

### Enfoques posibles

| Enfoque | Descripción | Pros | Contras | Esfuerzo |
|---------|-------------|------|---------|----------|
| **A — Autoencoder sobre 10 NLP features** | Dense autoencoder (10→4→2→4→10), usar latent dim como features | Captura no-linealidades entre métricas NLP; fácil de implementar en Keras | Beneficio marginal con solo 10 inputs; añade complejidad sin garantía de mejora | Medio |
| **B — PCA + Autoencoder comparativo** | Comparar PCA (baseline lineal) vs autoencoder (no lineal) en las 10 NLP features | Comparación rigurosa; si autoencoder > PCA, confirma la no-linealidad | Requiere NB nuevo con experimento controlado | Medio |
| **C — Red neuronal end-to-end (MLP/LSTM)** | Reemplazar XGBoost por red neuronal que aprende representación NLP + macro conjuntamente | Captura dependencias temporales + no-linealidades; potencialmente mejor RMSE | Mayor riesgo de overfitting; más complejo de justificar y tunear; fuera del alcance si la tesis ya está cerrada con XGBoost | Alto |
| **D — Factorización matricial / NMF** | Descomposición no negativa de la matriz NLP-tiempo para extraer "temas latentes" | Interpretable (cada componente = patrón mediático) | Menos estándar; puede ser difícil de justificar metodológicamente | Medio |

### Recomendación Cuestión 2

**Aplicable con matices: Enfoque A o B si hay tiempo; de lo contrario, mencionar como trabajo futuro.**

- **Si el objetivo es fortalecer la tesis antes del cierre:** Implementar un autoencoder simple (10→3→10) en un NB07 nuevo, usar las 3 dimensiones latentes en lugar de las 4 variables NLP originales, y comparar RMSE con el Grupo D actual (57.48 pb). Si mejora → resultado nuevo. Si no → confirma que la representación lineal es suficiente para estos datos agregados (también es un hallazgo válido).
- **Si el tiempo es limitado:** La propuesta del tutor se puede responder en la sección de Trabajo Futuro con el argumento: "Las variables NLP actuales son métricas agregadas diarias que capturan volumen y tono; una arquitectura con embeddings de texto crudo (BERT/FinBERT sobre noticias GDELT) y capas de compresión representa una extensión natural..."
- **LSTM/temporal:** Si los datos son diarios y el EMBI tiene autocorrelación fuerte (ya demostrada en NB02), una LSTM sobre la secuencia completa (macro + NLP) podría superar a XGBoost. Esto requeriría reestructurar el pipeline.

---

## Áreas Afectadas

- `notebooks/04_Modelling_ML_SHAP.ipynb` — análisis SHAP por subperiodo (cuestión 1B)
- `notebooks/02_Modeling_Baseline.ipynb` — test de Chow (cuestión 1A)
- `notebooks/` — potencial NB07 para autoencoder NLP (cuestión 2A)
- `notebooks/dataset_tesis_final.pkl` — sin cambios necesarios
- `data/raw/gdelt_20040101_20150218.csv` — solo si se extiende a 2008 (cuestión 1D, no recomendado)

---

## Resumen Ejecutivo

| Cuestión | ¿Aplicable? | Esfuerzo | Recomendación |
|----------|-------------|----------|---------------|
| 1 — Regímenes pre/post COVID | ✅ Sí, y fortalece la tesis | Bajo-Medio | Test Chow + SHAP por subperiodo en NB nuevo |
| 1 — Default Correa 2008 | ⚠️ Parcialmente | Alto | Contexto narrativo en discusión; no modelar si se cierra pronto |
| 2 — Autoencoder NLP | ✅ Sí, técnicamente viable | Medio | NB07 experimental; mencionar trabajo futuro si no hay tiempo |
| 2 — LSTM end-to-end | ⚠️ Ambicioso | Alto | Trabajo futuro |

### Ready for Proposal
**Sí** para ambas cuestiones en versión simplificada. Se recomienda:
1. `/sdd-new analisis-regimenes-embi` — para la cuestión 1 (test de Chow + SHAP subperiodo)
2. `/sdd-new autoencoder-nlp` — para la cuestión 2 (si hay tiempo antes de cierre)
