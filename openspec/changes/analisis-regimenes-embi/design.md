# Design: Análisis de Regímenes del EMBI

## Arquitectura del Notebook

`notebooks/07_Regimenes_Estructurales.ipynb` — notebook independiente, no modifica artefactos existentes.

## Secciones y Decisiones Técnicas

### 1. Imports y Carga de Datos

```python
# Librerías
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import shap, pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.structural import breaks_cusumolsresid
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Datos
df = pd.read_pickle('dataset_tesis_final.pkl')  # 3047 × 22
train = pd.read_pickle('train_feature_engineered.pkl')
val   = pd.read_pickle('val_feature_engineered.pkl')
test  = pd.read_pickle('test_feature_engineered.pkl')
with open('feature_selector_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)
feats = meta['selected_features']  # 60 features
```

### 2. Test de Chow

**Decisión técnica:** `statsmodels` no tiene `chow_test` directo; se implementa manualmente via comparación de RSS:

```
F = ((RSS_total - RSS_1 - RSS_2) / k) / ((RSS_1 + RSS_2) / (n - 2k))
```

Donde RSS se calcula con una regresión OLS de `target_embi` sobre lags AR (1,2,3) en cada submuestra.

**Alternativa explorada:** `linearmodels` — descartada por dependencia adicional. OLS de statsmodels es suficiente y ya está instalado.

**Fechas candidatas:**
```python
BREAK_DATES = ['2020-03-01', '2022-01-01']
```

### 3. CUSUM / Quandt-Andrews (detección de quiebre óptimo)

Usar `breaks_cusumolsresid` de statsmodels sobre los residuos del modelo OLS AR(3):

```python
# OLS AR(3) para CUSUM
y = df['target_embi'].dropna()
X = pd.DataFrame({'lag1': y.shift(1), 'lag2': y.shift(2), 'lag3': y.shift(3)}).dropna()
y = y.loc[X.index]
model_ols = OLS(y, add_constant(X)).fit()
# CUSUM
cusum_result = breaks_cusumolsresid(model_ols.resid)
```

Para Quandt-Andrews: iterar sobre fechas en el rango [15%, 85%] del período y reportar la fecha con F máximo (implementación manual con el mismo F de Chow).

### 4. Modelo XGBoost para SHAP

**Hiperparámetros exactos de NB04:**
```python
XGB_PARAMS = dict(
    colsample_bytree=0.9798,
    gamma=0.7354,
    learning_rate=0.0927,
    max_depth=4,
    min_child_weight=1,
    n_estimators=607,
    reg_alpha=0.6624,
    reg_lambda=1.4586,
    subsample=0.8483,
    objective='reg:squarederror',
    random_state=42, n_jobs=-1
)
```

**Estrategia SHAP por subperiodo:**

Opción seleccionada: re-fitear el modelo sobre el subset de train que cae en cada subperiodo, luego calcular SHAP sobre el correspondiente subset del dataset. Esto es más limpio metodológicamente que usar un único modelo global.

```python
SUBPERIODOS = {
    'Pre-COVID (2013-2019)': ('2013-01-01', '2020-02-28'),
    'COVID (2020-2021)':     ('2020-03-01', '2021-12-31'),
    'Post-COVID (2022-2025)': ('2022-01-01', '2025-12-31'),
}
TARGET = 'target_future'
```

Para cada subperiodo:
1. Filtrar `train_val = pd.concat([train, val])` por fechas del subperiodo
2. Si n < 200 días → reportar advertencia pero ejecutar igual
3. Re-fitear XGBoost con mismos hiperparámetros
4. Calcular SHAP con `shap.TreeExplainer`
5. Calcular mean(|SHAP|) por feature → ranking

### 5. Visualizaciones

- **Fig 1:** Serie EMBI con bandas de colores por régimen + líneas verticales en fechas de quiebre
- **Fig 2:** CUSUM plot con bandas de confianza (generado por statsmodels)
- **Fig 3:** Heatmap de SHAP mean(|value|) — filas=features top-10 global, columnas=subperiodos
- **Fig 4:** Bar chart comparativo de RMSE por subperiodo

### 6. Tabla Comparativa de Dominancia

```python
# Para cada subperiodo, top-5 variables por SHAP medio absoluto
# Columnas: Variable | Pre-COVID rank | COVID rank | Post-COVID rank | Tipo (Externo/Local)
```

Clasificación de variables:
- **Externas:** `oil_wti`, `us_treasury_10y`, `volatility_vix`, `gold`, `index_dxy`, `etf_emb`, `etf_hyg`
- **Locales macro:** `macro_*`
- **NLP:** `nlp_*`
- **AR:** lags del EMBI

## Output Files

- `notebooks/07_Regimenes_Estructurales.ipynb`
- `notebooks/chow_cusum_analysis.png`
- `notebooks/shap_regimenes_heatmap.png`
- `notebooks/rmse_por_regimen.png`

## Dependencias de Librerías

Todas ya instaladas en `tesis_env`:
- `statsmodels` — Chow / CUSUM
- `shap` — SHAP values
- `xgboost` — modelo
- `sklearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
