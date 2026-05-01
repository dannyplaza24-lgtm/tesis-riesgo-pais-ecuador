# Design: Autoencoder NLP

## Arquitectura del Notebook

`notebooks/08_Autoencoder_NLP.ipynb` — notebook independiente, no modifica artefactos existentes.

## Secciones y Decisiones Técnicas

### 1. Imports y Carga de Datos

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# TensorFlow/Keras para el autoencoder
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
tf.random.set_seed(42)
np.random.seed(42)

# Datos
df = pd.read_pickle('dataset_tesis_final.pkl')
train = pd.read_pickle('train_feature_engineered.pkl')
val   = pd.read_pickle('val_feature_engineered.pkl')
test  = pd.read_pickle('test_feature_engineered.pkl')
with open('feature_selector_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)
```

### 2. Extracción de las 10 Variables NLP Originales

```python
NLP_COLS_ORIG = [
    'nlp_avg_tone', 'nlp_goldstein', 'nlp_event_count', 'nlp_std_tone',
    'nlp_pct_positive', 'nlp_pct_negative', 'nlp_total_articles',
    'nlp_events_gov', 'nlp_events_biz', 'nlp_events_igo'
]
# Extraer de dataset_tesis_final.pkl (no del feature-engineered que tiene lags)
# Split por fechas alineadas con train/val/test
train_nlp = df.loc[train.index, NLP_COLS_ORIG].fillna(method='ffill')
val_nlp   = df.loc[val.index,   NLP_COLS_ORIG].fillna(method='ffill')
test_nlp  = df.loc[test.index,  NLP_COLS_ORIG].fillna(method='ffill')
```

### 3. Normalización (Solo en Train)

```python
scaler_nlp = StandardScaler()
X_train_nlp = scaler_nlp.fit_transform(train_nlp)   # fit aquí
X_val_nlp   = scaler_nlp.transform(val_nlp)          # solo transform
X_test_nlp  = scaler_nlp.transform(test_nlp)         # solo transform
```

### 4. Arquitectura del Autoencoder

**Decisión:** Functional API de Keras para separar encoder/decoder limpiamente.

```python
# Encoder
inputs = layers.Input(shape=(10,))
x = layers.Dense(5, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
latent = layers.Dense(3, activation='relu', name='latent')(x)

# Decoder
x = layers.Dense(5, activation='relu')(latent)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='linear')(x)

autoencoder = Model(inputs, outputs, name='autoencoder')
encoder = Model(inputs, latent, name='encoder')

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
```

**Entrenamiento:**
```python
es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = autoencoder.fit(
    X_train_nlp, X_train_nlp,        # autoencoder: input = target
    validation_data=(X_val_nlp, X_val_nlp),
    epochs=200, batch_size=32,
    callbacks=[es], verbose=0
)
```

### 5. Extracción de Latentes

```python
Z_train = encoder.predict(X_train_nlp)   # (n_train, 3)
Z_val   = encoder.predict(X_val_nlp)     # (n_val, 3)
Z_test  = encoder.predict(X_test_nlp)    # (n_test, 3)

# Columnas nombradas
ae_cols = ['nlp_ae_0', 'nlp_ae_1', 'nlp_ae_2']
```

### 6. XGBoost con Latentes del Autoencoder

**Hiperparámetros XGBoost (idénticos a NB05 Grupo D para comparativa controlada):**
```python
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0,
    random_state=42, n_jobs=-1, verbosity=0
)
```

**Nota:** Se usan los hiperparámetros de NB05 (no los tuneados de NB04) para mantener la comparativa controlada con el Grupo D original (57.48 pb).

```python
# Reemplazar las 4 vars NLP originales por las 3 latentes AE
NLP_ORIG_4 = ['nlp_event_count_roll_mean30', 'nlp_total_articles_roll_mean30',
              'nlp_events_gov_roll_mean30', 'nlp_events_biz_roll_mean30']
non_nlp_feats = [f for f in feats if f not in NLP_ORIG_4]  # 56 features

# Construir X con latentes
X_train_ae = pd.concat([train[non_nlp_feats], pd.DataFrame(Z_train, index=train.index, columns=ae_cols)], axis=1)
# ... igual para val y test
```

### 7. Baseline PCA-3

```python
pca = PCA(n_components=3, random_state=42)
P_train = pca.fit_transform(X_train_nlp)   # fit solo en train
P_val   = pca.transform(X_val_nlp)
P_test  = pca.transform(X_test_nlp)
pca_cols = ['nlp_pca_0', 'nlp_pca_1', 'nlp_pca_2']
```

Mismo pipeline XGBoost con PCA latentes en lugar de AE latentes.

### 8. Heatmap de Correlación

```python
# Correlación entre dims latentes y variables NLP originales (todo el dataset)
Z_full = encoder.predict(scaler_nlp.transform(df.loc[train.index.union(val.index).union(test.index), NLP_COLS_ORIG].fillna(method='ffill')))
corr_df = pd.DataFrame(
    np.corrcoef(Z_full.T, scaler_nlp.transform(...).T)[:3, 3:],
    index=ae_cols, columns=NLP_COLS_ORIG
)
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0)
```

### 9. Tabla Resumen Final

| Variante | Features NLP | RMSE Test | MAE Test | R² Test |
|----------|-------------|-----------|----------|---------|
| Grupo D original (NB05) | 4 (rolling 30d) | 57.48 | 37.70 | 0.9643 |
| PCA-3 | 3 componentes lineales | ? | ? | ? |
| Autoencoder-3 | 3 dims latentes no-lineales | ? | ? | ? |

## Verificación de TensorFlow

```python
# Al inicio del notebook
try:
    import tensorflow as tf
    print(f'TF: {tf.__version__}')
except ImportError:
    print('Instalar: pip install tensorflow')
    raise
```

Si TF no está disponible: alternativa con `sklearn.neural_network.MLPRegressor` como autoencoder manual (más lento pero sin dependencia extra).

## Output Files

- `notebooks/08_Autoencoder_NLP.ipynb`
- `notebooks/nlp_autoencoder.keras` (modelo guardado)
- `notebooks/autoencoder_loss_curve.png`
- `notebooks/nlp_latentes_correlacion.png`
- `notebooks/autoencoder_comparativa_rmse.png`
