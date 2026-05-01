# Spec: Autoencoder NLP — Compresión Neuronal

## REQ-1: Autoencoder Entrenado Sin Data Leakage

El sistema SHALL entrenar el autoencoder exclusivamente con el split de entrenamiento (`train_feature_engineered.pkl`), sin ver validación ni test.

#### Scenario: Entrenamiento sin leakage
- GIVEN las 10 variables NLP originales del dataset
- WHEN se ajusta el StandardScaler y se entrena el autoencoder
- THEN ambos (scaler y autoencoder) se ajustan SOLO sobre filas del split train
- AND val/test son transformados con el scaler/encoder ya ajustado, nunca re-ajustado

---

## REQ-2: Arquitectura del Autoencoder

El sistema SHALL implementar un autoencoder denso simétrico con cuello de botella de 3 dimensiones.

**Arquitectura obligatoria:** `10 → 5 (relu) → 3 (relu) → 5 (relu) → 10 (linear)`
**Regularización:** Dropout(0.2) en capas encoder, EarlyStopping(patience=10, restore_best_weights=True)
**Loss:** MSE sobre la reconstrucción de las 10 variables NLP normalizadas
**Optimizer:** Adam(learning_rate=0.001)

#### Scenario: Convergencia del autoencoder
- GIVEN arquitectura definida y datos de entrenamiento NLP normalizados
- WHEN se entrena con EarlyStopping
- THEN val_loss decrece y se estabiliza (no diverge)
- AND se muestra curva train_loss vs val_loss por época

---

## REQ-3: Latentes para Todos los Splits

El sistema SHALL extraer la representación latente (3 dimensiones) para train, val y test usando el encoder ya entrenado.

#### Scenario: Extracción sin re-entrenamiento
- GIVEN encoder entrenado en REQ-1
- WHEN se aplica sobre train, val y test
- THEN se obtienen matrices de shape `(n_dias, 3)` para cada split
- AND las 3 columnas se nombran `nlp_ae_0`, `nlp_ae_1`, `nlp_ae_2`

---

## REQ-4: XGBoost con Latentes del Autoencoder

El sistema SHALL reentrenar el modelo XGBoost Grupo D sustituyendo las 4 variables NLP originales por las 3 dimensiones latentes del autoencoder.

#### Scenario: Comparación controlada
- GIVEN mismo XGBoost (hiperparámetros idénticos a NB05 Grupo D) y mismo conjunto de features macro/AR
- WHEN se reemplaza `[nlp_event_count_roll_mean30, nlp_total_articles_roll_mean30, nlp_events_gov_roll_mean30, nlp_events_biz_roll_mean30]` por `[nlp_ae_0, nlp_ae_1, nlp_ae_2]`
- THEN se reporta RMSE test y se compara contra Grupo D original (57.48 pb)
- AND el número total de features es `(60 - 4 + 3) = 59`

---

## REQ-5: Baseline PCA-3

El sistema SHALL incluir una comparativa con PCA de 3 componentes sobre las mismas 10 variables NLP como baseline lineal.

#### Scenario: PCA vs Autoencoder
- GIVEN PCA(n_components=3) ajustado solo en train
- WHEN se aplica XGBoost Grupo D con las 3 componentes PCA en lugar de las 4 originales
- THEN se reporta RMSE test de PCA-3
- AND se construye tabla final: `Original-4 | PCA-3 | Autoencoder-3` con RMSE, MAE, R²

---

## REQ-6: Interpretación de Dimensiones Latentes

El sistema SHALL producir un heatmap de correlación entre las 3 dimensiones latentes del autoencoder y las 10 variables NLP originales.

#### Scenario: Heatmap de correlación
- GIVEN latentes extraídas para el set completo (train+val+test)
- WHEN se calcula correlación de Pearson entre dims latentes y variables NLP originales
- THEN se visualiza como heatmap 3×10
- AND se nombran las dims por su variable de mayor correlación absoluta

---

## Constraints

- MUST NOT ajustar scaler ni encoder sobre datos de validación o test
- MUST NOT modificar notebooks NB01–NB06 ni los `.pkl` existentes
- MUST usar los mismos hiperparámetros XGBoost que en NB05 para la comparativa controlada
- SHOULD guardar el encoder entrenado como `notebooks/nlp_autoencoder.keras` o `.h5`
- MAY guardar figuras como `.png` en `notebooks/`
