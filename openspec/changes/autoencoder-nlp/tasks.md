# Tasks: Autoencoder NLP

## NB08 — `notebooks/08_Autoencoder_NLP.ipynb`

### 1. Setup
- [ ] 1.1 Verificar TensorFlow instalado; mostrar versión o instrucción de instalación
- [ ] 1.2 Imports (tensorflow/keras, sklearn, xgboost, pandas, numpy, matplotlib, seaborn, pickle)
- [ ] 1.3 Cargar `dataset_tesis_final.pkl`, splits `.pkl`, `feature_selector_metadata.pkl`
- [ ] 1.4 Definir `NLP_COLS_ORIG` (10 columnas) y `NLP_ORIG_4` (4 usadas en NB05)

### 2. Preparación de Datos NLP
- [ ] 2.1 Extraer 10 variables NLP de `dataset_tesis_final.pkl` alineadas con índices de cada split
- [ ] 2.2 Aplicar ffill para NaN residuales
- [ ] 2.3 Ajustar `StandardScaler` solo sobre train; transformar val y test

### 3. Autoencoder
- [ ] 3.1 Definir arquitectura con Functional API: `10→5(relu)→3(relu)→5(relu)→10(linear)` + Dropout(0.2)
- [ ] 3.2 Crear objeto `encoder` separado (Input → capa latente)
- [ ] 3.3 Compilar con `Adam(0.001)`, `loss='mse'`
- [ ] 3.4 Entrenar con EarlyStopping(patience=10), epochs=200, batch_size=32
- [ ] 3.5 Plotear curva train_loss vs val_loss por época → `autoencoder_loss_curve.png`
- [ ] 3.6 Guardar encoder en `nlp_autoencoder.keras`

### 4. Extracción de Latentes
- [ ] 4.1 Extraer latentes para train, val, test con `encoder.predict()`
- [ ] 4.2 Crear DataFrames con columnas `nlp_ae_0`, `nlp_ae_1`, `nlp_ae_2` e índices originales
- [ ] 4.3 Reemplazar las 4 vars NLP originales en los splits feature-engineered

### 5. XGBoost con Autoencoder
- [ ] 5.1 Construir splits X con features no-NLP + latentes AE (59 features total)
- [ ] 5.2 Entrenar XGBoost (hiperparámetros NB05 Grupo D)
- [ ] 5.3 Evaluar en test: RMSE, MAE, R²
- [ ] 5.4 Comparar contra Grupo D original (57.48 pb) en tabla

### 6. Baseline PCA-3
- [ ] 6.1 Ajustar `PCA(n_components=3)` solo en train NLP
- [ ] 6.2 Transformar val y test
- [ ] 6.3 Mismo pipeline XGBoost con PCA latentes
- [ ] 6.4 Evaluar en test y añadir a tabla comparativa

### 7. Heatmap de Correlación
- [ ] 7.1 Extraer latentes AE para el dataset completo (train+val+test)
- [ ] 7.2 Calcular correlación de Pearson: 3 dims latentes × 10 vars NLP originales
- [ ] 7.3 Heatmap anotado → `nlp_latentes_correlacion.png`
- [ ] 7.4 Nombrar dims latentes según variable de mayor correlación absoluta

### 8. Tabla Resumen y Conclusiones
- [ ] 8.1 Tabla final: Original-4 | PCA-3 | AE-3 — RMSE, MAE, R² en test
- [ ] 8.2 Bar chart comparativo de RMSE → `autoencoder_comparativa_rmse.png`
- [ ] 8.3 Celda markdown con conclusión: ¿la no-linealidad aporta sobre PCA? ¿mejora sobre Grupo D?
- [ ] 8.4 Celda markdown con implicaciones para la tesis y trabajo futuro (BERT/FinBERT)
