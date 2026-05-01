# Proposal: Autoencoder NLP — Compresión Neuronal de Variables Mediáticas

## Intent

Investigar si una representación latente no lineal de las 10 variables NLP de GDELT mejora la contribución predictiva del NLP sobre el EMBI, respondiendo la sugerencia del tutor sobre arquitecturas de compresión. El modelo actual usa 4 variables de volumen NLP (rolling 30d) seleccionadas linealmente; un autoencoder podría capturar patrones no lineales entre ellas.

## Scope

### In Scope
- Autoencoder denso (10→5→3→5→10) sobre las 10 variables NLP originales (sin feature selection previa)
- Usar las 3 dimensiones latentes como features NLP en el pipeline XGBoost
- Experimento controlado: comparar RMSE test de Grupo D actual (57.48 pb) vs Grupo D con latentes del autoencoder
- Comparativa adicional: PCA 3 componentes como baseline lineal vs autoencoder (no lineal)
- Análisis de qué captura cada dimensión latente (correlación con variables originales)

### Out of Scope
- LSTM / redes recurrentes end-to-end (trabajo futuro)
- Embeddings de texto crudo BERT/FinBERT sobre noticias (requiere datos de texto, no disponibles en GDELT v2 aggregado)
- Reemplazar XGBoost como modelo principal
- Modificar variables macro o financieras

## Approach

Nuevo `notebooks/08_Autoencoder_NLP.ipynb`:
1. Preparar matriz NLP (10 features × 3047 días) desde `dataset_tesis_final.pkl`
2. Entrenar autoencoder en split train (sin ver val/test) — arquitectura: `Dense(10)→Dense(5,relu)→Dense(3,relu)→Dense(5,relu)→Dense(10,linear)`, loss=MSE
3. Extraer representación latente (encoder) para todos los splits
4. Reentrenar Grupo D (XGBoost) usando latentes autoencoder en lugar de 4 variables NLP originales
5. Comparar: PCA-3 vs Autoencoder-3 vs variables originales (4 seleccionadas) en RMSE test
6. Visualizar correlación entre dims latentes y variables NLP originales

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `notebooks/08_Autoencoder_NLP.ipynb` | New | Notebook de experimento autoencoder |
| `notebooks/dataset_tesis_final.pkl` | Read-only | Datos base sin modificar |
| `train/val/test_feature_engineered.pkl` | Read-only | Para el XGBoost con latentes |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Autoencoder no mejora sobre PCA (espacio NLP demasiado pequeño) | Med-High | Resultado válido: confirma que la compresión lineal es suficiente; reportar en sección de limitaciones |
| Overfitting del autoencoder con 3047 muestras y red pequeña | Low | Usar dropout(0.2) + early stopping; val loss como criterio de parada |
| Latentes difíciles de interpretar | Med | Heatmap de correlación con variables originales; nombrar dims por variable dominante |

## Rollback Plan

Notebook independiente — no modifica ningún artefacto del pipeline existente. Si el autoencoder no mejora, el resultado se reporta como hallazgo negativo válido (confirmación de representación lineal suficiente).

## Dependencies

- `dataset_tesis_final.pkl` (10 columnas NLP originales, no las 4 seleccionadas)
- `train/val/test_feature_engineered.pkl` (para reentrenar XGBoost con latentes)
- TensorFlow/Keras o PyTorch instalado en el entorno (`tesis_env`)

## Success Criteria

- [ ] Autoencoder entrenado con val_loss convergido (curva de pérdida mostrada)
- [ ] Comparación de RMSE test: originales vs PCA-3 vs Autoencoder-3
- [ ] Heatmap de correlación: dimensiones latentes vs variables NLP originales
- [ ] Conclusión clara: ¿la no-linealidad aporta sobre PCA? ¿mejora sobre Grupo D (57.48 pb)?
- [ ] Sección de interpretación: qué representa cada dimensión latente
