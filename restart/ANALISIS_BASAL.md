# Especificación de la primera ejecución basal

Esta implementación ejecuta una primera fase exploratoria del protocolo, no una validación clínica final. Origen, hash de código, semillas y versiones se registran en `reports/baseline/results.json`. No se exportan identificadores ni predicciones individuales.

- Orden: SDMT papel basal, después TMT-A y TMT-B papel basal. Solo una fila basal por código único. Sin retest, sin variables específicas de enfermedad como predictores de grupo.
- Ajuste inferencial: edad continua, sexo y escolarización categóricos; efecto de grupo OLS con HC3 y referencia t. Casos completos, sin imputar desenlaces. Modelos separados con las mismas covariables; interacción grupo×edad secundaria. Sensibilidad restringida al rango de edades compartido (no garantiza soporte multivariante).
- Regresión predictiva separada por grupo: media, lineal, ridge demográfico y ridge demográfico+TUG. Cinco particiones externas por persona y tres internas para seleccionar alpha entre 0,1/1/10/100 por MAE. Imputación mediana y escalado en cada entrenamiento. Las categorías están prefijadas por el diccionario, no aprendidas de los desenlaces. La elección de TUG representa un único dominio motor interpretable; 6MWT está vacío y otras medidas motoras requieren depuración. Comparar alternativas no convierte a la mejor en un modelo validado tras selección.
- Discriminación: regresión logística demográfica frente a demográfica+resultado cognitivo, C=1 fijo y sin ponderar clases. Umbral 0,5 prefijado, sin optimización sobre evaluación. Las probabilidades reflejan el muestreo de esta cohorte, no prevalencia de aplicación. Brier es una métrica agregada, no una evaluación completa de calibración.
- Norma sana independiente: ridge demográfico alpha=10 fijo, ajustado solo con controles de cada entrenamiento externo. AUC de la desviación predicción sana−observado; no se entrena un clasificador con residuos in-sample. La logística ampliada usa directamente el desenlace, no esos residuos. Es un análisis complementario de incremento informativo, no una validación del clasificador normativo propuesto en el protocolo.
- Clasificación con cinco folds estratificados y agrupados por persona, semilla 20260930. Dos semillas adicionales evalúan sensibilidad a particiones. IC percentiles con 200 remuestreos estratificados de personas y reajuste de todo el procedimiento; copias del mismo participante permanecen juntas. Estos IC son exploratorios y los extremos requieren más réplicas para una publicación.
- TMT: y=−log(segundos), para orientar todos los desenlaces a mayor=mejor. Efecto de grupo comunicado como exp(−coeficiente), razón de tiempos pacientes/controles; Holm sobre los dos efectos principales TMT. MAE/RMSE de TMT están en escala logarítmica, no segundos. No hay predicción retransfomada de tiempo medio.
- Potencia SDMT: 2.000 simulaciones por escenario de diferencia 0/3/5/8/10 puntos, desviación típica controles 8/12/16 y razón de desviaciones pacientes/controles 1/1,5. Diseño de covariables fijo de casos completos y errores normales, test HC3 bilateral con referencia t, α=0,05. Cero evalúa error tipo I. Son supuestos de planificación, no potencia observada. No simula faltantes informativos, no estudia interacciones ni calcula aún ampliaciones muestrales.

Las fases de perfiles individuales, revisión completa de predictores, estabilidad de selección, análisis digital/retest, ampliación muestral y validación externa permanecen pendientes. No inferir ausencia de diferencias de pendientes a partir de una interacción no significativa.

## Ejecución

```bash
python -m pip install -r restart/requirements-modeling.txt
OPENBLAS_NUM_THREADS=1 python -m restart.model_baseline '/ruta/libro.xls' --bootstrap 200
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest restart/test_model_baseline.py -q
```

Las pruebas usan datos sintéticos; comprueban imputación limitada a entrenamiento, separación de copias de participantes y reproducibilidad. Desactivar plugins externos evita un fallo del plugin global Dash ajeno a este repositorio.

Referencias: [prevención de fuga de información](https://scikit-learn.org/1.8/common_pitfalls.html) y [planificación del tamaño muestral](https://www.bmj.com/content/368/bmj.m441).
