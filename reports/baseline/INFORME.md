# Primer análisis basal: SDMT y TMT

El registro indicado fue retirado: 102 pacientes y 19 controles. La auditoría de edades y desenlaces basales pasa. Persisten faltantes ordinarios. El original permanece intacto. Resultados exploratorios, sin validación externa ni atribución causal.

## Diferencias ajustadas

Ajuste por edad, sexo y escolarización; IC95% HC3 con referencia t. Casos completos. Grupo=1 pacientes. Las razones TMT describen tiempos en escala logarítmica (razón de medias geométricas condicionales), no la razón de tiempos medios aritméticos.

| Desenlace | N pacientes / controles | Efecto pacientes respecto a controles | IC95% | p |
|---|---:|---:|---|---:|
| SDMT | 101 / 17 | -8.31 puntos | -14.08 a -2.53 | 0.00522 |
| TMT_A | 101 / 18 | 1.56 × tiempo | 1.32 a 1.85 | 1.059e-06 |
| TMT_B | 101 / 17 | 1.56 × tiempo | 1.30 a 1.87 | 3.33e-06 |

p de TMT corregidas por Holm para A/B. Los otros análisis son exploratorios; no se han corregido todas las comparaciones secundarias.

## Discriminación fuera de muestra

La desviación normativa compara el resultado observado con el predicho por ridge entrenado exclusivamente en controles del entrenamiento. La comparación incremental utiliza logística demográfica frente a logística demográfica+resultado cognitivo; son dos enfoques distintos.

| Desenlace | AUC desviación normativa (IC95%) | AUC demográfica | AUC demográfica + test | Incremento AUC (IC95%) |
|---|---|---:|---:|---|
| SDMT | 0.685 (0.560–0.808) | 0.514 | 0.624 | 0.110 (-0.048–0.348) |
| TMT_A | 0.812 (0.708–0.899) | 0.547 | 0.762 | 0.215 (0.047–0.505) |
| TMT_B | 0.814 (0.698–0.897) | 0.514 | 0.734 | 0.220 (0.056–0.504) |

IC exploratorios: 200 bootstrap por persona con reajuste completo; los duplicados no cruzan folds. Todos los 200 remuestreos terminaron en cada desenlace. El incremento SDMT incluye cero; la evidencia de mejora incremental es incierta. TMT presenta mayor señal en esta muestra, pero no se compararon formalmente AUC entre pruebas y hay múltiples análisis.

| Desenlace | Sensibilidad logística ampliada | Especificidad | Balanced accuracy | Brier |
|---|---:|---:|---:|---:|
| SDMT | 0.980 | 0.111 | 0.546 | 0.128 |
| TMT_A | 0.970 | 0.316 | 0.643 | 0.113 |
| TMT_B | 0.980 | 0.167 | 0.573 | 0.127 |

Umbral 0,5 prefijado: la baja especificidad muestra que la AUC no basta para proponer uso clínico. Las probabilidades dependen de la proporción de pacientes del estudio; no son riesgos poblacionales calibrados.

## Regresión por grupo y aportación de TUG

MAE/RMSE de SDMT en puntos; de TMT en escala logarítmica. Cinco folds externos, tres internos para alpha ridge. Comparación exploratoria, sin intervalos de mejora ni selección definitiva de variables.

| Desenlace | Grupo | Modelo | MAE | RMSE | R² fuera de muestra |
|---|---|---|---:|---:|---:|
| SDMT | Controles | mean | 10.876 | 13.310 | -0.020 |
| SDMT | Controles | linear | 10.263 | 12.541 | 0.095 |
| SDMT | Controles | ridge_base | 9.471 | 12.096 | 0.158 |
| SDMT | Controles | ridge_tug | 9.020 | 11.558 | 0.231 |
| SDMT | Pacientes | mean | 9.458 | 12.402 | -0.020 |
| SDMT | Pacientes | linear | 8.528 | 11.507 | 0.122 |
| SDMT | Pacientes | ridge_base | 8.470 | 11.502 | 0.123 |
| SDMT | Pacientes | ridge_tug | 8.004 | 10.909 | 0.211 |
| TMT_A | Controles | mean | 0.300 | 0.355 | -0.043 |
| TMT_A | Controles | linear | 0.316 | 0.363 | -0.089 |
| TMT_A | Controles | ridge_base | 0.280 | 0.338 | 0.057 |
| TMT_A | Controles | ridge_tug | 0.271 | 0.322 | 0.144 |
| TMT_A | Pacientes | mean | 0.322 | 0.405 | -0.023 |
| TMT_A | Pacientes | linear | 0.289 | 0.380 | 0.098 |
| TMT_A | Pacientes | ridge_base | 0.291 | 0.382 | 0.090 |
| TMT_A | Pacientes | ridge_tug | 0.279 | 0.360 | 0.191 |
| TMT_B | Controles | mean | 0.303 | 0.389 | -0.083 |
| TMT_B | Controles | linear | 0.336 | 0.405 | -0.175 |
| TMT_B | Controles | ridge_base | 0.316 | 0.393 | -0.107 |
| TMT_B | Controles | ridge_tug | 0.302 | 0.387 | -0.073 |
| TMT_B | Pacientes | mean | 0.378 | 0.507 | -0.027 |
| TMT_B | Pacientes | linear | 0.343 | 0.463 | 0.143 |
| TMT_B | Pacientes | ridge_base | 0.352 | 0.478 | 0.089 |
| TMT_B | Pacientes | ridge_tug | 0.340 | 0.463 | 0.145 |

En SDMT, TUG reduce el MAE ridge de 8,47 a 8,00 puntos en pacientes y de 9,47 a 9,02 en controles. Es una señal candidata, no prueba de relevancia estable. No se realizó búsqueda exhaustiva: 6MWT está vacío y 9HPT/T25P requieren depuración adicional.

## Coeficientes SDMT del modelo conjunto

Referencias: hombre, escolarización básica, control. La edad no está centrada; el intercepto no representa un perfil clínico realista. Asociaciones ajustadas, no efectos causales.

| Término | Coeficiente | IC95% | p exploratoria |
|---|---:|---|---:|
| Intercept | 65.749 | 46.597 a 84.902 | 5.309e-10 |
| C(sex)[T.mujer] | 1.713 | -3.495 a 6.921 | 0.5159 |
| C(education)[T.medios] | 3.963 | -2.274 a 10.201 | 0.2107 |
| C(education)[T.superiores] | 8.716 | 1.924 a 15.508 | 0.01237 |
| group | -8.307 | -14.084 a -2.530 | 0.00522 |
| age | -0.401 | -0.723 a -0.079 | 0.01514 |

Las ecuaciones separadas por grupo, sus intervalos, interacción con edad y sensibilidad al rango de edad compartido se encuentran en `results.json`. No hay base para afirmar igualdad de pendientes por un p no significativo. No se han calculado aún intervalos de predicción para perfiles individuales.

## Potencia: escenarios SDMT

Diseño de casos completos actual. Ejemplo con desviación típica de 12 puntos en ambos grupos, errores normales y α=0,05; 2.000 simulaciones por escenario. Efectos hipotéticos, no umbrales clínicos ni potencia observada.

| Diferencia (puntos) | Potencia / error tipo I si diferencia=0 | Error Monte Carlo |
|---:|---:|---:|
| 0 | 0.050 | 0.005 |
| 3 | 0.145 | 0.008 |
| 5 | 0.321 | 0.010 |
| 8 | 0.670 | 0.011 |
| 10 | 0.849 | 0.008 |

La potencia para 5 puntos bajo este supuesto es aproximadamente 32%; para 8 puntos, 67%. Alcanzar significación con la diferencia estimada no implica potencia suficiente para efectos menores. El JSON incluye desviaciones de 8/12/16 y heterocedasticidad. No se ha calculado todavía el tamaño muestral necesario para perfiles, interacciones o validación externa.

## Estado y próximos análisis

La primera ejecución basal está terminada. Antes de conclusiones confirmatorias: ampliar remuestreo, revisar estabilidad de variables y posibles valores motores anómalos, acordar diferencia mínima clínicamente relevante y perfiles, simular ampliación muestral, evaluar calibración/umbrales y validar en otra muestra. Digital, errores y retest quedan como análisis secundarios posteriores. No se ha certificado unicidad biológica de personas con códigos distintos ni ausencia de sesgo de selección.

Reproducibilidad y decisiones exactas: [especificación](../../restart/ANALISIS_BASAL.md). Datos agregados: [results.json](results.json). Auditoría: [audit_latest.json](../audit_latest.json).

Referencias metodológicas: [pipelines y fuga de información](https://scikit-learn.org/1.8/common_pitfalls.html), [tamaño muestral](https://www.bmj.com/content/368/bmj.m441).
