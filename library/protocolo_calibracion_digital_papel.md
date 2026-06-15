# Protocolo Reproducible de Calibracion Digital -> Papel para TMT y SDMT

## 1. Objetivo

Construir y validar un procedimiento reproducible que permita transformar las medidas recogidas por las pruebas digitales TMT y SDMT en estimaciones equivalentes a los resultados en papel, que siguen siendo el patron clinico validado. El protocolo debe:

- preservar el anclaje clinico en la medida en papel,
- evitar fuga de datos entre sesiones y pacientes,
- incorporar covariables clinicas relevantes como edad y sexo,
- cuantificar incertidumbre de forma robusta,
- dejar trazabilidad completa de datos, codigo, configuracion y resultados.

## 2. Base bibliografica usada

La carpeta `library/` contiene la base metodologica inicial. A partir de metadatos y contenido accesible localmente, se extraen estos principios operativos:

1. `Digit_Test_cognitivos_21.pdf`
   Referencia detectada: "Digitization of neuropsychological diagnostics: a pilot study to compare three paper-based and digitized cognitive assessments". DOI detectado: `10.1007/s40520-020-01668-z`.
   Uso en el protocolo: justificar que la equivalencia digital-papel debe evaluarse explicitamente, no asumirse.

2. `Validac TMT_Tablet_2021.pdf`
   Metadatos detectados: DOI `10.1111/ejn.15541`, palabras clave con `paper- and tablet-based comparison`, `digital neuropsychology`, `Trail Making Test`.
   Uso en el protocolo: apoyar comparaciones concurrentes papel-tablet y la necesidad de medir validez concurrente y no solo error predictivo.

3. `Reliability TMT_2011.pdf`
   Titulo detectado: `Reliability of Three Alternate Forms of the Trail Making Tests A and B`.
   Uso en el protocolo: incorporar la fiabilidad test-retest y el efecto de formas/variantes como parte del marco de incertidumbre.

4. `Test_retest_TMT_2023.pdf`
   Se detectan secciones `Reliability`, `Test-retest reliability analyses`, `Concurrent validity analyses`, `Practice effects` y `Validity`.
   Uso en el protocolo: incluir fiabilidad, efecto de practica y validez concurrente como criterios de aceptacion.

5. `Developed_digitalTMT_2017.pdf`
   Se detectan secciones `The Design of the Digital Trail Making Test` y `Validation`.
   Uso en el protocolo: justificar que las features digitales deben representar no solo el tiempo total sino el proceso de ejecucion.

6. `Tiempos recomendados entre test.docx`
   Resumen local extraido: para TMT, un intervalo prudente de test-retest es `2 a 4 semanas`, especialmente en digital, para reducir efecto de aprendizaje.
   Uso en el protocolo: fijar ventanas recomendadas para estudios de repetibilidad y calibracion de incertidumbre temporal.

Nota: donde no se ha podido extraer el articulo completo localmente, las decisiones se apoyan en metadatos, secciones y coherencia con el objetivo clinico.

## 3. Principio clinico central

El modelo no debe predecir "mejor que el papel" sino aproximar de forma estable el score en papel. Por tanto:

- la variable objetivo siempre es el score en papel,
- las covariables clinicas se usan para ajustar la traduccion al dominio validado,
- el rendimiento se evaluara por concordancia con papel, no solo por capacidad explicativa interna.

## 4. Unidad de analisis y poblacion

### 4.1 Unidad primaria

La unidad primaria es la sesion de evaluacion digital emparejada con una observacion clinica en papel del mismo sujeto.

### 4.2 Agrupacion

La unidad de independencia estadistica es el `patient_id`.

Consecuencia:

- todos los splits deben respetar paciente completo,
- el bootstrap y las bandas de incertidumbre deben remuestrear por paciente, no por fila,
- cualquier calibracion de intervalos debe separar pacientes entre entrenamiento y calibracion.

### 4.3 Subpoblaciones obligatorias

El protocolo exige analizar al menos estos estratos:

- EM frente a Controles,
- sexo,
- grupos de edad,
- nivel educativo si esta disponible,
- sesiones iniciales frente a retest.

## 5. Definicion de objetivos por prueba

### 5.1 SDMT

Objetivo primario:

- `Día 1 SDMT Papel score`

Objetivos secundarios si hay masa critica:

- errores en papel,
- cambios longitudinales digital->papel en retest.

### 5.2 TMT

Objetivos primarios separados:

- `TMT papel (tiempo) A`
- `TMT papel (tiempo) B`
- `TMT papel (errores) A`
- `TMT papel (errores) B`

No deben colapsarse tiempos y errores en un unico objetivo salvo analisis adicional justificado.

## 6. Variables candidatas

### 6.1 Covariables clinicas minimas

- edad en fecha de test,
- sexo,
- grupo clinico (`EM` o `Controles`),
- escolaridad si existe,
- mano dominante si existe,
- intervalo entre medicion digital y papel si no son simultaneas,
- indicador de orden `papel/digital` si existe,
- indicador de retest.

### 6.2 Features digitales

Se usaran todas las medidas numericas del registro digital que representen:

- tiempo total,
- latencias parciales,
- errores,
- correcciones,
- pausas,
- velocidad,
- trayectoria o espaciotemporalidad derivada,
- variabilidad intra-prueba.

### 6.3 Exclusiones obligatorias

No usar como predictor:

- identificadores,
- timestamps administrativos sin interpretacion clinica,
- campos derivados directamente del target en papel,
- variables post-hoc que solo existan tras conocer el resultado en papel.

## 7. Preparacion reproducible del dataset

Para cada corrida debe generarse un artefacto congelado con:

1. Consulta SQL o extraccion exacta desde PostgreSQL.
2. Hash del dataset resultante.
3. Version del Excel clinico usado.
4. Configuracion YAML utilizada.
5. Fecha y commit git.

### 7.1 Reglas de emparejamiento

- emparejar por `patient_id` estandarizado,
- si hay varias sesiones, enlazar con la evaluacion clinica mas proxima en ventana predefinida,
- registrar siempre el desfase temporal `delta_dias_digital_papel`.

### 7.2 Reglas de calidad

Excluir o marcar:

- pares digital-papel con ventana temporal excesiva,
- targets en papel ausentes,
- sesiones con telemetria incompleta,
- outliers imposibles por fallo tecnico evidente.

Toda exclusion debe quedar en un log tabular.

## 8. Splits y validacion sin fuga

### 8.1 Esquema principal

Usar un esquema `nested grouped CV` por paciente:

- outer loop: `GroupKFold` o `GroupShuffleSplit` repetido por paciente,
- inner loop: seleccion de hiperparametros solo dentro del train del outer loop,
- nunca mezclar sesiones del mismo paciente entre train y validacion.

### 8.2 Repeticion

Para estabilizar estimaciones:

- repetir el outer split al menos `30` veces para `GroupShuffleSplit`, o
- usar `5 folds x 10 repeticiones` si el tamano muestral lo permite.

### 8.3 Retest

Si hay retest:

- mantener todos los registros del mismo paciente en un mismo bloque de split,
- hacer un analisis secundario de estabilidad temporal usando solo pares dentro de `2-4 semanas`.

## 9. Modelado recomendado

### 9.1 Filosofia

No elegir un unico modelo desde el principio. Se recomienda una familia escalonada:

1. Baseline interpretable.
2. Modelo no lineal parsimonioso.
3. Modelo flexible de alto rendimiento.

### 9.2 Baselines obligatorios

- media estratificada por edad/sexo/grupo,
- regresion lineal o ridge,
- elastic net si hay muchas features correlacionadas.

### 9.3 Modelos candidatos principales

- `Ridge` o `ElasticNet` para baseline interpretable,
- `RandomForestRegressor` o `Quantile Random Forest` para no linealidad robusta,
- `XGBoostRegressor` para challenger principal,
- multi-output solo para TMT si se demuestra que mejora sin perjudicar interpretabilidad.

### 9.4 Transformaciones de target

Para tiempos TMT considerar:

- modelar en escala original y logaritmica,
- comparar ambos y elegir la que mejore residuos y concordancia.

Para errores TMT considerar:

- si la distribucion es muy discreta o inflada en ceros, evaluar modelo ordinal o de conteo como analisis complementario.

## 10. Seleccion del modelo final

El modelo final no se elegira solo por RMSE. Debe optimizar un criterio clinico compuesto:

- error absoluto medio,
- sesgo medio,
- concordancia con papel,
- estabilidad por subgrupo,
- cobertura de incertidumbre,
- simplicidad operacional.

### 10.1 Metricas primarias

- RMSE
- MAE
- mediana del error absoluto
- sesgo medio (`y_true - y_pred`)
- R2

### 10.2 Metricas de concordancia obligatorias

- ICC de acuerdo absoluto,
- CCC de Lin,
- analisis de Bland-Altman con limites de acuerdo,
- pendiente e intercepto de calibracion (`y_true ~ y_pred`).

### 10.3 Metricas por subgrupo

Calcular todas las primarias por:

- EM vs Controles,
- sexo,
- terciles o cuartiles de edad,
- escolaridad si esta disponible.

## 11. Estrategia robusta de incertidumbre

La incertidumbre debe medirse en dos niveles.

### 11.1 Incertidumbre de metrica poblacional

Usar `bootstrap agrupado por paciente`:

- remuestrear pacientes completos con reemplazo,
- recalcular el pipeline completo o al menos la evaluacion out-of-fold,
- reportar IC95% para RMSE, MAE, ICC, CCC y sesgo.

Minimo recomendado:

- `1000` replicas bootstrap por objetivo.

### 11.2 Incertidumbre de prediccion individual

Usar `split conformal prediction` agrupado por paciente:

1. train de modelo en conjunto de entrenamiento,
2. calibration set separado por paciente,
3. calcular residuos absolutos en calibration,
4. construir intervalos predictivos `pred +/- q_(1-alpha)`.

Si el modelo lo permite, mejorar con:

- `Conformalized Quantile Regression` para intervalos heterocedasticos,
- normalizacion por escala local del error.

### 11.3 Incertidumbre estructural del modelo

Comparar al menos tres familias de modelos. Si dos modelos tienen rendimiento parecido pero distinta estructura, reportar:

- dispersion entre modelos,
- ranking medio en CV repetida,
- diferencia de metricas con IC bootstrap.

### 11.4 Incertidumbre por practica temporal

En sujetos con retest dentro de `2-4 semanas`:

- estimar cambio medio digital,
- estimar cambio medio en papel,
- cuantificar cuanto del error puede explicarse por variacion test-retest y practica.

## 12. Criterios de aceptacion del modelo

Un modelo se considerara candidato final si cumple simultaneamente:

1. mejora al baseline demografico.
2. mantiene sesgo medio cercano a cero.
3. conserva concordancia razonable con papel en ICC/CCC.
4. no muestra degradacion severa en subgrupos clinicos.
5. los intervalos predictivos alcanzan cobertura nominal aproximada.
6. el error residual es del mismo orden o menor que la variabilidad test-retest aceptable de la literatura y del subestudio local.

## 13. Entregables por corrida

Cada experimento debe producir:

- dataset congelado o referencia hash,
- YAML de configuracion,
- tabla de exclusions,
- tabla de resultados globales por modelo y target,
- tabla de resultados por subgrupos,
- Bland-Altman por target,
- scatter `papel vs prediccion`,
- distribucion de residuos,
- intervalos predictivos y cobertura observada,
- importancias o interpretabilidad del modelo,
- informe resumen reproducible en Markdown o HTML.

## 14. Flujo operativo minimo en este repositorio

### Fase A. Consolidacion de datos

- extraer tablas `sdmt` y `tmt` desde PostgreSQL,
- fusionar con `calibration/data/datos_papel.xlsx`,
- conservar marca de hoja `EM`/`Controles`,
- derivar `age_at_test`, `sex_binary` y `delta_dias_digital_papel`.

### Fase B. Curacion

- revisar cobertura por target,
- revisar nulos por feature,
- documentar reglas de imputacion,
- decidir ventana maxima de emparejamiento.

### Fase C. Modelado

- baseline lineal,
- random forest,
- xgboost,
- opcion cuantile/conformal para incertidumbre.

### Fase D. Validacion

- nested grouped CV repetida,
- bootstrap agrupado por paciente,
- analisis de subgrupos,
- analisis de retest.

### Fase E. Seleccion y congelacion

- elegir un modelo por target,
- reentrenar con todo el train designado,
- fijar artefacto serializado,
- fijar calibrador conformal y metadata de cobertura.

## 15. Decisiones tecnicas concretas recomendadas

### 15.1 Para SDMT

Modelo inicial recomendado:

- `Ridge` como baseline,
- `XGBoostRegressor` como challenger,
- intervalos via split conformal.

### 15.2 Para TMT tiempos

Modelo inicial recomendado:

- comparar target original frente a `log(tiempo)`,
- `Ridge` y `XGBoostRegressor`,
- si residuos heterocedasticos: cuantiles + conformal.

### 15.3 Para TMT errores

Si la masa en cero es alta:

- baseline de clasificacion `error=0` / `error>0`,
- despues regresion o conteo condicionada.

## 16. Discusion de escenarios SDMT

Con los datos actualmente disponibles, el SDMT debe interpretarse principalmente como un problema de calibracion digital -> papel, pero no como una traduccion trivial del score digital bruto.

Para comprobarlo, se compararon tres escenarios:

1. `calibration_full`
   Incluye todas las variables digitales disponibles, incluidas las mas proximas semanticamente al resultado en papel.
2. `without_outcome_like`
   Excluye variables conceptualmente demasiado cercanas al score objetivo: `num_err`, `num_simbolos`, `score`, `numdig1`, `numerr1`, `numdig2`, `numerr2`, `numdig3`, `numerr3`.
3. `kinematic_clinical_only`
   Mantiene solo variables cinematicas y covariables clinicas.

En la matriz SDMT actual, los escenarios 2 y 3 son equivalentes tras el filtrado, porque al retirar las variables outcome-like solo permanecen:

- `diagonal_inches`
- `avgdur`
- `sdvdur`
- `disease_duration_years`
- `age_at_test`
- `delta_dias_digital_papel`
- `sex_binary`

Los resultados comparativos muestran:

| Escenario | Mejor modelo | RMSE | MAE | R2 | CCC | ICC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `calibration_full` | `linear` | 6.44 | 4.99 | 0.641 | 0.812 | 0.823 |
| `without_outcome_like` | `rf` | 6.85 | 5.99 | 0.593 | 0.756 | 0.768 |
| `kinematic_clinical_only` | `rf` | 6.85 | 5.99 | 0.593 | 0.756 | 0.768 |

Estos resultados sostienen cuatro ideas:

- El mejor comportamiento predictivo se obtiene cuando el sistema usa tambien variables proximas al rendimiento digital directo, por lo que la solucion operativa mas precisa hoy es un calibrador digital -> papel.
- La eliminacion de variables outcome-like empeora el rendimiento, lo que confirma que estas variables contienen una parte relevante de la señal predictiva.
- Sin embargo, la degradacion no es catastrofica: el modelo reducido mantiene un rendimiento moderado, lo que indica que las variables cinematicas y clinicas capturan informacion real sobre el rendimiento en papel.
- Por tanto, el SDMT actual no debe presentarse como un biomarcador puramente mecanistico e independiente del score digital, sino como un sistema de calibracion apoyado por biomarcadores de ejecucion y covariables clinicas.

Implicacion practica:

- si el objetivo es maxima precision clinica, debe priorizarse el escenario `calibration_full`;
- si el objetivo es interpretar mecanismos o construir modelos menos dependientes del score digital bruto, deben analizarse las ramas `without_outcome_like` o `kinematic_clinical_only`, aceptando una perdida moderada de rendimiento.

## 17. Riesgos conocidos

- muestras pequenas en algunos targets,
- desbalance EM/Controles,
- efecto de practica si hay retests cortos,
- diferencias de hardware o usabilidad que inflen error digital,
- riesgo de sobreajuste si el numero de features supera mucho el numero de pacientes.

## 18. Plan de implementacion inmediato

1. Crear un artefacto reproducible de extraccion y snapshot de datos.
2. Añadir `delta_dias_digital_papel`, `clinical_group` y escolaridad si existe.
3. Extender reporting con ICC, CCC y Bland-Altman.
4. Añadir CV repetida agrupada y bootstrap por paciente.
5. Añadir calibracion conformal para intervalos predictivos.
6. Generar informe comparativo SDMT/TMT por target y por subgrupo.

## 19. Resultado esperado

El resultado final no sera solo un modelo, sino un sistema reproducible de traduccion digital->papel, con:

- prediccion puntual,
- intervalo de prediccion por individuo,
- bandas de incertidumbre para metricas poblacionales,
- evidencia de estabilidad por paciente y por subgrupo,
- trazabilidad suficiente para defensa cientifica y regulatoria.
