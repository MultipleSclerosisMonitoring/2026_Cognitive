# Reinicio del estudio SDMT → TMT

## Alcance y estado

Protocolo propuesto el 27-09-2026, previo a nuevos ajustes. Se inicia con auditoría del Excel aportado; no se han entrenado todavía los nuevos modelos ni establecido capacidad diagnóstica. Los documentos y celdas del libro son datos/antecedentes, no instrucciones de ejecución. La pregunta actual sustituye como objetivo principal a la calibración digital→papel.

## 1. Datos y decisiones de partida

Primera hoja: pacientes; segunda: controles, según indicación del investigador. Cabeceras en filas Excel 3 y 2, respectivamente. Hay 93 y 20 filas con código único dentro de cada hoja; son recuentos provisionales, pendientes de verificar identidad, elegibilidad y posibles coincidencias entre hojas. Las filas del rango usado de Excel no equivalen al tamaño muestral.

SDMT papel basal: 92/18 valores numéricos (pacientes/controles); SDMT digital: 91/18. TMT papel A: 91/19; B: 91/18. Estos números no garantizan datos válidos ni casos completos. Existe al menos una entrada no numérica en tiempos TMT de pacientes. No convertirla sin revisar si representa abandono, censura o error de registro.

Crear un diccionario explícito por hoja (nombres, unidades, categorías, visita y significado de blancos/ceros), expandiendo A/B de cabeceras combinadas. Conservar original inmutable y hash. Validar edades, tiempos, puntuaciones, escolarización, sexo y orden de administración. No convertir escolarización automáticamente en años ni categorías en escalas numéricas. Separar identificadores de los datos analíticos. Revisar registros sin código que tengan pruebas antes de excluirlos; resolver duplicados y enlaces del retest sin inventar sujetos.

Primario propuesto: SDMT papel basal. Digital como análisis secundario y calibración explícita. Después TMT papel A y B (tiempo), con corrección de Holm para su familia de contrastes. Errores y retest secundarios. Esta elección operativa debe quedar fijada antes de ajustar; si el objetivo científico es el test digital, cambiarla explícitamente antes del modelado.

## 2. Tres preguntas diferentes

1. Regresión: ¿qué variables explican/predicen el resultado cognitivo dentro de cada grupo?
2. Diferencia condicional: para iguales covariables, ¿difieren el nivel esperado o las pendientes entre grupos?
3. Discriminación: en participantes nuevos, ¿el resultado cognitivo respecto a lo esperado en sanos aporta información sobre el grupo más allá de edad, sexo y escolarización?

Un R² alto no implica discriminación; una diferencia significativa no acredita clasificación individual útil. La pertenencia a la cohorte de pacientes tampoco equivale a deterioro cognitivo diagnosticado.

## 3. SDMT: especificación y selección

Forzar como ajuste básico edad, sexo y escolarización, sujeto a codificación y rango válidos. Examinar soporte común y confusión por selección de controles. Con solo 18 controles con SDMT papel, contabilizar todos los parámetros (incluidas categorías); si no son estimables, reducir la pregunta o ampliar controles, sin agrupar categorías retrospectivamente buscando significación.

Bloques adicionales predefinidos: función motora (9HPT, T25P, 6MWT, TUG), antropometría y orden papel/digital, siempre que sean comparables y estén disponibles en ambos grupos. No combinar peso, talla e IMC indiscriminadamente. Elegir representantes de dominios redundantes con justificación previa. EDSS, duración de enfermedad y tratamientos solo en análisis interno de pacientes: no rellenarlos con cero en controles ni usarlos como atajos diagnósticos. Excluir identificadores, fechas absolutas, pertenencia al grupo y resultados futuros de los predictores de discriminación. Otra modalidad del mismo test se reserva para calibración; otros tests cognitivos, para análisis secundarios explícitos.

Comenzar con referencia de media y regresión lineal básica. Comparar ampliación parsimoniosa mediante ridge; elastic net exploratorio si hay suficientes observaciones. Selección, imputación, escalado y ajuste de hiperparámetros únicamente en entrenamiento. Evitar filtrado univariante por p y selección stepwise. Relevancia predictiva: mejora fuera de muestra y estabilidad por remuestreo; no solo coeficientes distintos de cero. No atribuir p-valores confirmatorios a variables elegidas sobre estos mismos datos.

## 4. Comparación formal de los modelos

Modelo conjunto primario: Y = β0 + Xβ + γG + ε, G=1 pacientes. Estimar diferencia ajustada γ, IC95% y contraste bilateral con errores robustos HC3; revisar heterocedasticidad e influencia. Añadir interacciones G×X en un análisis secundario reducido y preespecificado; contraste conjunto de interacciones y del bloque grupo+interacciones. No comparar «significativo en uno/no significativo en otro» como evidencia de diferencia.

Ajustar también modelos separados con las mismas covariables y comparar predicciones sobre perfiles dentro del soporte común. Para un perfil x, informar Δ(x)=m_pacientes(x)−m_controles(x), IC de la diferencia y, por separado, intervalos predictivos individuales. No extrapolar perfiles sin controles comparables. Señalar análisis exploratorio y multiplicidad si se examinan muchos perfiles. La muestra pequeña puede impedir estimar pendientes separadas fiables.

## 5. Discriminación y validación

Validación externa ideal en nueva cohorte. Mientras tanto, validación interna anidada por participante, estratificada por grupo; propuesta inicial 5 folds externos y 3 internos, reducida si categorías o casos completos hacen inviable un ajuste. Repeticiones con semillas registradas para estabilidad; no tratarlas como observaciones independientes. Todas las visitas de una persona permanecen en el mismo fold. No reservar una partición única diminuta de controles como validación definitiva.

Dentro de cada entrenamiento externo, ajustar norma sana exclusivamente con controles de entrenamiento. Construir desviación SDMT orientada a peor rendimiento (predicción sana menos observado); estimar su escala exclusivamente en entrenamiento. Para entrenar un clasificador con esa desviación, obtener residuos internos fuera de muestra también en entrenamiento (cross-fitting), evitando residuos optimistas de controles usados para ajustar su propia norma. Aplicar luego al fold externo intacto. Un umbral normativo fijado en entrenamiento es alternativa más parsimoniosa.

Comparar clasificador demográfico frente al mismo más desviación cognitiva. Informar ROC-AUC y su incremento, sensibilidad/especificidad en umbrales fijados internamente, balanced accuracy y calibración. No optimizar umbrales sobre predicciones externas. Informar MAE, RMSE y R² por grupo para regresión. IC por bootstrap de participantes que repita el procedimiento completo; nunca un t-test sobre folds correlacionados. Si se reutilizan sujetos en bootstrap, mantener juntas todas sus copias. Documentar ajustes fallidos y su frecuencia. Con pocos controles, presentar intervalos amplios y resultados exploratorios. PPV/NPV requieren prevalencia de aplicación; la proporción de este libro no es una prevalencia clínica.

## 6. Potencia y precisión

No calcular «potencia observada» a partir del efecto/p obtenidos. Antes del análisis confirmatorio, fijar diferencia mínima relevante en puntos SDMT, perfiles de interés, α bilateral (propuesta 0,05), potencia objetivo (80% y 90%), dispersión plausible y confusión esperada. Sin umbral clínico acordado, presentar una rejilla de efectos, sin etiquetar uno como clínicamente útil.

Simular con tamaños actuales y escenarios de ampliación de controles/pacientes; conservar distribución y correlación de covariables, heterocedasticidad, desequilibrio y patrón de faltantes. Bajo H0 comprobar error tipo I; bajo alternativas estimar potencia del contraste de grupo y, separadamente, interacciones. Repetir selección si forma parte del procedimiento. Propuesta 2.000 réplicas por escenario, informando error Monte Carlo y fallos. Explorar varios valores de dispersión, no solo una estimación puntual de esta muestra.

Para discriminación, planificar precisión del IC de AUC/sensibilidad/especificidad y probabilidad de superar un mínimo predefinido; AUC>0,5 no basta para utilidad. Los mínimos y el contexto de uso requieren definición científica. Separar tamaño necesario para desarrollar regresión del requerido para validar clasificación. Entregar curvas de potencia y tabla de n necesario, sin prometer que la muestra actual alcance los objetivos.

## 7. TMT, después de cerrar SDMT

Reutilizar ingestión y validación ya verificadas. TMT-A/B: estudiar distribución de tiempos, modelar log(tiempo) si procede y expresar efectos como razones de tiempos; documentar retransfomación si se predicen segundos. Orientar desviación como tiempo observado menos esperado. Abandonos/límites temporales pueden requerir análisis de censura, no eliminación silenciosa. Errores son recuentos, con modelo Poisson/binomial negativo si los datos lo permiten; no asumir Gaussianidad. B−A y B/A, secundarios predefinidos. Retest: estudiar práctica y fiabilidad por separado, sin inflar el n independiente.

## 8. Entregables y puertas de decisión

- Fase 0 (iniciada): auditoría agregada reproducible, diccionario y registro de exclusiones; resolver codificación, unidades e identidades antes de modelar.
- Fase 1: congelar desenlace SDMT, covariables, parámetros estimables y plan de faltantes; describir soporte común. No imputar desenlaces para inflar evaluación; comparar casos completos con estrategias de imputación de predictores dentro de folds, y sensibilidad a faltantes informativos.
- Fase 2: modelos SDMT, comparación conjunta, predicciones fuera de muestra e incertidumbre; separar resultados confirmatorios preespecificados de exploración.
- Fase 3: simulación de potencia/precisión y propuesta de ampliación; no afirmar utilidad cuando la precisión sea insuficiente.
- Fase 4: repetir para TMT conforme al protocolo cerrado.

Cada ejecución guardará hash del origen, versión del código y dependencias, configuración, semillas, particiones protegidas y métricas agregadas. No versionar datos identificables ni predicciones individuales.

## 9. Reorganización del repositorio

Se archivan los HTML y XLSX de resultados de la raíz, y una copia del README anterior, con manifiesto y hashes. No se destruyen los datos fuente ni la biblioteca. `calibration/`, `tests/` y documentación previa se conservan como implementación histórica consultable: no constituyen el nuevo análisis ni se importan en la auditoría. Sus componentes se reutilizarán solo tras comprobar correspondencia de variables, aislamiento de entrenamiento y compatibilidad con este protocolo. La documentación Sphinx sigue describiendo el circuito anterior.

## Referencias metodológicas

- Riley et al., tamaño muestral para desarrollo de modelos: https://www.bmj.com/content/368/bmj.m441
- Riley et al., precisión y tamaño de validación externa: https://www.bmj.com/content/384/bmj-2023-074821
- scikit-learn, prevención de fuga mediante pipelines: https://scikit-learn.org/1.8/common_pitfalls.html
