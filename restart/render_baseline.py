"""Render aggregate results as a Spanish report and standalone figure."""
import json
from pathlib import Path

p=Path('reports/baseline'); r=json.loads((p/'results.json').read_text())
lines=['# Primer análisis basal: SDMT y TMT','',
'El registro indicado fue retirado: 102 pacientes y 19 controles. La auditoría de edades y desenlaces basales pasa. Persisten faltantes ordinarios. El original permanece intacto. Resultados exploratorios, sin validación externa ni atribución causal.','',
'## Diferencias ajustadas','',
'Ajuste por edad, sexo y escolarización; IC95% HC3 con referencia t. Casos completos. Grupo=1 pacientes. Las razones TMT describen tiempos en escala logarítmica (razón de medias geométricas condicionales), no la razón de tiempos medios aritméticos.','',
'| Desenlace | N pacientes / controles | Efecto pacientes respecto a controles | IC95% | p |',
'|---|---:|---:|---|---:|']
for name,v in r['outcomes'].items():
 i=v['inference']; n=i['n']-i['n_controls']; nc=i['n_controls']
 if name=='sdmt': effect=i['group_effect']; ci=i['group_ci95']; label=f'{effect:.2f} puntos'; pv=i['group_p']
 else: effect=i['patient_control_time_ratio']; ci=i['time_ratio_ci95']; label=f'{effect:.2f} × tiempo'; pv=i['group_p_holm']
 lines.append(f'| {name.upper()} | {n} / {nc} | {label} | {ci[0]:.2f} a {ci[1]:.2f} | {pv:.4g} |')
lines += ['', 'p de TMT corregidas por Holm para A/B. Los otros análisis son exploratorios; no se han corregido todas las comparaciones secundarias.', '',
'## Discriminación fuera de muestra','',
'La desviación normativa compara el resultado observado con el predicho por ridge entrenado exclusivamente en controles del entrenamiento. La comparación incremental utiliza logística demográfica frente a logística demográfica+resultado cognitivo; son dos enfoques distintos.','',
'| Desenlace | AUC desviación normativa (IC95%) | AUC demográfica | AUC demográfica + test | Incremento AUC (IC95%) |',
'|---|---|---:|---:|---|']
for name,v in r['outcomes'].items():
 d=v['discrimination']; ci=v['bootstrap']['ci95']; a,b=ci['normative_deviation_auc']; c,e=ci['auc_increment']
 lines.append(f"| {name.upper()} | {d['normative_deviation']['auc']:.3f} ({a:.3f}–{b:.3f}) | {d['demographic']['auc']:.3f} | {d['demographic_cognitive']['auc']:.3f} | {d['auc_increment']:.3f} ({c:.3f}–{e:.3f}) |")
lines += ['', 'IC exploratorios: 200 bootstrap por persona con reajuste completo; los duplicados no cruzan folds. Todos los 200 remuestreos terminaron en cada desenlace. El incremento SDMT incluye cero; la evidencia de mejora incremental es incierta. TMT presenta mayor señal en esta muestra, pero no se compararon formalmente AUC entre pruebas y hay múltiples análisis.', '',
'| Desenlace | Sensibilidad logística ampliada | Especificidad | Balanced accuracy | Brier |',
'|---|---:|---:|---:|---:|']
for name,v in r['outcomes'].items():
 d=v['discrimination']['demographic_cognitive']
 lines.append(f"| {name.upper()} | {d['sensitivity']:.3f} | {d['specificity']:.3f} | {d['balanced_accuracy']:.3f} | {d['brier']:.3f} |")
lines += ['', 'Umbral 0,5 prefijado: la baja especificidad muestra que la AUC no basta para proponer uso clínico. Las probabilidades dependen de la proporción de pacientes del estudio; no son riesgos poblacionales calibrados.', '',
'## Regresión por grupo y aportación de TUG','',
'MAE/RMSE de SDMT en puntos; de TMT en escala logarítmica. Cinco folds externos, tres internos para alpha ridge. Comparación exploratoria, sin intervalos de mejora ni selección definitiva de variables.','',
'| Desenlace | Grupo | Modelo | MAE | RMSE | R² fuera de muestra |',
'|---|---|---|---:|---:|---:|']
for name,v in r['outcomes'].items():
 for a in v['regression_cv']:
  lines.append(f"| {name.upper()} | {'Pacientes' if a['group'] else 'Controles'} | {a['model']} | {a['mae']:.3f} | {a['rmse']:.3f} | {a['r2']:.3f} |")
lines += ['', 'En SDMT, TUG reduce el MAE ridge de 8,47 a 8,00 puntos en pacientes y de 9,47 a 9,02 en controles. Es una señal candidata, no prueba de relevancia estable. No se realizó búsqueda exhaustiva: 6MWT está vacío y 9HPT/T25P requieren depuración adicional.', '',
'## Coeficientes SDMT del modelo conjunto','',
'Referencias: hombre, escolarización básica, control. La edad no está centrada; el intercepto no representa un perfil clínico realista. Asociaciones ajustadas, no efectos causales.','',
'| Término | Coeficiente | IC95% | p exploratoria |','|---|---:|---|---:|']
for name,a in r['outcomes']['sdmt']['inference']['coefficients'].items():
 lines.append(f"| {name} | {a['estimate']:.3f} | {a['ci95'][0]:.3f} a {a['ci95'][1]:.3f} | {a['p']:.4g} |")
lines += ['', 'Las ecuaciones separadas por grupo, sus intervalos, interacción con edad y sensibilidad al rango de edad compartido se encuentran en `results.json`. No hay base para afirmar igualdad de pendientes por un p no significativo. No se han calculado aún intervalos de predicción para perfiles individuales.', '',
'## Potencia: escenarios SDMT','',
'Diseño de casos completos actual. Ejemplo con desviación típica de 12 puntos en ambos grupos, errores normales y α=0,05; 2.000 simulaciones por escenario. Efectos hipotéticos, no umbrales clínicos ni potencia observada.','',
'| Diferencia (puntos) | Potencia / error tipo I si diferencia=0 | Error Monte Carlo |','|---:|---:|---:|']
for a in r['outcomes']['sdmt']['power']['scenarios']:
 if a['sd_controls']==12 and a['sd_patient_ratio']==1:
  lines.append(f"| {a['difference_points']} | {a['power_or_type1_if_zero']:.3f} | {a['mc_se']:.3f} |")
lines += ['', 'La potencia para 5 puntos bajo este supuesto es aproximadamente 32%; para 8 puntos, 67%. Alcanzar significación con la diferencia estimada no implica potencia suficiente para efectos menores. El JSON incluye desviaciones de 8/12/16 y heterocedasticidad. No se ha calculado todavía el tamaño muestral necesario para perfiles, interacciones o validación externa.', '',
'## Estado y próximos análisis','',
'La primera ejecución basal está terminada. Antes de conclusiones confirmatorias: ampliar remuestreo, revisar estabilidad de variables y posibles valores motores anómalos, acordar diferencia mínima clínicamente relevante y perfiles, simular ampliación muestral, evaluar calibración/umbrales y validar en otra muestra. Digital, errores y retest quedan como análisis secundarios posteriores. No se ha certificado unicidad biológica de personas con códigos distintos ni ausencia de sesgo de selección.', '',
'Reproducibilidad y decisiones exactas: [especificación](../../restart/ANALISIS_BASAL.md). Datos agregados: [results.json](results.json). Auditoría: [audit_latest.json](../audit_latest.json).', '',
'Referencias metodológicas: [pipelines y fuga de información](https://scikit-learn.org/1.8/common_pitfalls.html), [tamaño muestral](https://www.bmj.com/content/368/bmj.m441).']
(p/'INFORME.md').write_text('\n'.join(lines)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
names=list(r['outcomes'])
for j,name in enumerate(names):
 v=r['outcomes'][name]; x=v['discrimination']['normative_deviation']['auc']; lo,hi=v['bootstrap']['ci95']['normative_deviation_auc']
 axes[0].errorbar(x,j,xerr=[[x-lo],[hi-x]],fmt='o',capsize=4,color='navy')
axes[0].set_yticks(range(3),[n.upper() for n in names]); axes[0].axvline(.5,color='gray',linestyle='--'); axes[0].set_xlim(.45,1); axes[0].set_xlabel('AUC e IC95% bootstrap'); axes[0].set_title('Desviación respecto a norma sana')
for sigma in [8,12,16]:
 a=[v for v in r['outcomes']['sdmt']['power']['scenarios'] if v['sd_controls']==sigma and v['sd_patient_ratio']==1]
 axes[1].plot([v['difference_points'] for v in a],[v['power_or_type1_if_zero'] for v in a],marker='o',label=f'DE={sigma}')
axes[1].axhline(.8,color='gray',linestyle='--'); axes[1].set_ylim(0,1); axes[1].set_xlabel('Diferencia hipotética SDMT (puntos)'); axes[1].set_ylabel('Probabilidad de rechazo'); axes[1].set_title('Potencia condicional, igual varianza'); axes[1].legend()
fig.suptitle('Análisis exploratorio basal — validación interna'); fig.tight_layout(); fig.savefig(p/'resumen.png',dpi=180); plt.close(fig)
print('Report and figure generated')
