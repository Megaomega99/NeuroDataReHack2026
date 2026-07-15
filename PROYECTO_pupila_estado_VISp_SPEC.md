# Especificación de proyecto — Pupila como índice del estado poblacional de VISp

> **Documento para Claude Code.** Es una especificación de implementación, no código.
> El objetivo es construir un único Jupyter notebook que ejecute el análisis descrito
> abajo, respetando las decisiones científicas justificadas en cada sección. Varias de
> estas decisiones se tomaron descartando alternativas por razones concretas: **no las
> reviertas** sin consultar. Donde diga "reutilizar", el código ya existe y funciona.

---

## 0. Contexto y objetivo

**Objetivo científico único:** cuantificar en qué medida la pupila indexa el estado
poblacional de la corteza visual primaria (VISp) del ratón durante la **actividad
espontánea**, y con qué **latencia temporal**.

Es un proyecto de hackathon (NeuroDataReHack 2026) con **1 día** de plazo. La prioridad
es una sola pregunta bien contestada, no un pipeline extenso. El orden de los bloques de
abajo es también el orden de prioridad: si falta tiempo, se corta por el final.

**Idioma:** comentarios y markdown del notebook en español; nombres de variables en
inglés o español, indistinto, pero consistentes.

---

## 1. Datos y acceso

- **Dataset:** DANDI `000021` (Allen Institute – Visual Coding – Neuropixels).
- **Sesión:** `sub-738651046/sub-738651046_ses-760693773.nwb`.
- **Región objetivo:** `VISp`.
- **Acceso:** por **streaming** (el archivo pesa varios GB, no se descarga entero).
  Patrón: `DandiAPIClient` → URL S3 → `remfile.File` → `h5py.File` → `pynwb.NWBHDF5IO`.
- **Lectura perezosa:** los `spike_times` se leen solo para las unidades seleccionadas.

**Rutas relevantes dentro del NWB:**
- Pupila (área): `nwb.processing['filtered_gaze_mapping']['pupil_area']`
  (tiene `.data` y `.timestamps`; contiene NaN por parpadeos).
- Bloque espontáneo: `nwb.intervals['spontaneous_presentations']`
  (tabla con `start_time`, `stop_time`).
- Unidades: `nwb.units` (con `spike_times`, `peak_channel_id`, `quality`,
  `isi_violations`, `amplitude_cutoff`, `presence_ratio`, `firing_rate`, `snr`).
- Región por unidad: puente `nwb.units['peak_channel_id']` → `nwb.electrodes['location']`.
- Velocidad de carrera: **ubicación variable entre versiones**; hay que localizarla
  escaneando `nwb.processing` y `nwb.acquisition` por nombres que contengan
  `run` / `speed` / `velocity`, e imprimir los candidatos. Dejar una línea editable.

---

## 2. Librerías

- `pynwb`, `remfile`, `h5py`, `dandi` — streaming NWB.
- `numpy`, `pandas`, `matplotlib` — núcleo.
- `pynapple` — alineación de spikes y señales en rejilla común (`TsGroup.count`,
  `Tsd.bin_average`, `IntervalSet`).
- `scikit-learn` — `StandardScaler`, `PCA`, `r2_score`.
- `scipy.stats` — correlaciones y utilidades.
- Modelado (solo bloque 3): un GLM (lineal) y una GRU (`torch`).

**NO usar** (decisiones tomadas, ver justificación en cada bloque): `tsfel`, `xgboost`,
ni modelos encadenados decoding+forecasting.

---

## 3. Parámetros centrales (una sola celda de configuración)

```
REGION            = "VISp"
BIN               = 0.050     # s  (50 ms)
SMOOTH_SIGMA      = 0.100     # s  (suavizado gaussiano de tasas)
QC_ISI_MAX        = 0.5
QC_AMP_CUTOFF_MAX = 0.1
QC_PRESENCE_MIN   = 0.9
QC_QUALITY        = "good"
N_PCA             = 3         # componentes del estado neuronal
FRAC_TRAIN        = 0.70      # split temporal secuencial, sin shuffle
LAG_MAX           = 3.0       # s  (rango de la correlacion cruzada: +/- LAG_MAX)
N_PERM            = 2000      # permutaciones circulares para el nulo
TAU_GRID          = [-2.0 ... +2.0 en pasos de 0.25 s]  # barrido de lag del decoding
SEED              = 42
```

Ventana de análisis: **el bloque espontáneo completo**, no un recorte corto. (Ver bloque 0.)

---

## 4. Bloques de implementación

### Bloque 0 — Validación de viabilidad (PRIMERO, innegociable)

**Construir:** una celda que, tras abrir el NWB, imprima:
1. Duración total del bloque espontáneo (sumando `stop_time - start_time` de
   `spontaneous_presentations`), en minutos.
2. Número de unidades de VISp que superan el QC (región == VISp **y** quality == good
   **y** isi_violations ≤ 0.5 **y** amplitude_cutoff ≤ 0.1 **y** presence_ratio ≥ 0.9).

**Por qué:** todo el proyecto asume suficiente espontáneo y suficientes neuronas limpias
en esta sesión; hay que verificarlo antes de invertir tiempo. Es lo único que puede
obligar a cambiar de plan mientras aún hay margen.

**Criterio de decisión (imprimir un veredicto):**
- Si ≥ ~4–5 min de espontáneo **y** ≥ ~30 unidades VISp tras QC → PLAN A (seguir).
- Si no → PLAN B: incluir además los bloques `natural_movie_*` (se repiten, dan más
  minutos) dejando **anotado explícitamente** que introducen componente visual y que la
  interpretación "pupila = arousal" se debilita ahí.

---

### Bloque 1 — Sustrato común: alineación pupila ↔ spikes

**Construir:**
1. Selección de unidades VISp + QC (máscara booleana sobre metadatos; leer solo columnas
   escalares, no los spike_times todavía).
2. `IntervalSet` con **todos** los intervalos del bloque espontáneo.
3. Cargar `spike_times` solo de las unidades seleccionadas, recortados al espontáneo →
   `nap.TsGroup`.
4. Cargar pupila, **interpolar NaN** (`np.interp` sobre los huecos) → `nap.Tsd`.
5. Cargar velocidad de carrera (tras localizarla, ver §1) → `nap.Tsd`.
6. Binear todo a la MISMA rejilla: `TsGroup.count(BIN, ep) / BIN` → tasas;
   `.smooth(SMOOTH_SIGMA)`; `Tsd.bin_average(BIN, ep)` para pupila y carrera.
7. Alinear longitudes (mismo nº de bins) y `assert` de que los tiempos coinciden.
8. PCA sobre las tasas z-scored para obtener el **estado neuronal Y** (`n_bins × N_PCA`),
   **ajustando scaler y PCA solo con el tramo train** (los primeros FRAC_TRAIN bins).

**Por qué cada cosa:**
- Rejilla común: sin mismo reloj y misma resolución no hay comparación posible.
- Suavizado ~100 ms: un bin de 50 ms de una neurona es casi siempre 0/1 spike (ruido);
  el acoplamiento de estado vive en cientos de ms a segundos.
- PCA: la pupila es unidimensional; no tiene sentido correlacionar contra ~90 neuronas
  sueltas. PC1 suele capturar la modulación global de estado.
- Carrera registrada: arousal y locomoción son disociables y la carrera domina la señal;
  sin ella no se puede distinguir "pupila indexa cortex" de "ambas siguen a la carrera".
- Fit train-only: evita que la definición del estado use información del futuro (leakage).

**Reutilizar:** el notebook previo del hackathon ya implementa streaming, QC, `pynapple`,
PCA train-only correctamente. El único cambio es aplicarlo al **bloque espontáneo
completo**, no a 5 s de gratings.

**Entregar además:** una figura de comprobación (raster de una porción + pupila alineada
debajo, compartiendo eje X) para verificar visualmente la alineación.

---

### Bloque 2 — Correlación cruzada con lag + control nulo (EL CIMIENTO)

**Construir:**
1. Señal neuronal poblacional 1D: usar **PC1** (o media de z-scores de las unidades;
   dejar ambas opciones, PC1 por defecto).
2. Correlación cruzada entre pupila y señal neuronal en el rango de lags `±LAG_MAX`.
   Reportar el **lag del pico** y el valor de correlación ahí. Usar correlación de
   Spearman o Pearson (Spearman preferible por la no-gaussianidad; dejar parámetro).
3. **Control por permutación de desplazamiento circular:** repetir `N_PERM` veces,
   desplazando cíclicamente una señal respecto a la otra (con `np.roll` por un offset
   aleatorio), recalculando la correlación máxima; construir la distribución nula y
   obtener un p-valor empírico (fracción del nulo que iguala o supera lo observado).

**Por qué:**
- Correlación cruzada y no simple: la pupila va **retrasada** ~0.5–2 s respecto a la
  actividad neuronal (vía periférica lenta). Un análisis a lag cero perdería o
  infravaloraría la relación. El lag del pico es un resultado cuantitativo interpretable.
- Permutación circular **innegociable**: ambas señales son lentas y muy
  autocorrelacionadas; el p-valor ingenuo asume independencia y daría significancia
  espuria a cualquier par de señales lentas. El desplazamiento circular **preserva la
  autocorrelación de cada señal** y rompe solo la relación entre ambas → nulo correcto.

**Entregar:** figura de la función de correlación cruzada vs lag, con la banda del nulo
(percentiles del nulo) marcada y el lag del pico anotado. Esta es la figura central.

**Nota de prioridad:** este bloque **nunca se corta**. Responde la pregunta por sí solo.

---

### Bloque 3 — Decoding directo con barrido de lag (REFUERZO)

**Construir:**
1. Un decoder `pupila(t−τ … t) → estado(t)` con barrido de τ sobre `TAU_GRID`.
2. Dos modelos: **GLM** (lineal, baseline) y **GRU** (no lineal, recibe una ventana de
   pupila reciente). Métrica: R² en **test** (split secuencial, sin shuffle).
3. Como features de pupila basta la **pupila suavizada y su derivada** (no usar tsfel).
4. Reportar R² vs τ para cada modelo; identificar el τ óptimo.

**Por qué así y NO encadenando modelos:**
- Se descartó explícitamente la cadena "pupila → estado pasado → forecasting → estado
  actual": no aporta información nueva y solo compone error, porque propagar hacia
  delante una reconstrucción empobrecida (la sombra de arousal que la pupila recupera) no
  regenera la dinámica fina perdida al decodificar. El lag se incorpora **dentro de un
  único decoder** (τ), no con una etapa de forecasting intermedia.
- La GRU con ventana de pupila ya incorpora implícitamente "leer el pasado reciente".
- tsfel/xgboost eliminados: doce features sobre pocos datos es fragilidad innecesaria;
  GLM (base) + GRU (no lineal) es comparación suficiente.

**Validación cruzada esperada:** el τ óptimo del decoding debería **coincidir** con el
lag del pico de la correlación cruzada (bloque 2). Si convergen, dos métodos
independientes apuntan al mismo número → robustez. Comentarlo en el notebook.

**Reutilizar:** GLM y GRU de la "Tarea B" del notebook previo, añadiendo el desplazamiento τ.

**Recorte si falta tiempo:** dejar solo el GLM (sin GRU).

---

### Bloque 4 — Contraste: forecasting intrínseco del estado (OPCIONAL)

**Construir:** forecasting autorregresivo puro `estado(t−P … t) → estado(t+h)` a varios
horizontes h, **sin pupila**. Pieza independiente y claramente etiquetada.

**Por qué:** no responde la pregunta central, pero contextualiza el límite: comparar
cuánto predice el estado completo su propio futuro frente a cuánto reconstruye la pupila
el estado presente hace visible que la actividad poblacional es multidimensional y la
pupila solo ve su eje lento de arousal. El forecasting desde el estado completo predecirá
bastante mejor, y esa brecha **es** el argumento de que la pupila es un índice parcial.

**Recorte:** es lo **primero** que se elimina si el día va justo.

---

### Bloque 5 — Relato, figuras y límites declarados (ÚLTIMA HORA)

**Construir:** tres figuras en orden narrativo + celda de conclusiones.

**Narrativa:**
1. Pupila y estado poblacional están acopladas, pupila retrasada ~X s
   (figura correlación cruzada + nulo, del bloque 2).
2. Por eso se puede reconstruir el estado desde la pupila; R² máximo a ese lag
   (figura R² vs τ, del bloque 3).
3. Pero la pupila es unidimensional (arousal) y captura solo una fracción; el forecasting
   desde el estado completo predice mucho mejor (del bloque 4, si se hizo).

**Límites a declarar explícitamente (protegen el proyecto):**
- Es **una sola sesión**: todo es descriptivo, no generaliza a población de animales.
- La interpretación "pupila = arousal" es válida **porque** se trabaja bajo luminancia
  constante (el gris del bloque espontáneo); esto justifica a posteriori la elección del
  régimen espontáneo frente a los gratings.

---

## 5. Decisiones cerradas (no revertir sin consultar)

- **Régimen = espontáneo**, no drifting gratings. Con estímulo visual, la actividad de
  VISp está dominada por respuestas evocadas y "estado de arousal" se confunde con
  "respuesta al estímulo"; además la luminancia cambiante rompe el supuesto pupila=arousal.
- **Ventana = bloque espontáneo completo**, no recortes de segundos. Con pocos bins
  ningún R² ni correlación es fiable.
- **No encadenar** decoding + forecasting (compone error, no añade información).
- **Sin tsfel, sin xgboost.**
- **Split temporal secuencial sin shuffle**; scaler y PCA ajustados solo con train.
- **Permutación circular obligatoria** para la significancia del bloque 2.

## 6. Prioridad ante falta de tiempo

1. Bloque 0 — siempre, a primera hora.
2. Bloque 1 — siempre (sustrato).
3. Bloque 2 — **nunca se corta** (responde la pregunta).
4. Bloque 3 — reducir a solo GLM si hace falta.
5. Bloque 4 — primero en eliminarse.
6. Bloque 5 — siempre (el relato y los límites valen tanto como el análisis).

## 7. Riesgos conocidos

- Ubicación de la señal de carrera variable → escanear y dejar línea editable.
- Bins de pupila vacíos (NaN) tras binear → interpolar sobre índice de bin.
- Pocos datos espontáneos o pocas unidades → activar PLAN B (bloque 0).
- No se puede validar contra la sesión real fuera del entorno de ejecución: comprobar
  dimensiones y nombres de columnas en las primeras celdas.
