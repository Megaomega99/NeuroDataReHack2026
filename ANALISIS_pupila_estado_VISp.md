# Análisis: ¿la pupila indexa el estado poblacional de VISp?

**Proyecto:** NeuroDataReHack 2026 · Allen Visual Coding – Neuropixels (DANDI:000021)
**Sesión:** `sub-738651046_ses-760693773` · **Región:** VISp · **Régimen:** actividad espontánea
**Notebook:** `04_pupila_estado_VISp.ipynb`

> Este documento explica **qué medimos, de dónde sale cada métrica y qué significan los resultados**.
> Cada concepto va en dos niveles: **🟢 En cristiano** (sin matemáticas) y **🔧 Más técnico** (para el
> informe). La idea es que se entienda leyendo solo lo verde, y se pueda defender leyendo lo técnico.

---

## 0. TL;DR (resumen en 4 frases)

1. Aparentemente la pupila predice el estado de VISp (decoding con `r ≈ 0.31`, significativo, `p = 0.004`).
2. **Pero** la **locomoción (carrera) domina el estado cortical** (`r = 0.76`) y arrastra a la pupila (`r = 0.40`).
3. Al **descontar la carrera**, el vínculo pupila↔estado **se desploma** de `0.285` a `0.071` (`p = 0.79`, ya no significativo).
4. **Conclusión:** en esta sesión la pupila "indexa" el estado de VISp **casi solo porque ambos siguen al movimiento**; su aporte propio (arousal independiente de la locomoción) es despreciable, y no hay una latencia de arousal clara.

---

## 1. Los datos de esta sesión (contexto)

| Dato | Valor | Comentario |
|---|---|---|
| Bloque espontáneo | **20.62 min** (24 745 bins de 50 ms) | Suficiente → PLAN A |
| Unidades VISp tras QC | **92** | Buen tamaño de población |
| NaN de pupila (parpadeos) | **15.7 %** | Interpolados linealmente |
| Señal de carrera | `processing/running/running_speed` | Rango [0, 372] → contiene picos artefactuales |
| Varianza explicada PC1–PC3 | **7.8 / 4.8 / 4.1 %** (acum. 16.6 %) | Población de **alta dimensión** en espontáneo |

**🔧 Nota sobre la baja varianza de PC1.** Durante el espontáneo no hay un estímulo que sincronice a la
población, así que la actividad es de alta dimensionalidad: ningún eje único domina. Que PC1 capture solo
el 7.8 % implica que es un **resumen parcial** del estado; conviene recordarlo al interpretar cualquier
correlación "pupila ↔ PC1".

---

## 2. Diccionario de métricas (qué son y de dónde vienen)

### 2.1 Correlación `r`
**🟢 En cristiano.** Un número de −1 a +1: "cuando una señal sube, ¿la otra también?" `+1` = van idénticas;
`−1` = opuestas; `0` = nada que ver. Usamos el **valor absoluto |r|** porque el signo de PC1 es arbitrario
(el PCA puede devolver "PC1" o "−PC1"); nos interesa la **fuerza** del vínculo, no su signo.

**🔧 Más técnico.** Usamos **Spearman** (correlación de rangos) en lugar de Pearson: es Pearson calculada
sobre los *rankings* de los datos, no sobre los valores crudos. Ventajas aquí: (i) es robusta a valores
atípicos (la carrera tenía picos ~372), y (ii) capta relaciones monótonas aunque no sean estrictamente
lineales. `r²` (el cuadrado) es la fracción de varianza de rango compartida: `r = 0.76 → ~58 %`.

### 2.2 PC1 y el "estado neuronal"
**🟢 En cristiano.** Tenemos 92 neuronas. En vez de mirarlas una a una, buscamos **el patrón que más
comparten** y lo resumimos en una sola señal, PC1: "hacia dónde se mueve la población en conjunto".

**🔧 Más técnico.** PCA (Análisis de Componentes Principales) sobre la matriz de tasas z-scored
(bins × neuronas). PC1 es el autovector de mayor autovalor de la matriz de covarianza (equivalente a
correlación al estar estandarizada): la dirección de **máxima varianza compartida**. Se ajustó **solo con
el 70 % de entrenamiento** (`fit` train-only) para no filtrar información del futuro (*leakage*).

### 2.3 Correlación cruzada y *lag* (retardo)
**🟢 En cristiano.** La `r` normal compara ambas señales en el mismo instante. La **cruzada** desliza una en
el tiempo para ver si encajan mejor con un desfase. *Analogía:* el trueno llega después del relámpago;
deslizando el sonido encuentras el desfase (el **lag**). Buscábamos: "¿la pupila va retrasada respecto al
cortex, y cuánto?"

**🔧 Más técnico.** Calculamos `corr(neural[t], pupila[t+τ])` para τ en `±3 s` (±60 bins de 50 ms). Convención:
**lag positivo = la pupila va por detrás** del estado. Se implementó de forma eficiente con la **correlación
cruzada circular vía FFT** (`ifft(conj(Â)·B̂)`), matemáticamente equivalente a deslizar con `np.roll` pero
mucho más rápida. El estadístico reportado es el **máximo de |correlación|** sobre la ventana de lags.

### 2.4 El "nulo" y el p-valor (control por permutación) — la pieza central
**🟢 En cristiano.** Obtienes `|r| = 0.28`. ¿Es real o suerte? Fabricas **muchas versiones falsas** de los
datos donde, por construcción, NO hay relación verdadera pero se conserva la **misma textura** (la misma
"lentitud"). Mides la correlación en cada falsa → obtienes "lo que da el puro azar". Si tu valor real
supera al 95 % de las falsas → **significativo**. El **p-valor** es la fracción de falsas que igualan o
superan lo real: `p = 0.004` → solo el 0.4 % del azar llega tan alto (muy improbable que sea casualidad);
`p = 0.21` → el 21 % del azar llega ahí (podría ser suerte).

**🔧 Más técnico.** Nulo por **permutación de desplazamiento circular**: se rota cíclicamente una señal un
*offset* aleatorio. Esto **preserva la autocorrelación (el espectro de potencia) de cada señal** y solo
rompe la relación temporal *entre* ellas. Es imprescindible porque dos señales lentas y muy
autocorrelacionadas exhiben correlaciones espurias altas; un test paramétrico ingenuo (que asume muestras
independientes) daría significancia falsa. `p_emp = (1 + #{nulo ≥ observado}) / (N_perm + 1)`, con `N_perm = 2000`.

### 2.5 Drift (deriva) / no-estacionariedad
**🟢 En cristiano.** El registro dura 20 min; en ese tiempo las condiciones cambian despacio (el electrodo
se mueve un poco, el estado basal del ratón se desplaza), así que "el estado" del principio no es
comparable con el del final. *Analogía:* medir con un termómetro que se va descalibrando lentamente.

**🔧 Más técnico.** No-estacionariedad de media/escala entre *train* (primeros 14 min) y *test* (últimos
6 min). Rompe el supuesto de que un único `StandardScaler`/PCA ajustado en train transfiere a test; es la
causa de que el R² del decoding salga fuertemente negativo (ver 2.7).

### 2.6 Detrend (quitar la tendencia)
**🟢 En cristiano.** Arreglo del drift: le restas la "cuesta" lenta a la señal para quedarte con las
fluctuaciones rápidas y compararlas de forma justa.

**🔧 Más técnico.** `detrend_train`: ajusta una recta por mínimos cuadrados **solo con los índices de
train** y la resta de toda la serie (estado y features de pupila). Elimina la deriva lineal de orden bajo
sin usar información del test.

### 2.7 R² frente a `r`: por qué reportamos las dos
**🟢 En cristiano.** Ambas miden "qué tan bien el modelo reprodujo el estado", pero distinto:
- **R²** = ¿acertaste el **número exacto** (nivel incluido)? Estricta; si el drift desplazó el nivel, la
  castiga. Puede ser **negativa** = "peor que decir siempre el promedio".
- **`r`** = ¿acertaste las **subidas y bajadas** (la forma), aunque el nivel esté desplazado? **Robusta al drift**.

*Analogía:* predices la temperatura semanal; si aciertas el patrón día a día pero te equivocas siempre en
+5°, tu **`r` es alta** (forma buena) y tu **R² mala/negativa** (nivel mal). → **R² negativa + `r` positiva
= "la forma está, el drift desplazó el nivel"**.

**🔧 Más técnico.** `R² = 1 − SS_res/SS_tot`, con `SS_tot` alrededor de la media del **test**. Un desplazamiento
de media inducido por drift hace `SS_res > SS_tot` → `R² < 0`. La correlación `r(pred, real)` es invariante a
desplazamientos afines de media, de ahí que sea el diagnóstico correcto para separar "no hay señal" de "hay
señal pero hay drift".

### 2.8 Decoding: GLM y GRU
**🟢 En cristiano.** Un modelo recibe la **pupila** (ventana reciente) e intenta **reconstruir el estado**.
El **GLM** es un ajuste tipo "línea" simple; la **GRU** es una red neuronal con **memoria**. Se entrenan con
el primer 70 % y se prueban en el último 30 % **que nunca vieron** → prueba honesta de predicción real.

**🔧 Más técnico.** GLM gaussiano ≡ **regresión lineal regularizada** (`Ridge`, una por componente de Y):
como el objetivo es continuo, no hace falta `nemos` (reservado para GLMs de Poisson sobre spikes). La GRU
(*Gated Recurrent Unit*, `torch`) recibe una secuencia de `DEC_SEQ = 20` bins (1 s) de features de pupila y
resume el pasado reciente en su estado oculto. **Barrido de τ**: para cada retardo desplazamos la ventana
de pupila `pupila(t−τ … t) → estado(t)` y medimos la métrica en test. *Features* de pupila = **pupila
suavizada + su derivada** (sin `tsfel`, por decisión del SPEC).

### 2.9 Correlación **parcial** (control por carrera) — el análisis decisivo
**🟢 En cristiano.** Pupila y cortex podrían moverse juntos **solo porque ambos siguen a la carrera**
(correr dispara el cortex y también dilata la pupila). La correlación parcial pregunta: **"¿queda vínculo
pupila↔cortex DESPUÉS de descontar la carrera?"** Le quitas a la pupila "la parte que explica la carrera"
(te quedas con el **residuo**), haces lo mismo con el cortex, y correlacionas los residuos. Si siguen
correlacionados → hay vínculo propio. Si se desinfla → era todo carrera. *Analogía:* dos empleados parecen
coordinados, pero solo siguen al jefe; descontadas las órdenes del jefe, ya no coinciden.

**🔧 Más técnico.** Residualización lineal: regresamos cada señal sobre la carrera contemporánea
(`sig − [carrera, 1]·β` por mínimos cuadrados) y aplicamos la correlación cruzada + nulo circular sobre los
**residuos**. **Límite del control:** solo elimina la componente **lineal y contemporánea** de la carrera;
efectos retardados o no lineales de la locomoción podrían persistir. Es un control razonable, no una prueba
de mediación causal.

---

## 3. Resultados

### 3.1 Correlación cruzada pupila ↔ estado (Bloque 2)

| Señal neuronal | \|r\| | lag del pico | p (nulo circular) |
|---|---|---|---|
| **PC1** | 0.285 | −0.10 s | **0.207** (n.s.) |
| PC2 | 0.284 | +0.35 s | 0.399 (n.s.) |
| PC3 | 0.220 | +1.10 s | 0.087 (n.s.) |
| mean_z (media población) | 0.094 | −0.70 s | 0.667 (n.s.) |

- El mejor acoplamiento (PC1, |r|=0.285) **no supera el nulo** (p=0.21): para dos señales tan lentas, ese
  nivel de correlación es lo esperable por azar.
- Curiosidad: la **media de la población** (`mean_z`) apenas se acopla (0.094); PC1 —combinación ponderada—
  capta mejor el eje ligado a la pupila.
- **🔧** Los p-valores por señal son **exploratorios** (4 comparaciones, sin corrección de comparaciones
  múltiples). PC3 (p=0.087) es el único con el patrón "pupila retrasada ~1 s", pero no sobrevive y solo
  explica un 4 % de varianza.

### 3.2 Control por carrera — **la tabla decisiva** (Bloque 2.3)

| Comparación | \|r\| | p | Lectura |
|---|---|---|---|
| pupila ↔ PC1 (simple) | 0.285 | 0.21 | vínculo aparente, modesto |
| **pupila ↔ PC1 (parcial, sin carrera)** | **0.071** | **0.79** | **se desinfla a casi nada** |
| **carrera ↔ PC1** | **0.758** | **0.005** | **enorme y significativo** |
| carrera ↔ pupila | 0.403 | 0.005 | pupila y carrera van juntas |

Historia que se resuelve sola:
1. La **carrera domina el cortex** (`r = 0.76`): correr es, con diferencia, lo que más mueve VISp.
2. La **pupila acompaña a la carrera** (`r = 0.40`).
3. Por eso la pupila *parecía* ligada al cortex (0.285)…
4. …pero **descontando la carrera, el vínculo se evapora** (0.285 → **0.071**, n.s.).

> Es exactamente el *confound* que el SPEC advertía: "sin la carrera no puedes distinguir *la pupila indexa
> el cortex* de *ambas siguen a la carrera*". Aquí la respuesta es clara: **es lo segundo**.

### 3.3 Decoding: correlación robusta y R² (Bloque 3)

- La correlación pred↔real dibuja una **U invertida limpia**: pico **`r ≈ 0.31` (GLM, τ = −0.25 s)** y
  **`r ≈ 0.29` (GRU, τ = −0.5 s)**. Ambos métodos convergen cerca de lag 0 (pupila ligeramente retrasada).
- El **R² sigue negativo** en todo el rango (−0.6…−1.0) → síntoma de drift, no de ausencia de señal.

**Nulo del decoding (Bloque 3.4):**
```
r_max observado = 0.312
Nulo circular (500 perm): P50 = 0.018 · P95 = 0.050 · P97.5 = 0.053
p-valor = 0.0040  → SIGNIFICATIVO
```

**Confirmación del drift (Bloque 3.5):**

| | R² | r |
|---|---|---|
| sin detrend | −0.742 | 0.312 |
| con detrend | **−0.146** | 0.410 |

El R² sube hacia ~0 al quitar la deriva mientras la correlación se mantiene (sube a 0.41) → **el R²≈−0.75
era drift**, no falta de señal.

### 3.4 Forecasting intrínseco (Bloque 4, contraste)

| Horizonte | R² |
|---|---|
| 50 ms | 0.996 |
| 250 ms | 0.664 |
| 500 ms | 0.282 |

El estado predice su **propio** futuro muy bien a corto plazo (autocorrelación por el suavizado de 100 ms) y
decae con el horizonte. Contextualiza que el estado es rico y multidimensional; la pupila solo veía su eje
lento (y ese, vía carrera).

---

## 4. Reconciliar los dos "veredictos" (importante, porque choca)

Parece una contradicción:
- **Bloque 2** (correlación cruzada): pupila↔PC1 → **p = 0.21 (NO significativo)**.
- **Bloque 3.4** (nulo del decoding): pupila→estado → **p = 0.004 (SÍ significativo)**.

No se contradicen: **miden preguntas distintas.**

- **🔧 El Bloque 2** es un test **muy conservador para señales lentas**: pregunta si el emparejamiento *en el
  instante real* es más correlacionado que en *cualquier desfase circular al azar*. Como ambas son lentas,
  casi cualquier desfase da ~0.3 → el emparejamiento real no es "especial" → n.s. Detecta **acoplamiento
  temporalmente específico** (un lag privilegiado), y aquí no lo hay.
- **🔧 El decoding** pregunta algo distinto y más potente: **"¿la pupila predice el estado en datos nuevos
  (test) mejor que cuando rompo el vínculo?"** Sí lo predice (p=0.004) → poder **predictivo real y
  fuera de muestra**. **Pero no controla la carrera.**

**La correlación parcial resuelve la tensión:** el poder predictivo del decoding es real, pero lo lleva la
**locomoción** (pupila como espía del movimiento). Quitada la carrera, no queda predicción propia (0.071).
Las tres piezas, ya coherentes:

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿La pupila lleva información del estado? | **Sí** | decoding fuera de muestra, p=0.004 |
| ¿Esa info es suya (arousal) o es locomoción? | **Locomoción** | parcial: 0.285 → 0.071 (n.s.) |
| ¿Hay un retardo de arousal específico? | **No claro** | cruzada n.s.; pico pegado a lag 0 |

---

## 5. Respuesta a la pregunta del proyecto

> *¿En qué medida la pupila indexa el estado poblacional de VISp durante la actividad espontánea, y con qué
> latencia?*

**En esta sesión, la pupila predice el estado de VISp (r ≈ 0.31, decoding significativo), pero ese vínculo
está explicado casi por completo por la locomoción: la carrera domina el cortex (r = 0.76) y arrastra a la
pupila (r = 0.40). Descontando la carrera, la pupila no aporta prácticamente nada (r = 0.07, n.s.), y no
aparece una latencia de arousal clara.** La pupila, aquí, es más un **reflejo del movimiento** que un índice
independiente del estado cortical.

Es un resultado **sólido, honesto y bien controlado** — más informativo y defendible que un ingenuo "sí, la
pupila predice el cortex".

---

## 6. Límites declarados

- **Una sola sesión / un solo animal:** todo es **descriptivo**, no generaliza a una población.
- **Luminancia constante:** la lectura "pupila = arousal" es válida porque el bloque espontáneo es gris
  (justifica *a posteriori* elegir espontáneo frente a *gratings*).
- **Control de carrera parcial:** solo lineal y contemporáneo; efectos retardados/no lineales de la
  locomoción podrían quedar (ver 2.9).
- **PC1 explica poco (7.8 %):** es un resumen parcial del estado; otras definiciones podrían capturar ejes
  distintos (aunque `mean_z` fue peor y ninguna superó el nulo).
- **Artefactos de carrera:** la señal tenía picos (~372); Spearman mitiga su efecto, pero conviene depurarla
  para un análisis definitivo.

---

*Documento generado como acompañamiento del notebook `04_pupila_estado_VISp.ipynb`.*
