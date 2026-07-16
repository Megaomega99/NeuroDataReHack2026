# Análisis de **grupo**: ¿la pupila indexa el estado poblacional de VISp? (N = 25 ratones)

**Proyecto:** NeuroDataReHack 2026 · Allen Visual Coding – Neuropixels (DANDI:000021)
**Notebook:** `04_pupila_estado_VISp2.ipynb` · **Régimen:** actividad espontánea · **Región:** VISp

> Continuación del análisis de sesión única (`ANALISIS_pupila_estado_VISp.md`). Aquí escalamos a
> **todas las sesiones con pupila** y hacemos **estadística de grupo**. Mantengo el doble nivel:
> **🟢 En cristiano** y **🔧 Más técnico**. El glosario base (correlación, PC1, nulo circular, drift,
> R² vs r, GLM/GRU, correlación parcial) está en el documento de sesión única; aquí solo añado lo nuevo.

---

## 0. TL;DR — el grupo **corrige** a la sesión única

1. En la **sesión única** concluimos que la pupila era casi solo un reflejo de la locomoción. **Pero esa
   sesión era atípica**: resultó ser la de **mayor** acoplamiento carrera↔estado de todo el dataset
   (|r|=0.76) y de las de **menor** acoplamiento parcial (0.07).
2. En el **grupo (25 sesiones)** el panorama cambia: la pupila↔estado es **robusta** (|r| mediano 0.40) y,
   al descontar la carrera, **baja pero NO se desploma** (→0.32; el descenso es significativo,
   p=0.0001, pero modesto).
3. Y lo más importante: el acoplamiento **parcial** (pupila↔estado *sin* carrera) es **significativo en
   14 de 25 sesiones** (esperado por azar ~1). → **La pupila indexa el estado de VISp más allá de la
   locomoción, de forma consistente entre ratones.**
4. El *decoding* lo confirma: la pupila decodifica el estado (r≈0.28) **igual o mejor** que la carrera
   (r≈0.20), y **conserva** su poder al descontar la carrera (→0.24, caída no significativa).

**Conclusión de grupo:** *la pupila SÍ es un índice (parcial pero genuino) del estado poblacional de
VISp, por encima de la locomoción.* La carrera contribuye (~40 % de la asociación), pero no la explica.

---

## 1. Métricas nuevas de este cuaderno

### 1.1 Test de Wilcoxon de rangos con signo (emparejado)
**🟢 En cristiano.** Tenemos, por sesión, dos números (p. ej. acoplamiento *simple* vs *parcial*).
Queremos saber si **de forma sistemática** uno es mayor que el otro entre sesiones, sin asumir que los
datos siguen una campana de Gauss. Wilcoxon mira, sesión a sesión, en qué dirección va la diferencia y
si esa dirección se repite lo bastante como para no ser casualidad.

**🔧 Más técnico.** Alternativa no paramétrica al *t-test* pareado: ordena por magnitud las diferencias
intra-sesión, suma los rangos con signo y evalúa si la mediana de la diferencia ≠ 0. Robusto a atípicos
y a no-normalidad (ideal con N=25 y distribuciones sesgadas de |r|).

### 1.2 Test binomial sobre la fracción significativa
**🟢 En cristiano.** Si en cada sesión hacemos un test con umbral p<0.05, por puro azar esperaríamos que
~5 % "salgan significativas". Si salen **muchas más** (p. ej. 14 de 25), eso *en sí mismo* es
improbable por azar. El test binomial pone número a esa sorpresa.

**🔧 Más técnico.** Bajo H₀ (sin efecto real), el nº de sesiones con p<0.05 sigue una Binomial(N, 0.05).
Comparamos el conteo observado contra esa distribución (`binomtest`, alternativa "greater"). Es un
**test de segundo nivel**: agrega evidencia débil de muchas sesiones en una afirmación de grupo fuerte.

### 1.3 Por qué el grupo vale más que una sesión
**🟢** Un ratón puede ser raro (¡y el nuestro lo era!). Repetir en 25 sujetos separa "casualidad de esta
sesión" de "fenómeno real". **🔧** Convierte un resultado **descriptivo** (n=1) en uno **inferencial** a
nivel de población de sesiones.

---

## 2. Datos del grupo

| Cantidad | Valor |
|---|---|
| Sesiones con pupila procesadas | 25 válidas (PLAN A + carrera) |
| Duración espontáneo por sesión | ~20–21 min |
| Unidades VISp tras QC | 35–107 (mediana ~77) |
| Varianza explicada por PC1 | mediana **0.118** (rango 0.07–0.27) → estado de alta dimensión, PC1 modesto |

---

## 3. Resultados

### 3.1 Correlación cruzada + control por carrera (análisis principal)

```
1) Pupila<->estado SIMPLE vs PARCIAL (Wilcoxon, H1: simple>parcial)
   medianas: simple=0.403  parcial=0.315   p=0.0001
2) Sesiones con acoplamiento PARCIAL significativo (p<0.05)
   14/25   (esperado por azar ~1.2)   binomial p≈0.0000
3) Sesiones con CARRERA<->estado significativo (p<0.05)
   15/25   binomial p≈0.0000
   |r| mediano  carrera<->estado = 0.200      carrera<->pupila = 0.403
```

Lectura:
- **La pupila↔estado es real y robusta** (|r| mediano 0.40; significativa en 14/25 sesiones).
- **Controlar la carrera la reduce, pero no la elimina:** 0.403 → 0.315 (descenso significativo,
  p=0.0001). **🔧** En varianza compartida (r²): 0.162 → 0.099, es decir **~39 % de la asociación
  pupila↔estado es atribuible a la locomoción**, y **~61 % sobrevive**.
- **El acoplamiento parcial es significativo en 14/25** (binomial p≈0): la pupila aporta información del
  estado **independiente de la carrera** en la mayoría de ratones.
- **La carrera también importa** (15/25 significativas), pero su acoplamiento típico (mediana 0.20) es
  **menor** que el de la pupila — al contrario que en la sesión única.
- **Panel de mediación:** las sesiones donde la pupila está más ligada a la carrera tienden a perder más
  acoplamiento al descontarla (tendencia positiva) → mediación **parcial**, no total.

### 3.2 Decoding de grupo (GLM / GRU)

```
Mediana de r (pred<->real, pico sobre tau):
  pupila  -> estado  GLM: 0.282   GRU: 0.273
  carrera -> estado  GLM: 0.203
  pupila  -> estado  GLM (parcial, sin carrera): 0.238

Carrera decodifica MEJOR que pupila (Wilcoxon, H1: carrera>pupila): p=0.9742
Pupila decodifica PEOR al quitar carrera  (Wilcoxon, H1: simple>parcial): p=0.1205
```

Lectura (un modelo predictivo, fuera de muestra, confirma la correlación):
- **La pupila decodifica el estado (r≈0.28) igual o mejor que la carrera (r≈0.20).** El test de que "la
  carrera decodifica mejor" da p=0.97 → **rechazado**: es la pupila la que decodifica algo más.
- **La pupila conserva su poder al descontar la carrera** (0.282 → 0.238; caída **no significativa**,
  p=0.12). Es decir, el decoding desde pupila **no** se explica por locomoción.
- **GLM ≈ GRU** (0.282 vs 0.273): la relación pupila→estado es esencialmente **lineal**; la red
  recurrente no añade nada → coherente con un índice de arousal lento y suave.

---

## 4. La corrección importante: la sesión única era un **outlier**

| Métrica | Sesión única (760693773) | Grupo (mediana, N=25) |
|---|---|---|
| carrera ↔ estado \|r\| | **0.758** (el **máximo** del dataset) | 0.200 |
| pupila ↔ estado parcial \|r\| | **0.071** (n.s., de los más bajos) | 0.315 (sig. en 14/25) |
| decoding pupila (GLM) | 0.312 | 0.282 |
| decoding pupila parcial | 0.182 | 0.238 |

**🟢** Elegimos, sin saberlo, justo la sesión donde la locomoción mandaba más y la pupila menos aportaba
por su cuenta. Por eso allí "la pupila = reflejo de la carrera" encajaba. **En el grupo eso no se
sostiene:** la carrera típica acopla la mitad y la pupila mantiene una señal propia.

**🔧** Es un recordatorio clásico de por qué n=1 no basta: la varianza entre sesiones es grande
(carrera↔estado va de 0.02 a 0.76 en el dataset), y un solo punto puede caer en la cola.

---

## 5. Reconciliación con la sesión única

No hay contradicción, hay **contexto**:
- En la sesión única, la carrera dominaba (0.76) y "se comía" casi todo el acoplamiento pupila↔estado
  al partializar → parecía que la pupila no aportaba nada. Correcto **para esa sesión**.
- En el grupo, la carrera típica es más modesta (0.20) y la pupila retiene señal en la mayoría de
  ratones. La conclusión general **no** es "pupila = carrera", sino "**pupila y carrera son índices
  parcialmente solapados; la pupila aporta información del estado más allá de la locomoción**".

---

## 6. Respuesta (a nivel de grupo) a la pregunta del proyecto

> *¿En qué medida la pupila indexa el estado poblacional de VISp durante la actividad espontánea?*

**A nivel de población (25 sesiones), la pupila es un índice parcial pero genuino del estado de VISp:
covaría con él (|r| mediano 0.40), lo decodifica (r≈0.28) y —crucialmente— mantiene un acoplamiento
significativo tras descontar la locomoción en 14/25 sesiones (muy por encima del azar). La carrera
contribuye (~40 % de la asociación y acoplamiento significativo en 15/25), pero no explica el vínculo
pupila↔estado.** La lectura *pupila ≈ arousal cortical* se sostiene mejor en el grupo que en la sesión
única — que resultó ser un caso extremo dominado por el movimiento.

---

## 7. Límites declarados

- **Control de carrera lineal y contemporáneo:** el "residuo" no elimina efectos **retardados o no
  lineales** de la locomoción. "Más allá de la carrera" significa "más allá de la carrera lineal
  simultánea". Una extensión sería regresar también la carrera desplazada (varios lags).
- **Acoplamientos individuales modestos (~0.3):** la fuerza del resultado está en la **consistencia entre
  sesiones** (tests de grupo), no en un efecto grande por sesión.
- **PC1 explica poca varianza (~12 %):** es un resumen parcial del estado; otras señales podrían capturar
  más (en la sesión única, `mean_z` fue peor; queda por barrer a nivel de grupo).
- **Un solo dataset y régimen (espontáneo, Neuropixels, Allen):** no generaliza a otras áreas, especies
  o estados. La extensión a *drifting gratings* (sección opcional del notebook) aún está pendiente.
- **Heterogeneidad y artefactos:** la señal de carrera tiene picos espurios en algunas sesiones; Spearman
  mitiga los atípicos, pero conviene depurar para un análisis definitivo.

---

*Documento generado como acompañamiento del notebook `04_pupila_estado_VISp2.ipynb`. Métricas base
explicadas en `ANALISIS_pupila_estado_VISp.md`.*
