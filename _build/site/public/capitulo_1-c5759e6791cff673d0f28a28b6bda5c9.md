# Capítulo 1 de Machine Learning: "¿Qué es el aprendizaje automático?" (MyST / Jupyter Book v2)

El capítulo está listo para pegar en un archivo `.md` de Jupyter Book v2. La principal advertencia es esta: la definición más citada de *machine learning*, "campo de estudio que da a los computadores la capacidad de aprender sin ser programados explícitamente", **no aparece textualmente** en el artículo de Arthur Samuel de 1959, así que el capítulo cita las frases que sí aparecen en ese artículo y presenta la definición de Mitchell (1997) como la definición formal de referencia.

## TL;DR

- **Definiciones verificadas:** Mitchell (1997) es la definición operativa estándar, y Murphy (2022) la adopta textualmente. Bishop (2006), Murphy (2012), Hastie et al. (2009), James et al. (2021) y Goodfellow et al. (2016) ofrecen formulaciones complementarias, que el capítulo cita textualmente. La frase atribuida a Samuel es una paráfrasis que no figura en su artículo de 1959 y se presenta marcada como tal.
- **Núcleo del capítulo:** una taxonomía por tipo de supervisión (supervisado, no supervisado, semisupervisado, autosupervisado, por refuerzo) y por otros ejes (batch/online, instancias/modelos, paramétrico/no paramétrico, generativo/discriminativo, transferencia, activo, federado). Va con tabla comparativa, diagrama Mermaid, formalización (riesgo esperado vs. empírico, sesgo-varianza, No Free Lunch) y cinco prácticas con scikit-learn.
- **Confiabilidad:** casi todas las afirmaciones fácticas y referencias se contrastaron con la editorial, un DOI o el texto original, y se estiman en ≥95 %. Cuatro detalles bibliográficos quedan por debajo de ese umbral y se marcan explícitamente: la página exacta de la definición de Mitchell (que solo indican fuentes secundarias), las páginas de Wirth y Hipp (2000), la paginación de Vaswani et al. (2017), que difiere entre la edición de Curran Associates y la de ACM, y la segunda edición de Géron. Nada de esto cambia el contenido conceptual.

## Hallazgos clave y nivel de confiabilidad

| Afirmación | Fuente verificada | Confiabilidad |
|---|---|---|
| La frase "learn without being explicitly programmed" no está textual en Samuel (1959) ni en su versión de 1967 | Revisión del texto citada en foros técnicos; las frases reales sí constan en el artículo | 95 % |
| Frase real de Samuel: "Programming computers to learn from experience should eventually eliminate the need for much of this detailed programming effort." | Samuel (1959), citado también por Nilsson | 97 % |
| Definición E-T-P de Mitchell (1997) | Mitchell (1997); reproducida textualmente por Murphy (2022, §1.1) | 99 % (texto); 90 % (que esté en la p. 2) |
| Definición de reconocimiento de patrones de Bishop (2006, cap. 1) | Texto del capítulo 1 | 98 % |
| Definición de Murphy (2012) | Borrador oficial del autor (UBC) | 98 % |
| "Statistical learning refers to a vast set of tools for understanding data" | James et al. (2021, p. 1, Springer) | 99 % |
| "Machine learning is essentially a form of applied statistics…" | Goodfellow et al. (2016, cap. 5, sitio oficial) | 99 % |
| Breiman (2001): "Estimated culture population. 98% of all statisticians" (modelado de datos) frente a "2% of statisticians, many in other fields" (modelado algorítmico) | Statistical Science 16(3), 199–231 | 99 % |
| Gender Shades: error de hasta 34,7 % en mujeres de piel oscura (de 20,8 % en Microsoft a 34,7 % en IBM) frente a 0,8 % máx. en hombres de piel clara | Buolamwini y Gebru (2018), PMLR 81:77–91; MIT News | 99 % |
| Obermeyer et al. (2019): el sesgo surge porque el modelo predice costos en lugar de necesidad | Science 366(6464), 447–453 | 99 % |
| CONPES 4144 (14-feb-2025), seis ejes, "106 acciones hasta 2030 con una inversión total aproximada de 479.273 millones de pesos" | PDF del DNP y fuentes secundarias coincidentes | 96 % |

---

## Capítulo listo para copiar y pegar

Guárdelo, por ejemplo, como `capitulos/01_que_es_ml.md`. Las celdas `{code-cell}` se ejecutan si construye con `myst build --execute` (o `jupyter book build --execute`) y tiene un kernel de Python con `scikit-learn`, `numpy`, `pandas` y `matplotlib`. Si prefiere bloques estáticos, cambie ` ```{code-cell} python ` por ` ```python `.

````markdown
---
kernelspec:
  name: python3
  display_name: "Python 3"
---

# Capítulo 1. ¿Qué es el aprendizaje automático?

```{admonition} Objetivos de aprendizaje
:class: important
Al terminar este capítulo, usted debería ser capaz de:

1. Enunciar y comparar las definiciones clásicas del aprendizaje automático (Samuel, Mitchell, Bishop, Murphy, Hastie et al., James et al., Goodfellow et al.) e identificar cuáles son citas textuales y cuáles son paráfrasis populares.
2. Distinguir con precisión entre inteligencia artificial, aprendizaje automático, aprendizaje profundo, estadística y ciencia de datos.
3. Situar en una línea de tiempo los hitos principales del campo, incluidos los llamados "inviernos de la IA".
4. Formalizar el problema de aprendizaje: función objetivo, hipótesis, espacio de hipótesis, función de pérdida, riesgo esperado y riesgo empírico.
5. Explicar la generalización, el sobreajuste, el subajuste, el compromiso sesgo-varianza y el teorema *No Free Lunch*.
6. Clasificar un problema real según la taxonomía del aprendizaje automático y justificar la elección de una familia de algoritmos.
7. Describir el ciclo de vida de un proyecto de ML (CRISP-DM) y el papel de los conjuntos de entrenamiento, validación y prueba.
8. Implementar en Python, con scikit-learn, ejemplos básicos de regresión, clasificación, agrupamiento y reducción de dimensionalidad.
9. Reconocer riesgos éticos y fuentes de sesgo algorítmico, con ejemplos documentados en la literatura.
```

## 1.1 Introducción: aprender de los datos

Piense en cómo funciona un filtro de correo no deseado. Una opción es escribir reglas a mano: "si el asunto contiene *GANASTE* y hay más de tres signos de exclamación, márcalo como spam". Funciona un tiempo, hasta que los emisores cambian de táctica y hay que reescribir las reglas. La otra opción es mostrarle al computador miles de correos ya etiquetados como "spam" o "legítimo" y dejar que un algoritmo *encuentre* por sí mismo los patrones que los distinguen. Esa segunda vía es, en esencia, el aprendizaje automático (*machine learning*, ML).

El cambio de fondo es de paradigma. En la programación tradicional una persona escribe las reglas y el computador las aplica a los datos. En el aprendizaje automático, el computador recibe datos (y, a menudo, las respuestas esperadas) y produce las reglas, es decir, un **modelo**. Ese modelo luego se usa sobre datos nuevos que nunca vio.

```{note}
Esto no significa que "no se programe nada". Alguien diseña la representación de los datos, elige la familia de modelos, define qué significa "hacerlo bien" y escribe el procedimiento de ajuste. Lo que se aprende de los datos son los *parámetros* o la *estructura* del modelo, no el programa completo. El propio Arthur Samuel, pionero del área, programó cuidadosamente su sistema de damas.
```

## 1.2 Definiciones de aprendizaje automático

### 1.2.1 Arthur Samuel (1959): el origen del término y una cita mal atribuida

Arthur L. Samuel, investigador de IBM, publicó en 1959 el artículo *Some Studies in Machine Learning Using the Game of Checkers* en el *IBM Journal of Research and Development* (vol. 3, n.º 3, pp. 210–229). Allí describió un programa que mejoraba su juego de damas a partir de la experiencia, y el artículo se reconoce como una de las primeras apariciones del término *machine learning*.

```{warning}
**Punto ciego frecuente.** En libros, cursos y sitios web circula la "definición de Samuel (1959)": *"Field of study that gives computers the ability to learn without being explicitly programmed"*. **Esa frase no aparece textualmente ni en el artículo de 1959 ni en su segunda versión de 1967.** Todo indica que es una paráfrasis posterior que condensa la idea del artículo. Si la usa, preséntela como paráfrasis atribuida, no como cita textual.

Hay un segundo error recurrente: algunas fuentes citan el artículo como "IBM Journal 44(1.2)". Ese volumen corresponde a una **reimpresión del año 2000**; la publicación original es el vol. 3, n.º 3 de 1959.
```

Lo que Samuel sí escribió (cita textual del artículo de 1959) fue:

> "Programming computers to learn from experience should eventually eliminate the need for much of this detailed programming effort."
>
> *(Traducción propia: "Programar computadores para que aprendan de la experiencia debería, con el tiempo, eliminar la necesidad de gran parte de este esfuerzo de programación detallada").*

Y también:

> "A computer can be programmed so that it will learn to play a better game of checkers than can be played by the person who wrote the program."
>
> *(Traducción propia: "Un computador puede programarse de modo que aprenda a jugar damas mejor que la persona que escribió el programa").*

La segunda frase es especialmente poderosa: plantea que el sistema puede superar a su creador en la tarea, algo que hoy damos por sentado en ajedrez o Go, pero que en 1959 era una afirmación audaz.

### 1.2.2 Tom Mitchell (1997): la definición operativa E-T-P

La definición más útil en la práctica es la de Tom M. Mitchell en su libro *Machine Learning* (McGraw-Hill, 1997):

> "A computer program is said to learn from experience *E* with respect to some class of tasks *T* and performance measure *P*, if its performance at tasks in *T*, as measured by *P*, improves with experience *E*."
>
> *(Traducción propia: "Se dice que un programa de computador aprende de la experiencia E con respecto a una clase de tareas T y una medida de desempeño P si su desempeño en las tareas de T, medido por P, mejora con la experiencia E").*

Su valor está en que obliga a especificar tres cosas antes de hablar de "aprendizaje":

| Componente | Pregunta que responde | Ejemplo: filtro de spam | Ejemplo: damas (Mitchell) |
|---|---|---|---|
| **T** (tarea) | ¿Qué debe hacer el sistema? | Clasificar correos en spam / legítimo | Jugar damas |
| **P** (desempeño) | ¿Cómo medimos si lo hace bien? | Exactitud, precisión, *recall* | Porcentaje de partidas ganadas |
| **E** (experiencia) | ¿De qué datos o interacciones aprende? | Correos etiquetados por usuarios | Partidas jugadas contra sí mismo |

```{tip}
Use la terna E-T-P como "prueba de humo" al formular cualquier proyecto. Si no puede decir con claridad cuál es la tarea, cuál es la métrica y de dónde sale la experiencia, todavía no tiene un problema de aprendizaje automático bien planteado.
```

Kevin Murphy adopta esta misma definición como punto de partida en *Probabilistic Machine Learning: An Introduction* (2022, §1.1) y añade que, por eso, hay muchos tipos de ML, según la naturaleza de la tarea, de la medida de desempeño y de la señal de entrenamiento.

### 1.2.3 Otras definiciones de referencia

**Christopher Bishop (2006)**, *Pattern Recognition and Machine Learning*, abre el libro con una definición de reconocimiento de patrones que, en la práctica, funciona como definición de ML:

> "The field of pattern recognition is concerned with the automatic discovery of regularities in data through the use of computer algorithms and with the use of these regularities to take actions such as classifying the data into different categories."

Es decir: descubrir regularidades **automáticamente**, con algoritmos, y **usarlas para actuar** (por ejemplo, clasificar).

**Kevin Murphy (2012)**, *Machine Learning: A Probabilistic Perspective*, propone:

> "[…] we can define machine learning as a set of methods that can automatically detect patterns in data, and then use the uncovered patterns to predict future data, or to perform other kinds of decision making under uncertainty."

Aquí aparece un matiz clave: la **incertidumbre**. El ML no promete certezas; produce predicciones y decisiones bajo incertidumbre, idealmente cuantificada.

**Hastie, Tibshirani y Friedman (2009)**, *The Elements of Statistical Learning* (2.ª ed.), parten de la estadística: los datos se generan en enormes volúmenes y el trabajo del estadístico es "extraer patrones y tendencias importantes, y entender 'lo que dicen los datos'". Lo resumen en una frase: *"We call this learning from data"* ("A esto lo llamamos aprender de los datos").

**James, Witten, Hastie y Tibshirani (2021)**, *An Introduction to Statistical Learning* (2.ª ed.):

> "Statistical learning refers to a vast set of tools for understanding data. These tools can be classified as supervised or unsupervised."

**Goodfellow, Bengio y Courville (2016)**, *Deep Learning*, sitúan el ML dentro de la IA: los sistemas basados en conocimiento codificado a mano tienen límites, por lo que "los sistemas de IA necesitan la capacidad de adquirir su propio conocimiento, extrayendo patrones de datos en bruto. Esta capacidad se conoce como aprendizaje automático" (traducción propia del cap. 1). En el capítulo 5 añaden una caracterización muy honesta:

> "Machine learning is essentially a form of applied statistics with increased emphasis on the use of computers to statistically estimate complicated functions and a decreased emphasis on proving confidence intervals around these functions."

### 1.2.4 Síntesis

Si juntamos estas definiciones aparecen cuatro ideas comunes:

1. **Datos/experiencia** como materia prima.
2. **Automatización**: el patrón lo encuentra un algoritmo, no una persona.
3. **Generalización**: lo aprendido debe servir para datos nuevos, no solo para los vistos.
4. **Un criterio de desempeño** explícito (P en Mitchell, la pérdida en la formulación estadística).

```{admonition} Definición de trabajo para este curso
:class: tip
El **aprendizaje automático** es el estudio y la construcción de algoritmos que, a partir de datos o experiencia, ajustan un modelo para mejorar su desempeño en una tarea, medido por un criterio explícito, de manera que ese desempeño se mantenga en datos nuevos.
```

## 1.3 IA, ML, aprendizaje profundo, estadística y ciencia de datos

Estos términos se usan a menudo como sinónimos, pero no lo son.

- **Inteligencia artificial (IA):** el campo amplio que busca construir sistemas que realicen tareas que asociamos con la inteligencia (razonar, planificar, percibir, comunicarse). Incluye enfoques que **no** aprenden de datos, como los sistemas expertos basados en reglas o la búsqueda heurística.
- **Aprendizaje automático (ML):** el subcampo de la IA donde el comportamiento se obtiene ajustando modelos a datos.
- **Aprendizaje profundo (*deep learning*, DL):** el subconjunto del ML basado en redes neuronales con muchas capas, que aprenden **representaciones jerárquicas** de los datos (Goodfellow et al., 2016).
- **Estadística:** la disciplina que estudia la recolección, el análisis y la inferencia a partir de datos bajo incertidumbre. Comparte con el ML gran parte de su fundamento matemático, pero históricamente ha puesto más énfasis en la inferencia (intervalos, pruebas, interpretabilidad de parámetros) que en la predicción.
- **Ciencia de datos:** un campo aplicado e interdisciplinario que combina estadística, computación, ML, ingeniería de datos y conocimiento del dominio para extraer valor de los datos. Usa ML, pero también incluye limpieza de datos, visualización, comunicación, gobierno de datos y diseño experimental.

```{mermaid}
flowchart TB
    subgraph IA["Inteligencia Artificial"]
        direction TB
        R["Sistemas basados en reglas,<br/>búsqueda, lógica"]
        subgraph ML["Aprendizaje automático"]
            direction TB
            C["Métodos clásicos:<br/>regresión, árboles, SVM, k-NN"]
            subgraph DL["Aprendizaje profundo"]
                T["Redes profundas,<br/>Transformers, LLM"]
            end
        end
    end
    E["Estadística"] -. "fundamentos compartidos" .-> ML
    CD["Ciencia de datos"] -. "usa" .-> ML
    CD -. "usa" .-> E
```

### 1.3.1 Las "dos culturas" de Breiman (2001)

Leo Breiman, estadístico de Berkeley y creador de Random Forest, publicó en 2001 *Statistical Modeling: The Two Cultures* en *Statistical Science* (16(3), 199–231). Allí distingue dos formas de usar modelos para sacar conclusiones de los datos:

1. **Cultura del modelado de datos (*data modeling*):** supone que los datos se generan por un modelo estocástico conocido (por ejemplo, $y = \beta_0 + \beta_1 x + \varepsilon$, con $\varepsilon \sim \mathcal{N}(0,\sigma^2)$) y estima sus parámetros. Breiman la describe así: "Estimated culture population. 98% of all statisticians" (población estimada: el **98 %** de todos los estadísticos).
2. **Cultura del modelado algorítmico (*algorithmic modeling*):** trata el mecanismo generador como desconocido y busca una función $f(x)$ que prediga bien $y$, validada por su **exactitud predictiva** en datos nuevos. Su estimación textual es "2% of statisticians, many in other fields" (el **2 %** de los estadísticos, muchos de ellos en otros campos); esta es la cultura que hoy asociamos con el ML.

Breiman sostuvo que la dependencia casi exclusiva de los modelos de datos "Led to irrelevant theory and questionable scientific conclusions" (condujo a teoría irrelevante y a conclusiones científicas cuestionables), y pidió adoptar "un conjunto más diverso de herramientas".

```{important}
Para un científico de datos la lección no es "la predicción gana a la inferencia". Es que **la pregunta define el método**. Si necesita estimar el efecto causal de un subsidio sobre la deserción escolar, un Random Forest con buena exactitud no responde esa pregunta. Si necesita predecir qué estudiantes tienen alto riesgo de desertar para intervenir a tiempo, un modelo lineal "interpretable" pero mal especificado puede ser la peor opción. Los comentarios publicados con el artículo (D. R. Cox, B. Efron, entre otros) y los de su aniversario veinte muestran que el debate sigue abierto.
```

## 1.4 Breve historia del aprendizaje automático

| Año | Hito | Por qué importa |
|---|---|---|
| 1950 | Turing, *Computing Machinery and Intelligence* (Mind, 59(236), 433–460) | Plantea si las máquinas pueden pensar, propone el "juego de imitación" y dedica una sección a las "máquinas que aprenden". |
| 1956 | Taller de Dartmouth (McCarthy, Minsky, Shannon, Rochester) | Se consolida el término "inteligencia artificial" como campo de investigación. |
| 1958 | Rosenblatt, perceptrón (*Psychological Review*, 65(6), 386–408) | Primer modelo de red neuronal entrenable especificado con precisión computacional. |
| 1959 | Samuel, programa de damas (*IBM J. Res. Dev.*, 3(3), 210–229) | Primer programa que mejora con la experiencia en un juego; populariza el término *machine learning*. |
| 1969 | Minsky y Papert, *Perceptrons* (MIT Press) | Demuestran los límites del perceptrón de una capa (p. ej., no puede representar XOR); contribuye al declive de la investigación en redes neuronales. |
| 1973 | Informe Lighthill (Reino Unido) | Evaluación crítica de la IA que llevó a recortes de financiación; se asocia al **primer invierno de la IA**. |
| 1986 | Rumelhart, Hinton y Williams, retropropagación (*Nature*, 323, 533–536) | Popularizan un método eficiente para entrenar redes multicapa, que aprenden representaciones internas útiles. |
| Finales de los 80 y comienzos de los 90 | Colapso del mercado de sistemas expertos | **Segundo invierno de la IA**: expectativas infladas, sistemas frágiles y costosos de mantener. |
| 1995 | Cortes y Vapnik, máquinas de soporte vectorial (*Machine Learning*, 20(3), 273–297) | Clasificadores de margen máximo con kernels y sólida base teórica (teoría estadística del aprendizaje). |
| 2001 | Breiman, Random Forest (*Machine Learning*, 45(1), 5–32) | Ensambles de árboles robustos y precisos; aún son una base fuerte para datos tabulares. |
| 2012 | Krizhevsky, Sutskever y Hinton, AlexNet (NIPS 25) | Una red convolucional profunda entrenada en GPU reduce drásticamente el error en ImageNet; arranca la era del *deep learning*. |
| 2016 | AlphaGo (Silver et al., *Nature*, 529, 484–489) | Combina aprendizaje profundo y por refuerzo para vencer a un campeón profesional de Go. |
| 2017 | Vaswani et al., *Attention Is All You Need* (NIPS 30) | Arquitectura Transformer basada solo en atención; base de los modelos de lenguaje actuales. |
| 2018–2019 | BERT (Devlin et al., 2019) | Preentrenamiento autosupervisado de lenguaje a gran escala y ajuste fino posterior. |
| 2020 | GPT-3 (Brown et al., 2020) | Muestra que un modelo de lenguaje muy grande puede resolver tareas con pocos ejemplos en el *prompt* (*few-shot*). |
| 2021 | Bommasani et al., "modelos fundacionales" | Acuñan el término para modelos entrenados con datos amplios, adaptables a muchas tareas. |
| 2022 en adelante | Asistentes conversacionales basados en LLM de uso masivo | El ML pasa a ser infraestructura cotidiana; crecen los debates de regulación, ética y evaluación. |

```{note}
**Matiz histórico sobre la retropropagación.** El artículo de 1986 popularizó el método en redes neuronales, pero la idea de derivar gradientes en cadena de forma eficiente tiene antecedentes anteriores (p. ej., la diferenciación automática en modo inverso y trabajos de los años setenta). Conviene decir "popularizaron" y no "inventaron".
```

Bommasani et al. (2021) definen un **modelo fundacional** como "any model that is trained on broad data (generally using self-supervision at scale) that can be adapted (e.g., fine-tuned) to a wide range of downstream tasks" ("cualquier modelo entrenado con datos amplios, generalmente mediante autosupervisión a escala, que puede adaptarse, p. ej. mediante ajuste fino, a una amplia gama de tareas posteriores"). Ponen como ejemplos BERT, GPT-3 y CLIP.

```{admonition} Lección de los inviernos de la IA
:class: warning
Los dos inviernos siguieron a ciclos de expectativas desmedidas. Cuando escuche promesas del tipo "la IA resolverá X en dos años", pida la terna E-T-P, la métrica en datos de prueba independientes y la comparación con una línea base simple.
```

## 1.5 Planteamiento formal del problema de aprendizaje

### 1.5.1 Elementos

Sea $\mathcal{X}$ el **espacio de entradas** (por ejemplo, $\mathbb{R}^d$, vectores de $d$ características) y $\mathcal{Y}$ el **espacio de salidas** ($\mathbb{R}$ en regresión, $\{1,\dots,K\}$ en clasificación). Suponemos que existe una distribución de probabilidad conjunta, **desconocida**, $P(\mathbf{x}, y)$ sobre $\mathcal{X}\times\mathcal{Y}$.

- **Función objetivo** $f: \mathcal{X}\to\mathcal{Y}$: la relación "verdadera" que quisiéramos conocer. En la práctica suele haber ruido, $y = f(\mathbf{x}) + \varepsilon$, por lo que es más preciso pensar en la distribución condicional $P(y\mid\mathbf{x})$.
- **Datos de entrenamiento**: una muestra $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$, que suponemos independiente e idénticamente distribuida (i.i.d.) según $P$.
- **Espacio de hipótesis** $\mathcal{H}$: el conjunto de funciones candidatas que el algoritmo puede producir (por ejemplo, todas las funciones lineales $h(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b$).
- **Hipótesis** $h \in \mathcal{H}$: un modelo concreto.
- **Algoritmo de aprendizaje** $\mathcal{A}$: el procedimiento que, dado $\mathcal{D}$, elige $\hat{h} = \mathcal{A}(\mathcal{D}) \in \mathcal{H}$.
- **Función de pérdida** $L(y, h(\mathbf{x}))$: cuantifica el costo de predecir $h(\mathbf{x})$ cuando la verdad es $y$.

Pérdidas típicas:

$$
L_{\text{cuadrática}}(y,\hat{y}) = (y-\hat{y})^2, \qquad
L_{0\text{-}1}(y,\hat{y}) = \mathbb{1}[y \neq \hat{y}], \qquad
L_{\text{log}}(y,\hat{p}) = -\big[y\log \hat{p} + (1-y)\log(1-\hat{p})\big].
$$

### 1.5.2 Riesgo esperado y riesgo empírico (Vapnik)

Lo que realmente queremos minimizar es el **riesgo esperado** (o error de generalización):

$$
R(h) = \mathbb{E}_{(\mathbf{x},y)\sim P}\big[L(y, h(\mathbf{x}))\big] = \int L(y,h(\mathbf{x}))\, dP(\mathbf{x},y).
$$

Como $P$ es desconocida, no podemos calcular $R(h)$. Lo aproximamos con el **riesgo empírico** sobre la muestra:

$$
\hat{R}_n(h) = \frac{1}{n}\sum_{i=1}^{n} L\big(y_i, h(\mathbf{x}_i)\big).
$$

El principio de **minimización del riesgo empírico** (ERM), formalizado por Vapnik en su teoría estadística del aprendizaje (Vapnik, 1995, 1998), consiste en elegir

$$
\hat{h} = \arg\min_{h\in\mathcal{H}} \hat{R}_n(h).
$$

La pregunta central de la teoría del aprendizaje es **cuándo un $\hat{R}_n(\hat{h})$ pequeño garantiza un $R(\hat{h})$ pequeño**. La respuesta depende de dos cosas: el tamaño de la muestra $n$ y la **capacidad** (complejidad) de $\mathcal{H}$. Goodfellow et al. (2016, cap. 5) lo resumen así: los resultados más importantes de esta teoría muestran que la brecha entre el error de entrenamiento y el de generalización está acotada por una cantidad que **crece** con la capacidad del modelo y **decrece** con el número de ejemplos.

En la práctica se añade un término de **regularización** que penaliza la complejidad:

$$
\hat{h} = \arg\min_{h\in\mathcal{H}} \; \hat{R}_n(h) + \lambda\,\Omega(h), \qquad \lambda \ge 0.
$$

### 1.5.3 Generalización, sobreajuste y subajuste

- **Generalización:** capacidad del modelo para mantener un buen desempeño en datos no vistos. Se estima con un conjunto de prueba independiente o con validación cruzada.
- **Subajuste (*underfitting*):** el modelo es demasiado simple para capturar la estructura de los datos; el error es alto tanto en entrenamiento como en prueba.
- **Sobreajuste (*overfitting*):** el modelo memoriza el ruido de la muestra; el error de entrenamiento es muy bajo pero el de prueba es alto.

### 1.5.4 Compromiso sesgo-varianza

Para pérdida cuadrática, con $y = f(\mathbf{x}) + \varepsilon$, $\mathbb{E}[\varepsilon]=0$ y $\operatorname{Var}(\varepsilon)=\sigma^2$, el error esperado en un punto $\mathbf{x}$, promediado sobre las posibles muestras de entrenamiento $\mathcal{D}$, se descompone así (Geman, Bienenstock y Doursat, 1992):

$$
\mathbb{E}_{\mathcal{D},\varepsilon}\Big[\big(y - \hat{h}_{\mathcal{D}}(\mathbf{x})\big)^2\Big]
= \underbrace{\sigma^2}_{\text{ruido irreducible}}
+ \underbrace{\Big(f(\mathbf{x}) - \mathbb{E}_{\mathcal{D}}\big[\hat{h}_{\mathcal{D}}(\mathbf{x})\big]\Big)^2}_{\text{sesgo}^2}
+ \underbrace{\mathbb{E}_{\mathcal{D}}\Big[\big(\hat{h}_{\mathcal{D}}(\mathbf{x}) - \mathbb{E}_{\mathcal{D}}[\hat{h}_{\mathcal{D}}(\mathbf{x})]\big)^2\Big]}_{\text{varianza}}.
$$

- **Sesgo:** error por suposiciones demasiado rígidas (por ejemplo, ajustar una recta a una relación curva).
- **Varianza:** sensibilidad del modelo a la muestra particular de entrenamiento.

Por lo general, aumentar la complejidad reduce el sesgo y aumenta la varianza. El mejor modelo equilibra ambos.

```{warning}
**Advertencia moderna.** La curva clásica en "U" del error de prueba frente a la complejidad no siempre se cumple en modelos muy sobreparametrizados, como las redes profundas. Belkin, Hsu, Ma y Mandal (2019) documentaron el fenómeno del **doble descenso**: pasado el punto de interpolación, el error de prueba puede volver a bajar. La descomposición sesgo-varianza sigue siendo válida como identidad matemática; lo que se matiza es la intuición de que "más parámetros siempre implica más varianza".
```

### 1.5.5 El teorema *No Free Lunch*

Wolpert (1996, *Neural Computation*, 8(7), 1341–1390) demostró que, medido por el error fuera del conjunto de entrenamiento y promediado uniformemente sobre todas las posibles funciones objetivo, **ningún algoritmo de aprendizaje es mejor que otro**. Según su resumen, para dos algoritmos cualesquiera A y B hay "tantos" problemas en los que A supera a B como al revés; esto incluye comparar la validación cruzada con la "anti-validación cruzada". Wolpert y Macready (1997, *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82) obtuvieron resultados análogos para optimización.

**Implicación práctica:** no existe un "mejor algoritmo universal". El éxito depende de que los **supuestos** (sesgo inductivo) del algoritmo coincidan con la estructura real del problema. Por eso siempre se comparan varios modelos en validación y se usa el conocimiento del dominio.

```{note}
El teorema *No Free Lunch* no dice que "todos los algoritmos sean igual de buenos en la práctica". Los problemas reales no se distribuyen uniformemente sobre todas las funciones posibles: tienen suavidad, estructura y regularidades. Lo que el teorema dice es que **esa ventaja proviene de supuestos**, no de una superioridad intrínseca.
```

## 1.6 Clasificación del aprendizaje automático

Este es el núcleo del capítulo. El eje principal es **el tipo de señal de supervisión** disponible en la experiencia E. Después veremos otros ejes de clasificación que no son excluyentes con este.

```{mermaid}
flowchart TD
    ML["Aprendizaje automático"] --> S["Por tipo de supervisión"]
    ML --> O["Otros ejes de clasificación"]
    S --> SUP["Supervisado"]
    S --> NOSUP["No supervisado"]
    S --> SEMI["Semisupervisado"]
    S --> AUTO["Autosupervisado"]
    S --> RL["Por refuerzo"]
    SUP --> REG["Regresión"]
    SUP --> CLA["Clasificación"]
    NOSUP --> CLU["Agrupamiento"]
    NOSUP --> RD["Reducción de dimensionalidad"]
    NOSUP --> ANO["Detección de anomalías"]
    NOSUP --> ASO["Reglas de asociación"]
    O --> B["Batch vs. online"]
    O --> IM["Basado en instancias vs. en modelos"]
    O --> PN["Paramétrico vs. no paramétrico"]
    O --> GD["Generativo vs. discriminativo"]
    O --> TL["Aprendizaje por transferencia"]
    O --> AL["Aprendizaje activo"]
    O --> FL["Aprendizaje federado"]
```

### 1.6.1 Aprendizaje supervisado

**Definición.** Se dispone de ejemplos **etiquetados** $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^n$ y el objetivo es aprender $h:\mathcal{X}\to\mathcal{Y}$ que prediga $y$ para entradas nuevas. Desde la perspectiva probabilística (Murphy, 2022), equivale a ajustar un modelo condicional $p(y\mid\mathbf{x})$.

- **Regresión:** $\mathcal{Y}\subseteq\mathbb{R}$. Ejemplos: predecir el precio de una vivienda, la demanda de energía o el rendimiento de un cultivo.
- **Clasificación:** $\mathcal{Y}$ es un conjunto finito de clases. Ejemplos: aprobar o no un crédito, detectar una transacción fraudulenta, diagnosticar una enfermedad en una imagen.

**Algoritmos representativos:** regresión lineal y regularizada (Ridge, Lasso), regresión logística, k vecinos más cercanos (k-NN), árboles de decisión, Random Forest, *gradient boosting* (XGBoost, LightGBM), máquinas de soporte vectorial, Naive Bayes y redes neuronales.

**Métricas habituales:** en regresión, MAE, RMSE y $R^2$; en clasificación, exactitud, precisión, *recall*, F1, AUC-ROC y matriz de confusión.

**Casos de uso en Colombia y Latinoamérica:** *scoring* de crédito y detección de fraude en banca y *fintech*; predicción de deserción escolar o universitaria para focalizar programas de permanencia; estimación de precios de vivienda; predicción de rendimiento de cultivos (café, palma, arroz) a partir de clima y suelo; clasificación de imágenes satelitales para monitorear deforestación en la Amazonía.

### 1.6.2 Aprendizaje no supervisado

**Definición.** Solo se dispone de entradas $\mathcal{D}=\{\mathbf{x}_i\}_{i=1}^n$, sin etiquetas. El objetivo es descubrir estructura: grupos, direcciones de variación, densidades o regularidades. Murphy (2012) lo llama enfoque "descriptivo" o de "descubrimiento de conocimiento" y advierte que es un problema "mucho menos definido", porque no hay un error obvio que medir. Desde la perspectiva probabilística equivale a modelar $p(\mathbf{x})$.

| Subtarea | Qué busca | Algoritmos representativos | Ejemplo aplicado |
|---|---|---|---|
| **Agrupamiento (*clustering*)** | Particionar los datos en grupos de elementos similares | K-means, jerárquico, DBSCAN, mezclas gaussianas | Segmentar clientes de una cooperativa o municipios por indicadores socioeconómicos |
| **Reducción de dimensionalidad** | Representar los datos con menos variables preservando información | PCA, t-SNE, UMAP, autoencoders | Resumir decenas de indicadores del DANE en pocos componentes |
| **Detección de anomalías** | Identificar observaciones atípicas | Isolation Forest, One-Class SVM, LOF | Alertas de fraude, fallas en sensores o contratación pública irregular |
| **Reglas de asociación** | Encontrar co-ocurrencias frecuentes | Apriori, FP-Growth | Análisis de canasta en tiendas de barrio o supermercados |

### 1.6.3 Aprendizaje semisupervisado

**Definición.** Se combina un conjunto **pequeño** de datos etiquetados $\mathcal{D}_L$ con uno **grande** sin etiquetar $\mathcal{D}_U$, porque etiquetar es costoso (requiere expertos, tiempo o dinero). La obra de referencia es Chapelle, Schölkopf y Zien (2006), *Semi-Supervised Learning* (MIT Press).

**Supuestos clave** (sin ellos, los datos no etiquetados no ayudan e incluso pueden perjudicar):

- **Suavidad:** puntos cercanos en regiones de alta densidad tienden a compartir etiqueta.
- **Clúster:** los datos forman grupos y los puntos de un mismo grupo tienden a tener la misma clase.
- **Variedad (*manifold*):** los datos de alta dimensión viven cerca de una variedad de menor dimensión.

**Algoritmos:** auto-entrenamiento (*self-training*), propagación de etiquetas (*label propagation*), co-entrenamiento y métodos basados en grafos.

**Ejemplo:** clasificar miles de radiografías cuando solo unas cientos fueron etiquetadas por radiólogos, o clasificar documentos jurídicos con pocos ejemplos anotados.

### 1.6.4 Aprendizaje autosupervisado

**Definición.** Las etiquetas **se generan automáticamente a partir de los propios datos** mediante una tarea pretexto: ocultar una palabra y predecirla, predecir el siguiente *token* o reconocer que dos versiones alteradas de una imagen provienen del mismo original. Técnicamente se entrena como un problema supervisado, pero sin anotación humana.

**Ejemplos emblemáticos:** BERT (Devlin et al., 2019) se preentrena prediciendo palabras enmascaradas; los modelos GPT, prediciendo el siguiente *token*. Como señalan Bommasani et al. (2021), la autosupervisión a escala es la base de los **modelos fundacionales**, que luego se adaptan con ajuste fino a tareas específicas.

```{note}
Hay quien clasifica el autosupervisado dentro del no supervisado (no hay etiquetas humanas) y quien lo trata como categoría propia (la formulación es supervisada). En este curso lo tratamos aparte porque su lógica, "los datos se supervisan a sí mismos", y su importancia actual lo justifican.
```

### 1.6.5 Aprendizaje por refuerzo

**Definición.** Un **agente** interactúa con un **entorno**: en cada paso $t$ observa un estado $s_t$, ejecuta una acción $a_t$, recibe una recompensa $r_{t+1}$ y pasa a $s_{t+1}$. No se le dice cuál era la acción correcta; debe descubrirla por ensayo y error. La referencia estándar es Sutton y Barto (2018), *Reinforcement Learning: An Introduction* (2.ª ed.).

**Formalización.** Usualmente se modela como un **proceso de decisión de Markov** (MDP) $(\mathcal{S},\mathcal{A},P,R,\gamma)$. El objetivo es encontrar una política $\pi(a\mid s)$ que maximice el retorno esperado descontado:

$$
\pi^* = \arg\max_{\pi}\; \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t\, r_{t+1}\right], \qquad 0\le\gamma<1.
$$

**Retos característicos:** el dilema **exploración-explotación**, la recompensa diferida (asignación de crédito) y la eficiencia de muestreo.

**Algoritmos:** Q-learning, SARSA, métodos de gradiente de política, actor-crítico, PPO.

**Ejemplos:** AlphaGo (Silver et al., 2016); control de robots; gestión de inventarios; control de semáforos; ajuste de modelos de lenguaje con retroalimentación humana (RLHF). El programa de damas de Samuel ya contenía, en germen, ideas de este tipo de aprendizaje.

### 1.6.6 Otras clasificaciones

**a) Por forma de aprendizaje: batch (offline) vs. online (incremental)** (Géron, 2022)

- **Batch:** el modelo se entrena con todos los datos disponibles de una vez. Para incorporar datos nuevos hay que reentrenarlo.
- **Online:** el modelo se actualiza de forma incremental con cada observación o mini-lote. Sirve para flujos de datos (transacciones, sensores) y datos que no caben en memoria. Su parámetro crítico es la **tasa de aprendizaje**: si es alta se adapta rápido pero "olvida"; si es baja es estable pero lenta. En scikit-learn, los estimadores con `partial_fit` (como `SGDClassifier`) permiten este modo.

**b) Por enfoque de generalización: basado en instancias vs. basado en modelos** (Géron, 2022)

- **Basado en instancias:** memoriza los ejemplos y predice por similitud (k-NN). Murphy (2012) lo llama aprendizaje basado en memoria o en instancias.
- **Basado en modelos:** construye un modelo con parámetros (regresión lineal, redes neuronales) y predice con él.

**c) Paramétrico vs. no paramétrico** (James et al., 2021)

- **Paramétrico:** supone una forma funcional con un número **fijo** de parámetros, independiente de $n$ (regresión lineal o logística). Es eficiente e interpretable, pero tiene riesgo de sesgo si la forma es incorrecta.
- **No paramétrico:** la complejidad **crece con los datos** (k-NN, árboles sin poda, métodos de kernel). Es flexible, pero requiere más datos y tiene riesgo de varianza.

**d) Generativo vs. discriminativo** (Ng y Jordan, 2002)

- **Generativo:** modela la distribución conjunta $p(\mathbf{x},y) = p(\mathbf{x}\mid y)\,p(y)$ y clasifica con la regla de Bayes (Naive Bayes, análisis discriminante, mezclas gaussianas). Puede generar datos nuevos.
- **Discriminativo:** modela directamente $p(y\mid\mathbf{x})$ o la frontera de decisión (regresión logística, SVM).

Ng y Jordan (2002) compararon regresión logística y Naive Bayes. Mostraron, contra la creencia extendida de que los discriminativos son casi siempre preferibles, que hay dos regímenes: el discriminativo tiene **menor error asintótico**, pero el generativo puede **acercarse a su propio error asintótico (más alto) mucho más rápido**. Con pocos datos, el generativo puede ganar.

**e) Aprendizaje por transferencia (*transfer learning*)** (Pan y Yang, 2010). Reutiliza el conocimiento de un modelo entrenado en una tarea o dominio de origen (por ejemplo, ImageNet o un corpus enorme de texto) para una tarea de destino con pocos datos. Es la estrategia dominante con modelos fundacionales: preentrenar y luego ajustar.

**f) Aprendizaje activo** (Settles, 2009). El algoritmo **elige qué ejemplos pedir que se etiqueten**, normalmente aquellos sobre los que tiene más incertidumbre, para maximizar el desempeño con un presupuesto limitado de anotación. Es útil cuando etiquetar exige expertos, como médicos o abogados.

**g) Aprendizaje federado** (McMahan et al., 2017). El modelo se entrena **sin centralizar los datos**: cada dispositivo o institución entrena localmente y solo comparte actualizaciones del modelo, que un servidor agrega (algoritmo *FederatedAveraging*). Es relevante cuando la privacidad o la regulación impiden mover los datos, por ejemplo entre hospitales o bancos. En Colombia, la Ley 1581 de 2012 de protección de datos personales hace atractivo este enfoque.

### 1.6.7 Tabla comparativa

| Tipo | Datos de entrada | Señal de aprendizaje | Objetivo típico | Algoritmos | Ejemplo |
|---|---|---|---|---|---|
| Supervisado | $(\mathbf{x}, y)$ etiquetados | Etiqueta correcta | Predecir $y$ | Regresión lineal/logística, árboles, RF, SVM, redes | Riesgo crediticio |
| No supervisado | Solo $\mathbf{x}$ | Ninguna explícita | Descubrir estructura | K-means, PCA, DBSCAN, Isolation Forest, Apriori | Segmentación de clientes |
| Semisupervisado | Pocos $(\mathbf{x},y)$ + muchos $\mathbf{x}$ | Etiquetas parciales + estructura | Predecir $y$ con menos anotación | Self-training, label propagation | Imágenes médicas con pocas anotaciones |
| Autosupervisado | Solo $\mathbf{x}$ (grandes volúmenes) | Etiquetas derivadas de los propios datos | Aprender representaciones | Modelos de lenguaje enmascarado o autorregresivo, aprendizaje contrastivo | Preentrenamiento de LLM |
| Por refuerzo | Interacción estado-acción | Recompensa (posiblemente diferida) | Política óptima | Q-learning, PPO, actor-crítico | Control de semáforos, juegos |

```{tip}
**Cómo identificar el tipo de problema en tres preguntas.**
1. ¿Tengo la respuesta correcta para cada ejemplo? Sí: supervisado. Solo para algunos: semisupervisado (o activo, si puedo pedir más).
2. ¿No tengo respuestas y quiero entender la estructura? No supervisado.
3. ¿El sistema toma decisiones secuenciales y recibe recompensas? Por refuerzo.
```

## 1.7 Flujo de trabajo de un proyecto de ML

### 1.7.1 CRISP-DM

El modelo de proceso **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*; Wirth y Hipp, 2000) sigue siendo el marco más difundido para organizar proyectos de datos. Tiene seis fases **iterativas**:

```{mermaid}
flowchart LR
    A["1. Comprensión<br/>del negocio"] --> B["2. Comprensión<br/>de los datos"]
    B --> A
    B --> C["3. Preparación<br/>de los datos"]
    C --> D["4. Modelado"]
    D --> C
    D --> E["5. Evaluación"]
    E --> A
    E --> F["6. Despliegue"]
```

1. **Comprensión del negocio:** objetivos, restricciones, criterio de éxito y terna E-T-P.
2. **Comprensión de los datos:** fuentes, calidad, análisis exploratorio.
3. **Preparación de los datos:** limpieza, imputación, codificación, ingeniería de características. Suele consumir la mayor parte del tiempo.
4. **Modelado:** selección de algoritmos, entrenamiento y ajuste de hiperparámetros.
5. **Evaluación:** ¿cumple el criterio de negocio, y no solo la métrica técnica?
6. **Despliegue:** puesta en producción. Las prácticas modernas de **MLOps** añaden monitoreo, detección de deriva de datos (*data drift*) y reentrenamiento.

### 1.7.2 Tipos de datos

- **Estructurados (tabulares):** filas como observaciones y columnas como variables, numéricas (continuas o discretas) o categóricas (nominales u ordinales).
- **No estructurados:** texto, imágenes, audio, video.
- **Semiestructurados:** JSON, XML, registros de eventos (*logs*).
- **Series de tiempo y datos espaciales:** tienen dependencia temporal o geográfica que **rompe el supuesto i.i.d.** y exige técnicas de validación específicas.

### 1.7.3 Conjuntos de entrenamiento, validación y prueba

- **Entrenamiento:** para ajustar los parámetros del modelo.
- **Validación:** para elegir hiperparámetros y comparar modelos (o se usa validación cruzada $k$-fold sobre el conjunto de entrenamiento).
- **Prueba:** se usa **una sola vez**, al final, para estimar el desempeño de generalización.

```{warning}
**Fuga de información (*data leakage*).** Es uno de los errores más comunes y graves. Ocurre cuando información del conjunto de prueba (o del futuro) se filtra al entrenamiento. Ejemplos: estandarizar con la media de **todo** el conjunto antes de dividirlo; incluir una variable que solo se conoce después del evento que se quiere predecir; dividir al azar datos temporales. Solución: dividir primero y luego ajustar las transformaciones solo con entrenamiento, idealmente dentro de un `Pipeline` de scikit-learn. En series de tiempo, use particiones cronológicas (`TimeSeriesSplit`).
```

## 1.8 Componente práctico con scikit-learn

Los ejemplos usan scikit-learn (Pedregosa et al., 2011). Fije siempre `random_state` para que los resultados sean reproducibles.

### 1.8.1 Preparación del entorno

```{code-cell} python
# Importaciones generales que usaremos en todo el capítulo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print("Versión de scikit-learn:", sklearn.__version__)
SEMILLA = 42  # semilla para reproducibilidad
```

### 1.8.2 Regresión: precio de viviendas en California

```{note}
`fetch_california_housing` descarga el conjunto la primera vez (requiere internet) y luego lo guarda en caché. Si trabaja sin conexión, puede sustituirlo por `load_diabetes`, que viene incluido en scikit-learn.
```

```{code-cell} python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar datos: X = características de cada distrito, y = valor mediano de la vivienda (en cientos de miles de USD)
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
print(X.shape)
print(X.head())

# 2. Dividir ANTES de cualquier transformación para evitar fuga de información
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEMILLA
)

# 3. Definir dos modelos: uno paramétrico (lineal) y uno no paramétrico (bosque aleatorio)
modelos = {
    "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEMILLA, n_jobs=-1),
}

# 4. Entrenar y evaluar en el conjunto de prueba
resultados = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados.append({
        "Modelo": nombre,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    })

pd.DataFrame(resultados).round(3)
```

```{code-cell} python
# Visualizar valores reales vs. predichos para el Random Forest
y_pred_rf = modelos["Random Forest"].predict(X_test)

plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.2, s=8)
plt.plot([0, 5], [0, 5], "r--", label="Predicción perfecta")
plt.xlabel("Valor real")
plt.ylabel("Valor predicho")
plt.title("California Housing: real vs. predicho (Random Forest)")
plt.legend()
plt.show()
```

**Interpretación.** Normalmente el Random Forest obtiene menor error que la regresión lineal porque captura relaciones no lineales e interacciones (por ejemplo, entre latitud y longitud). A cambio pierde interpretabilidad directa de coeficientes: es el dilema de las "dos culturas" de Breiman en miniatura.

### 1.8.3 Clasificación: diagnóstico de cáncer de mama

```{code-cell} python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1. Datos: 569 tumores, 30 características; clases 0 = maligno, 1 = benigno
datos = load_breast_cancer(as_frame=True)
X, y = datos.data, datos.target
print(y.value_counts().rename(index=dict(enumerate(datos.target_names))))

# 2. División estratificada: conserva la proporción de clases en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEMILLA
)

# 3. Tres clasificadores de familias distintas
clasificadores = {
    "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=4, random_state=SEMILLA),
}

# 4. Validación cruzada de 5 particiones SOLO sobre el conjunto de entrenamiento
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
for nombre, clf in clasificadores.items():
    puntajes = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
    print(f"{nombre:22s} F1 (CV) = {puntajes.mean():.3f} ± {puntajes.std():.3f}")
```

```{code-cell} python
# 5. Elegimos la regresión logística y la evaluamos UNA vez en prueba
mejor = clasificadores["Regresión logística"].fit(X_train, y_train)
y_pred = mejor.predict(X_test)
print(classification_report(y_test, y_pred, target_names=datos.target_names))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=datos.target_names, cmap="Blues"
)
plt.title("Matriz de confusión - Regresión logística")
plt.show()
```

```{important}
En diagnóstico médico **no todos los errores cuestan lo mismo**. Un falso negativo (decir "benigno" a un tumor maligno) es mucho más grave que un falso positivo. Por eso la exactitud global rara vez es la métrica adecuada; mire el *recall* de la clase "malignant" y considere ajustar el umbral de decisión. Esto es la "P" de Mitchell definida desde el dominio, no desde la comodidad técnica.
```

### 1.8.4 Sobreajuste y subajuste en acción

```{code-cell} python
from sklearn.preprocessing import PolynomialFeatures

# Datos sintéticos: y = sen(2*pi*x) + ruido
rng = np.random.default_rng(SEMILLA)
x = np.sort(rng.uniform(0, 1, 40))
y_sint = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, size=x.size)
x_tr, x_te, y_tr, y_te = train_test_split(x, y_sint, test_size=0.5, random_state=SEMILLA)

x_grid = np.linspace(0, 1, 300)
fig, ejes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for ax, grado in zip(ejes, [1, 4, 15]):
    modelo = make_pipeline(PolynomialFeatures(grado), LinearRegression())
    modelo.fit(x_tr.reshape(-1, 1), y_tr)
    ecm_tr = mean_squared_error(y_tr, modelo.predict(x_tr.reshape(-1, 1)))
    ecm_te = mean_squared_error(y_te, modelo.predict(x_te.reshape(-1, 1)))
    ax.scatter(x_tr, y_tr, label="Entrenamiento")
    ax.scatter(x_te, y_te, marker="x", label="Prueba")
    ax.plot(x_grid, modelo.predict(x_grid.reshape(-1, 1)), "k")
    ax.set_ylim(-2, 2)
    ax.set_title(f"Grado {grado}\nECM train={ecm_tr:.3f} | test={ecm_te:.3f}")
    ax.legend()

plt.suptitle("Subajuste (grado 1), buen ajuste (grado 4) y sobreajuste (grado 15)")
plt.tight_layout()
plt.show()
```

**Qué observar.** Con grado 1 ambos errores son altos (sesgo alto). Con grado 15 el error de entrenamiento es casi nulo y el de prueba se dispara (varianza alta). El grado intermedio equilibra ambos.

### 1.8.5 Agrupamiento con K-means

```{code-cell} python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

iris = load_iris(as_frame=True)
X_iris = StandardScaler().fit_transform(iris.data)  # K-means es sensible a la escala

# 1. ¿Cuántos grupos? Método del codo (inercia) y coeficiente de silueta
ks = range(2, 9)
inercias, siluetas = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X_iris)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_iris, km.labels_))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(ks, inercias, "o-"); a1.set_xlabel("k"); a1.set_ylabel("Inercia"); a1.set_title("Método del codo")
a2.plot(ks, siluetas, "o-"); a2.set_xlabel("k"); a2.set_ylabel("Silueta"); a2.set_title("Coeficiente de silueta")
plt.tight_layout(); plt.show()

# 2. Ajustar con k = 3 y comparar con las especies reales
#    (en un problema real NO tendríamos estas etiquetas; aquí solo sirven para ilustrar)
km3 = KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit(X_iris)
print("Índice de Rand ajustado vs. especies reales:",
      round(adjusted_rand_score(iris.target, km3.labels_), 3))
```

```{note}
El coeficiente de silueta suele favorecer $k=2$ en Iris, porque *versicolor* y *virginica* se solapan. Es una lección valiosa: **las métricas internas de agrupamiento no "descubren la verdad"**; reflejan la geometría de los datos bajo los supuestos del algoritmo (K-means asume grupos más o menos esféricos y de tamaño similar). La elección final de $k$ requiere criterio del dominio.
```

### 1.8.6 Reducción de dimensionalidad con PCA

```{code-cell} python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 1. PCA en Iris: de 4 a 2 dimensiones para visualizar
pca = PCA(n_components=2)
Z = pca.fit_transform(X_iris)
print("Varianza explicada por componente:", pca.explained_variance_ratio_.round(3))
print("Varianza explicada acumulada:", pca.explained_variance_ratio_.sum().round(3))

plt.figure(figsize=(6, 5))
for clase, nombre in enumerate(iris.target_names):
    plt.scatter(Z[iris.target == clase, 0], Z[iris.target == clase, 1], label=nombre, alpha=0.7)
plt.xlabel("CP1"); plt.ylabel("CP2"); plt.legend()
plt.title("Iris proyectado en los dos primeros componentes principales")
plt.show()

# 2. PCA en dígitos manuscritos (64 dimensiones): ¿cuántos componentes retienen el 95 % de la varianza?
X_dig = load_digits().data
pca95 = PCA(n_components=0.95).fit(X_dig)
print("Componentes necesarios para el 95 % de la varianza en digits:", pca95.n_components_)
```

**Interpretación.** PCA encuentra direcciones ortogonales de máxima varianza. En Iris, dos componentes conservan la gran mayoría de la variabilidad y separan visualmente *setosa* de las otras especies, aunque PCA **no usó las etiquetas**: es un método no supervisado.

## 1.9 Ética, sesgo algorítmico y limitaciones

Un modelo de ML aprende **los patrones de los datos que recibe**, incluidos los sesgos históricos y de medición. Dos casos documentados en revistas arbitradas lo muestran con claridad.

**Caso 1: Gender Shades (Buolamwini y Gebru, 2018).** Las autoras auditaron tres sistemas comerciales de clasificación de género a partir del rostro (de Microsoft, IBM y Face++). Encontraron que los conjuntos de referencia más usados estaban compuestos mayoritariamente por personas de piel clara (79,6 % en IJB-A y 86,2 % en Adience). Con un nuevo conjunto balanceado, las mujeres de piel oscura fueron el grupo peor clasificado: según la tabla 4 del artículo (PMLR 81:77–91), sus tasas de error iban de 20,8 % (Microsoft) a **34,7 %** (IBM), mientras que el error máximo para hombres de piel clara fue **0,8 %**.

**Caso 2: sesgo por variable proxy en salud (Obermeyer et al., 2019).** Un algoritmo comercial usado en Estados Unidos para decidir qué pacientes recibían programas de atención adicional mostró un sesgo racial importante: para un mismo puntaje de riesgo, los pacientes negros estaban más enfermos que los blancos. La causa no fue una variable de raza explícita, sino que el modelo **predecía costos de atención como proxy de necesidad de salud**, y por desigualdades de acceso se gasta menos en pacientes negros con la misma necesidad. Los autores estiman que corregir esta disparidad aumentaría la proporción de pacientes negros que reciben ayuda adicional de 17,7 % a 46,5 %.

```{important}
**Lecciones para el científico de datos.**
1. **Sesgo de representación:** si un grupo está poco representado en los datos, el modelo funcionará peor para ese grupo. Evalúe siempre el desempeño **desagregado** por subgrupos relevantes (sexo, etnia, región, estrato, zona urbana/rural).
2. **Sesgo de medición o de etiqueta:** la variable objetivo que elige (la "y") puede ser un proxy sesgado de lo que realmente le interesa.
3. **Retroalimentación:** los modelos desplegados cambian el mundo que luego generará los datos de reentrenamiento.
4. **Opacidad y rendición de cuentas:** documente datos y modelos, y explique sus límites a quienes toman decisiones.
5. **Privacidad:** cumpla la normativa de protección de datos (en Colombia, la Ley 1581 de 2012) y considere técnicas como la anonimización o el aprendizaje federado.
```

En Colombia, el documento **CONPES 4144**, aprobado el 14 de febrero de 2025, establece la **Política Nacional de Inteligencia Artificial** con horizonte a 2030. Tiene seis ejes estratégicos, entre ellos ética y gobernanza, datos e infraestructura, I+D+i y desarrollo de talento. El propio documento del DNP señala que las entidades responsables ejecutarán "106 acciones hasta 2030 con una inversión total aproximada de 479.273 millones de pesos", en una política de seis años (2024-2030). Al ser un documento de política pública, orienta y no crea sanciones por sí mismo, pero señala la dirección del país en materia de IA responsable. Para profundizar en fundamentos de equidad algorítmica, véanse Barocas, Hardt y Narayanan (2023) y Mehrabi et al. (2021).

**Limitaciones técnicas que conviene recordar:**

- **Correlación no es causalidad:** un modelo predictivo no identifica efectos causales sin supuestos y diseños adicionales.
- **Cambio de distribución:** si los datos de producción difieren de los de entrenamiento (nueva normativa, pandemia, inflación), el desempeño se degrada.
- **Calidad de datos:** ningún algoritmo compensa datos erróneos, incompletos o sesgados.
- **Interpretabilidad:** los modelos más precisos suelen ser los menos transparentes, lo que es crítico en crédito, salud o justicia.

## 1.10 Resumen

- El aprendizaje automático estudia algoritmos que mejoran su desempeño en una tarea a partir de la experiencia (Mitchell, 1997). La famosa frase atribuida a Samuel (1959) es una paráfrasis, no una cita textual.
- La IA es el campo amplio, el ML es un subcampo que aprende de datos y el aprendizaje profundo es un subconjunto del ML basado en redes profundas. La estadística aporta fundamentos y la ciencia de datos integra todo con el dominio.
- Breiman (2001) distinguió entre modelar el mecanismo generador y modelar algorítmicamente para predecir. La pregunta de investigación debe guiar la elección.
- El problema de aprendizaje busca minimizar el riesgo esperado, que es desconocido, a través del riesgo empírico, controlando la complejidad para generalizar.
- El compromiso sesgo-varianza explica el subajuste y el sobreajuste. El teorema *No Free Lunch* recuerda que no hay un algoritmo universalmente mejor.
- La taxonomía principal distingue aprendizaje supervisado, no supervisado, semisupervisado, autosupervisado y por refuerzo. Hay ejes complementarios: batch/online, instancias/modelos, paramétrico/no paramétrico, generativo/discriminativo, transferencia, activo y federado.
- CRISP-DM organiza el proyecto en seis fases iterativas. La separación rigurosa entre entrenamiento, validación y prueba evita la fuga de información.
- Los sesgos de los datos se convierten en sesgos del modelo; evaluar por subgrupos y cuestionar la variable objetivo es parte del trabajo técnico, no un añadido.

## 1.11 Ejercicios propuestos

1. **E-T-P.** Para tres problemas de su contexto laboral o regional (por ejemplo, predicción de deserción en su universidad, detección de fraude en subsidios o pronóstico de demanda de energía), especifique la tarea T, la medida P y la experiencia E. Indique a qué tipo de aprendizaje corresponde cada uno.
2. **Regresión.** En el ejemplo de California, agregue un modelo `Ridge` con distintos valores de `alpha` y un `GradientBoostingRegressor`. Compare con validación cruzada de 5 particiones y explique cuál elegiría y por qué.
3. **Umbral de decisión.** En el ejemplo de cáncer de mama, obtenga `predict_proba` y calcule el *recall* de la clase maligna para umbrales de 0,3, 0,5 y 0,7. Discuta el compromiso con la precisión.
4. **Sesgo-varianza.** Modifique el ejemplo polinomial para graficar el ECM de entrenamiento y de prueba contra el grado (1 a 15). Repita con 200 puntos en lugar de 40. ¿Qué cambia y por qué?
5. **Clustering.** Aplique K-means y `AgglomerativeClustering` al conjunto `load_wine`. Compare las siluetas y discuta si la estandarización cambia los resultados.
6. **PCA.** Con `load_digits`, grafique la varianza explicada acumulada contra el número de componentes y proyecte los datos en 2D. ¿Qué dígitos se confunden más?
7. **Datos abiertos de Colombia.** Descargue un conjunto del portal datos.gov.co (por ejemplo, resultados de las pruebas Saber 11 o registros de siniestralidad vial), formule un problema supervisado o no supervisado y describa las fases 1 a 3 de CRISP-DM aplicadas a él, incluyendo los riesgos de sesgo.
8. **Fuga de información.** Escriba un ejemplo en el que estandarizar antes de dividir los datos sobrestime el desempeño, y corríjalo con `Pipeline`.

## 1.12 Preguntas de reflexión y autoevaluación

1. ¿Por qué es incorrecto presentar "learn without being explicitly programmed" como cita textual de Samuel (1959)? ¿Qué frase real del artículo usaría en su lugar?
2. Explique con sus palabras la diferencia entre riesgo esperado y riesgo empírico. ¿Por qué no podemos minimizar directamente el primero?
3. Un modelo tiene 99 % de exactitud en entrenamiento y 70 % en prueba. ¿Qué diagnóstico hace y qué tres acciones propondría?
4. ¿Qué afirma el teorema *No Free Lunch* y qué **no** afirma?
5. ¿En qué se diferencian el aprendizaje semisupervisado y el autosupervisado? Dé un ejemplo de cada uno.
6. Según Ng y Jordan (2002), ¿en qué situación podría preferir Naive Bayes sobre la regresión logística?
7. Relacione el caso de Obermeyer et al. (2019) con la elección de la "P" y la "E" en la definición de Mitchell.
8. ¿Por qué la exactitud puede ser una métrica engañosa en un problema de detección de fraude con 0,5 % de transacciones fraudulentas?
9. Desde la perspectiva de Breiman (2001), ¿qué cultura de modelado usaría para evaluar el impacto de un programa social y cuál para focalizarlo? Justifique.
10. ¿Qué ventaja ofrece el aprendizaje federado frente al centralizado en un consorcio de hospitales colombianos? ¿Qué limitaciones tiene?

## 1.13 Glosario

| Término | Definición breve |
|---|---|
| **Algoritmo de aprendizaje** | Procedimiento que, a partir de datos, selecciona una hipótesis del espacio de hipótesis. |
| **Aprendizaje activo** | Enfoque en el que el algoritmo elige qué ejemplos deben etiquetarse. |
| **Aprendizaje federado** | Entrenamiento distribuido en el que los datos permanecen en sus fuentes y solo se comparten actualizaciones del modelo. |
| **Autosupervisión** | Aprendizaje con etiquetas generadas a partir de los propios datos mediante una tarea pretexto. |
| **Característica (*feature*)** | Variable de entrada que describe una observación. |
| **Espacio de hipótesis** | Conjunto de funciones que un algoritmo puede producir. |
| **Fuga de información** | Filtración de información de prueba o futura al proceso de entrenamiento. |
| **Función de pérdida** | Medida del costo de una predicción frente al valor real. |
| **Generalización** | Desempeño de un modelo en datos no vistos. |
| **Hiperparámetro** | Parámetro que se fija antes del entrenamiento (p. ej., $k$ en k-NN) y se ajusta con validación. |
| **Modelo fundacional** | Modelo entrenado con datos amplios, generalmente con autosupervisión a escala, adaptable a muchas tareas. |
| **Regularización** | Penalización de la complejidad del modelo para mejorar la generalización. |
| **Riesgo empírico** | Pérdida promedio sobre la muestra de entrenamiento. |
| **Riesgo esperado** | Pérdida esperada bajo la distribución verdadera de los datos. |
| **Sesgo (estadístico)** | Error sistemático por suposiciones demasiado rígidas del modelo. |
| **Sesgo algorítmico** | Desempeño o decisiones sistemáticamente desfavorables para ciertos grupos. |
| **Sobreajuste** | Modelo que memoriza el ruido y generaliza mal. |
| **Subajuste** | Modelo demasiado simple que no captura la estructura de los datos. |
| **Validación cruzada** | Técnica que rota particiones de los datos para estimar el desempeño de forma robusta. |
| **Varianza (del modelo)** | Sensibilidad de las predicciones a la muestra particular de entrenamiento. |

## Referencias bibliográficas

Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and machine learning: Limitations and opportunities*. MIT Press.

Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *Proceedings of the National Academy of Sciences, 116*(32), 15849–15854. https://doi.org/10.1073/pnas.1903070116

Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.

Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., Brynjolfsson, E., Buch, S., Card, D., Castellon, R., Chatterji, N., Chen, A., Creel, K., Davis, J. Q., Demszky, D., … Liang, P. (2021). *On the opportunities and risks of foundation models* (arXiv:2108.07258). arXiv. https://doi.org/10.48550/arXiv.2108.07258

Breiman, L. (2001a). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Breiman, L. (2001b). Statistical modeling: The two cultures (with comments and a rejoinder by the author). *Statistical Science, 16*(3), 199–231. https://doi.org/10.1214/ss/1009213726

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., … Amodei, D. (2020). Language models are few-shot learners. In *Advances in Neural Information Processing Systems 33* (pp. 1877–1901). Curran Associates.

Buolamwini, J., & Gebru, T. (2018). Gender shades: Intersectional accuracy disparities in commercial gender classification. In *Proceedings of the 1st Conference on Fairness, Accountability and Transparency* (Proceedings of Machine Learning Research, Vol. 81, pp. 77–91). PMLR.

Chapelle, O., Schölkopf, B., & Zien, A. (Eds.). (2006). *Semi-supervised learning*. MIT Press. https://doi.org/10.7551/mitpress/9780262033589.001.0001

Consejo Nacional de Política Económica y Social [CONPES]. (2025). *Documento CONPES 4144: Política Nacional de Inteligencia Artificial*. Departamento Nacional de Planeación.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423

Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural networks and the bias/variance dilemma. *Neural Computation, 4*(1), 1–58. https://doi.org/10.1162/neco.1992.4.1.1

Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems* (3rd ed.). O'Reilly Media.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning: With applications in R* (2nd ed.). Springer. https://doi.org/10.1007/978-1-0716-1418-1

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, & K. Q. Weinberger (Eds.), *Advances in Neural Information Processing Systems 25* (pp. 1097–1105). Curran Associates.

Lighthill, J. (1973). Artificial intelligence: A general survey. In *Artificial intelligence: A paper symposium* (pp. 1–21). Science Research Council.

McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. In A. Singh & J. Zhu (Eds.), *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics* (Vol. 54, pp. 1273–1282). PMLR.

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys, 54*(6), 1–35. https://doi.org/10.1145/3457607

Minsky, M., & Papert, S. (1969). *Perceptrons: An introduction to computational geometry*. MIT Press.

Mitchell, T. M. (1997). *Machine learning*. McGraw-Hill.

Murphy, K. P. (2012). *Machine learning: A probabilistic perspective*. MIT Press.

Murphy, K. P. (2022). *Probabilistic machine learning: An introduction*. MIT Press.

Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. In T. G. Dietterich, S. Becker, & Z. Ghahramani (Eds.), *Advances in Neural Information Processing Systems 14* (pp. 841–848). MIT Press.

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science, 366*(6464), 447–453. https://doi.org/10.1126/science.aax2342

Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering, 22*(10), 1345–1359. https://doi.org/10.1109/TKDE.2009.191

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408. https://doi.org/10.1037/h0042519

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533–536. https://doi.org/10.1038/323533a0

Samuel, A. L. (1959). Some studies in machine learning using the game of checkers. *IBM Journal of Research and Development, 3*(3), 210–229.

Settles, B. (2009). *Active learning literature survey* (Computer Sciences Technical Report 1648). University of Wisconsin–Madison.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Lillicrap, T., Leach, M., Kavukcuoglu, K., Graepel, T., & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature, 529*(7587), 484–489. https://doi.org/10.1038/nature16961

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Turing, A. M. (1950). Computing machinery and intelligence. *Mind, 59*(236), 433–460. https://doi.org/10.1093/mind/LIX.236.433

Vapnik, V. N. (1995). *The nature of statistical learning theory*. Springer. https://doi.org/10.1007/978-1-4757-2440-0

Vapnik, V. N. (1998). *Statistical learning theory*. Wiley.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30* (pp. 5998–6008). Curran Associates.

Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. In *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining* (pp. 29–39).

Wolpert, D. H. (1996). The lack of a priori distinctions between learning algorithms. *Neural Computation, 8*(7), 1341–1390. https://doi.org/10.1162/neco.1996.8.7.1341

Wolpert, D. H., & Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Transactions on Evolutionary Computation, 1*(1), 67–82. https://doi.org/10.1109/4235.585893
````

---

## Advertencias y puntos ciegos

- **Samuel (1959):** la frase "without being explicitly programmed" no está en el texto, y algunas fuentes (incluido un capítulo de Springer) citan por error el volumen 44(1.2), que corresponde a una reimpresión del año 2000. El capítulo cita el original, vol. 3(3), pp. 210–229. No incluí el DOI de este artículo porque no pude confirmarlo con una fuente primaria; puede añadirlo desde IEEE Xplore.
- **Mitchell (1997):** el texto de la definición es seguro (99 %). La página 2 no está confirmada en la fuente primaria; la indican fuentes secundarias como un capítulo de Springer (978-3-031-74227-9_2: "Mitchell, 1997, p. 2") y arXiv:1802.03768 ("[Mit97, p. 2]") (90 %), por eso el capítulo no indica número de página.
- **Detalles bibliográficos por debajo del 95 %:** para Wirth y Hipp (2000), la mayoría de las fuentes (un artículo de la Design Society publicado por Cambridge University Press, el trabajo CRISP-ML(Q) en arXiv, ScienceDirect y Springer) dan pp. 29–39 y Manchester, UK; las variantes 29–40 (un artículo de ScienceDirect) y "London" (una ficha de ResearchGate) son minoritarias. Las dos paginaciones de Vaswani et al. (2017) son reales: 5998–6008 en la edición de Curran Associates de *Advances in NIPS 30* y 6000–6010 en la edición de ACM DL (*NIPS'17 Proceedings*, DOI 10.5555/3295222.3295349). Para Géron uso la 3.ª ed. (2022), verificada en O'Reilly; la 2.ª ed. (2019) solo se pudo confirmar en un catálogo secundario.
- **Ejemplos latinoamericanos:** los casos colombianos de la sección 1.6 (crédito, deserción, cultivos, deforestación) son **escenarios de aplicación ilustrativos**, no estudios citados. Si quiere casos con evidencia publicada, conviene añadir referencias locales específicas en una revisión posterior.
- **CONPES 4144:** la fecha, los seis ejes, las 106 acciones y el presupuesto coinciden en varias fuentes secundarias y en el PDF del DNP, que afirma: "Estas entidades ejecutarán 106 acciones hasta 2030 con una inversión total aproximada de 479.273 millones de pesos". Antes de la versión final, cite el PDF oficial de SisCONPES/DNP.
- **Compatibilidad técnica:** `{mermaid}`, `{code-cell}` y las admonitions con `:class:` son sintaxis válida de MyST en Jupyter Book v2 (mystmd). Si su compilación no ejecuta código, las celdas se mostrarán sin resultados. El ejemplo de California necesita conexión a internet la primera vez.
- **Historia:** "inviernos de la IA" es una periodización convencional, no un hecho fechado con precisión. El capítulo la presenta como tal y aclara que Rumelhart et al. (1986) *popularizaron* la retropropagación, no la inventaron.