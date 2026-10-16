---
kernelspec:
  name: python3
  display_name: "Python 3"
---

# Módulo 2A. Los modelos de Machine Learning y cómo piensan

> **Subtítulo:** Qué es un modelo, y cómo "piensa" cada familia (supervisados, no supervisados y una mirada a las redes).

```{admonition} Cómo leer este módulo (importante para una audiencia de varias disciplinas)
:class: important
Este módulo está escrito para que lo siga **cualquier profesional**, venga de salud, administración, ingeniería, economía o derecho. Por eso usamos una regla:

- **Primero la intuición** (una analogía del mundo real).
- **Después el "cómo piensa" el modelo**, en palabras.
- **Al final, el código**, siempre con datasets pequeños que corren en cualquier portátil, sin GPU y sin internet.

Las fórmulas son **opcionales**. Cada vez que aparezca una, irá dentro de una caja titulada *"Para los curiosos"*. Si usted la salta, **no pierde nada** de lo esencial. No hay demostraciones matemáticas en este módulo: no las necesita para usar bien los modelos.
```

```{admonition} Objetivos de aprendizaje (parte 2A)
:class: note
Al terminar esta parte usted será capaz de:

1. Explicar con sus propias palabras qué es un "modelo" de machine learning y en qué se diferencia de un "algoritmo".
2. Describir, con una analogía, cómo funciona cada familia de modelos clásicos (lineales, vecinos, árboles, ensambles, SVM, Naive Bayes, agrupamiento, reducción de dimensión) y entender, por encima, qué es una red neuronal.
```
```{admonition} Conexión con los módulos anteriores
:class: tip
En el **Módulo 1** vimos *qué es* el aprendizaje automático y su **taxonomía** (supervisado, no supervisado, etc.). Aquí **no repetimos esa teoría**: la usamos como mapa. Si en algún momento no recuerda la diferencia entre supervisado y no supervisado, vuelva al Módulo 1. Este módulo es el **taller de modelos**: abrimos la caja de herramientas y vemos para qué sirve cada una.

Este Módulo 2 está dividido en tres partes que se leen en orden: **2A** (qué es cada modelo), **2B** (compararlos y conocer las librerías) y **2C** (practicar y ejercitar). La numeración de secciones (2.1, 2.2, …) es continua entre las tres partes.
```

---

## 2.1 Primero lo primero: ¿qué es "un modelo"?

Imagine que le entregan miles de recibos de nómina y le piden una "fórmula" para estimar el salario de un empleado nuevo a partir de su edad, sus años de experiencia y su cargo. Usted, a mano, observaría patrones: "a más experiencia, más salario"; "los cargos directivos pagan más". Al final tendría una especie de **regla** que convierte datos de entrada (edad, experiencia, cargo) en una salida (salario estimado).

Eso, exactamente, es un **modelo**: una regla aprendida de los datos que transforma **entradas** en una **salida**.

```{admonition} Dos palabras que la gente confunde: algoritmo vs. modelo
:class: important
- El **algoritmo** es la *receta de cocina*: el procedimiento paso a paso para aprender a partir de los datos (por ejemplo, "regresión lineal" o "árbol de decisión").
- El **modelo** es el *plato ya cocinado*: el resultado de aplicar esa receta a **sus** datos. Tiene valores concretos ya ajustados y sirve para hacer predicciones.

Una misma receta (algoritmo) produce **platos distintos** (modelos) según los ingredientes (datos). En scikit-learn esto es literal: usted escribe el algoritmo (`LinearRegression()`), le da datos con `.fit(...)` y obtiene un modelo entrenado que predice con `.predict(...)`.
```

Todos los modelos de este módulo, por diferentes que parezcan, hacen lo mismo en el fondo: **buscan patrones en los datos y los usan para decidir sobre datos nuevos**. Lo que cambia es *la forma* en que buscan esos patrones. Esa "forma de pensar" es lo que distingue a una familia de otra, y es lo que vamos a aprender a reconocer.

```{mermaid}
flowchart LR
    D["Datos de entrada\n(ejemplos del pasado)"] --> A["Algoritmo\n(la receta)"]
    A --> M["Modelo entrenado\n(el plato cocinado)"]
    N["Dato nuevo\n(nunca visto)"] --> M
    M --> P["Predicción\n(decisión o número)"]
```

---

## 2.2 Mapa de los modelos que veremos

Antes de entrar uno por uno, este es el mapa completo. Guárdelo: cada modelo del módulo ocupa un lugar aquí.

```{mermaid}
flowchart TD
    ML["Modelos de ML (clásicos)"] --> SUP["SUPERVISADOS\n(hay respuesta conocida)"]
    ML --> NSUP["NO SUPERVISADOS\n(no hay respuesta)"]
    ML --> RN["Introducción:\nRedes neuronales"]

    SUP --> LIN["Modelos lineales:\nregresión lineal y logística"]
    SUP --> VEC["Vecinos: k-NN"]
    SUP --> ARB["Árboles de decisión"]
    SUP --> ENS["Ensambles:\nRandom Forest y Boosting"]
    SUP --> SVM["Máquinas de soporte vectorial (SVM)"]
    SUP --> NB["Naive Bayes"]

    NSUP --> CLU["Agrupamiento:\nK-means, jerárquico, DBSCAN"]
    NSUP --> RD["Reducción de dimensión:\nPCA"]
```

```{tip}
**Regla de oro del módulo (y de la vida real como científico de datos):** empiece siempre por el modelo **más simple** que pueda resolver el problema. Un modelo lineal o un árbol pequeño que usted entiende y puede explicar vale más que un modelo complejísimo que nadie sabe por qué decide lo que decide. Lo complejo se justifica solo cuando lo simple se queda corto.
```

---

## 2.3 Los modelos supervisados, uno por uno

Recuerde del Módulo 1: **supervisado** significa que en los datos de entrenamiento ya conocemos la respuesta correcta (la "etiqueta"). El modelo aprende de esos ejemplos resueltos. Hay dos sabores:

- **Regresión:** la respuesta es un **número** (precio, temperatura, ventas).
- **Clasificación:** la respuesta es una **categoría** (spam/no spam, aprueba/no aprueba, benigno/maligno).

Veamos cada familia con su tarjeta: *idea en una frase → analogía → cómo piensa → cuándo sí y cuándo no → código mínimo*.

### 2.3.1 Regresión lineal

**Idea en una frase:** traza la "mejor línea recta" que pasa por una nube de puntos para predecir un número.

**Analogía:** usted mira un gráfico de "horas estudiadas" contra "nota obtenida" y, a ojo, dibuja la raya que mejor sigue la tendencia. La regresión lineal hace eso, pero con precisión matemática y con muchas variables a la vez.

**Cómo piensa:** supone que la salida es una **suma ponderada** de las entradas. Cada variable tiene un "peso" que dice cuánto empuja el resultado hacia arriba o hacia abajo. Aprender = encontrar los pesos que hacen que la línea quede lo más cerca posible de todos los puntos.

```{admonition} Para los curiosos (opcional)
:class: note
La forma del modelo es $\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$. Los $w_i$ son los pesos y $b$ es el punto de partida (intercepto). El algoritmo elige los pesos que minimizan el **error cuadrático** promedio, es decir, la distancia vertical al cuadrado entre la línea y los puntos reales.
```

**Cuándo sí:** cuando la relación entre entradas y salida es más o menos **recta**, y cuando necesita **interpretar** (los pesos le dicen qué variable importa y en qué dirección).

**Cuándo no:** cuando la relación es muy curva o hay interacciones complejas. Ahí la recta se queda corta (subajuste).

```python
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(X_train, y_train)      # aprende los pesos
y_pred = modelo.predict(X_test)   # predice números
```

### 2.3.2 Regresión logística (ojo: sirve para CLASIFICAR)

**Idea en una frase:** parecida a la lineal, pero en vez de un número devuelve una **probabilidad** de pertenecer a una clase.

**Analogía:** un médico estima "qué tan probable es que este paciente tenga la enfermedad" a partir de varios indicadores. No responde un número abierto, responde algo entre 0 % y 100 %, y con eso decide.

**Cómo piensa:** calcula una suma ponderada (igual que la lineal) y luego la "aplasta" con una curva en forma de **S** para que el resultado quede entre 0 y 1. Si la probabilidad supera un umbral (típicamente 0,5), asigna la clase.

```{admonition} Punto ciego muy común
:class: warning
El nombre engaña: la **regresión logística NO sirve para regresión**, sirve para **clasificación**. Es de los primeros modelos que todo el mundo usa para problemas de "sí/no" porque es rápido, interpretable y sorprendentemente competitivo.
```

**Cuándo sí:** clasificación binaria o multiclase donde quiere **probabilidades** e **interpretabilidad** (quién empuja la decisión).

**Cuándo no:** cuando la frontera entre clases es muy enredada y no lineal.

```python
from sklearn.linear_model import LogisticRegression
modelo = LogisticRegression(max_iter=5000)
modelo.fit(X_train, y_train)
y_pred  = modelo.predict(X_test)        # la clase
y_proba = modelo.predict_proba(X_test)  # la probabilidad de cada clase
```

### 2.3.3 k vecinos más cercanos (k-NN)

**Idea en una frase:** para decidir sobre un caso nuevo, mira los ejemplos más parecidos que ya conoce y copia su respuesta.

**Analogía:** usted llega a un barrio nuevo y quiere saber si una casa es cara o barata. Mira las **5 casas más cercanas** (k = 5) y concluye según ellas. "Dime con quién andas y te diré quién eres."

**Cómo piensa:** no "aprende" casi nada en el entrenamiento; simplemente **memoriza** todos los ejemplos. Cuando llega un dato nuevo, calcula la distancia a todos los demás, se queda con los *k* más cercanos y vota (clasificación) o promedia (regresión).

**Cuándo sí:** problemas pequeños, con pocas variables, donde "parecerse" tiene sentido geométrico. Excelente para enseñar la idea de similitud.

**Cuándo no:** con muchos datos o muchas variables se vuelve **lento** (compara contra todo) y se confunde (la noción de "cercanía" se diluye en muchas dimensiones). **Exige escalar** las variables (ver más adelante).

```python
from sklearn.neighbors import KNeighborsClassifier
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

### 2.3.4 Árbol de decisión

**Idea en una frase:** una cadena de preguntas de "sí/no" que va partiendo los datos hasta llegar a una decisión.

**Analogía:** un árbol de triaje en urgencias: "¿Tiene fiebre? Sí → ¿Más de 39°? Sí → ...". Cada pregunta divide a los pacientes en grupos más homogéneos hasta recomendar una acción.

**Cómo piensa:** busca, variable por variable, la **pregunta que mejor separa** los datos en grupos "más puros" (que dentro de cada grupo haya casi una sola clase). Repite el proceso en cada rama hasta que los grupos quedan lo bastante limpios.

**Cuándo sí:** cuando necesita **explicar** la decisión con reglas claras (muy valorado en negocio, salud y crédito), y cuando hay relaciones no lineales o por tramos.

**Cuándo no:** un árbol solo, sin control, **memoriza** (sobreajusta) con facilidad. Por eso casi nunca se usa solo en producción, sino dentro de un ensamble (lo siguiente).

```python
from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(max_depth=4, random_state=42)  # max_depth evita que memorice
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

### 2.3.5 Random Forest (bosque aleatorio) — el primer "ensamble"

**Idea en una frase:** en vez de un árbol, construye **cientos de árboles distintos** y promedia sus votos.

**Analogía:** no le pregunta a **un** experto, sino a **un jurado** de cien expertos entrenados cada uno con una muestra y un enfoque ligeramente distinto. El consenso del jurado casi siempre es más estable y acertado que la opinión de uno solo. Esto se llama *sabiduría de las multitudes*.

**Cómo piensa:** entrena muchos árboles, cada uno con una muestra aleatoria de los datos y un subconjunto aleatorio de las variables (por eso "aleatorio"). Al final, promedia (regresión) o hace votación por mayoría (clasificación). Esa aleatoriedad hace que los errores individuales se cancelen entre sí.

**Cuándo sí:** es la **opción por defecto** para datos tabulares. Funciona bien "de fábrica", resiste el sobreajuste mucho mejor que un árbol solo y no exige escalar variables.

**Cuándo no:** cuando necesita una explicación de **una sola regla** simple (un bosque es más difícil de interpretar que un árbol), o cuando el tiempo de predicción debe ser mínimo.

```python
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

### 2.3.6 Gradient Boosting (y sus estrellas: XGBoost y LightGBM)

**Idea en una frase:** construye árboles **en serie**, donde cada árbol nuevo se dedica a corregir los errores del anterior.

**Analogía:** un estudiante que repasa para un examen. Hace un simulacro, identifica **en qué falló**, estudia justo esos temas, vuelve a intentar, corrige de nuevo... En cada ronda se concentra en lo que todavía no domina. Al final, la suma de muchas correcciones pequeñas da un resultado muy afinado.

**Cómo piensa:** a diferencia del Random Forest (árboles **en paralelo**, independientes), el boosting es **secuencial**: cada árbol mira los errores que dejó el conjunto anterior y trata de arreglarlos. Es más preciso, pero también más fácil de sobreajustar si no se ajusta con cuidado.

```{admonition} ¿Y XGBoost / LightGBM qué son?
:class: note
Son **implementaciones muy optimizadas** del gradient boosting, no modelos "nuevos" conceptualmente:
- **XGBoost** (Chen y Guestrin, 2016): el que popularizó la técnica y arrasó en competencias de Kaggle.
- **LightGBM** (Ke et al., 2017): más rápido y liviano en memoria, pensado para datos más grandes.

Ambos son hoy, junto con Random Forest, lo que más gana en problemas con **datos tabulares**. Vienen en librerías aparte (`pip install xgboost lightgbm`), no dentro de scikit-learn.
```

**Cuándo sí:** cuando quiere exprimir la **máxima precisión** en datos tabulares y está dispuesto a ajustar algunos parámetros.

**Cuándo no:** si busca algo rápido de montar y fácil de explicar, o si no tiene tiempo de ajustar hiperparámetros (un Random Forest da el 90 % del resultado con la mitad del esfuerzo).

```python
# Requiere: pip install xgboost
from xgboost import XGBClassifier
modelo = XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

### 2.3.7 Máquinas de soporte vectorial (SVM)

**Idea en una frase:** busca la frontera que separa las clases dejando **el mayor espacio posible** a cada lado.

**Analogía:** imagine dos barrios y quiere trazar una **avenida** que los divida. No cualquier línea: la **avenida más ancha** que quepa entre las casas de ambos barrios, de modo que ninguna casa quede pegada al borde. Esa "calle ancha" (el *margen*) hace la separación más robusta.

**Cómo piensa:** encuentra la frontera de máxima separación. Cuando las clases no se pueden separar con una línea recta, usa un truco (llamado *kernel*) que, en términos simples, **dobla el espacio** para que lo que estaba enredado quede separable.

**Cuándo sí:** problemas con muchas variables y pocos datos (texto, bioinformática), donde una frontera clara y robusta ayuda.

**Cuándo no:** con muchos datos (cientos de miles de filas) se vuelve **lento**. Además, **exige escalar** las variables y no entrega probabilidades de forma natural.

```python
from sklearn.svm import SVC
modelo = SVC(kernel="rbf", C=1.0)   # casi siempre va con escalado previo
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

### 2.3.8 Naive Bayes

**Idea en una frase:** calcula probabilidades usando el teorema de Bayes y una suposición "ingenua" (*naive*): que las variables son independientes entre sí.

**Analogía:** un filtro de spam que ve la palabra "premio", la palabra "gratis" y tres signos de admiración, y multiplica las "sospechas" de cada pista como si fueran independientes, aunque no lo sean del todo. La suposición es falsa, pero funciona asombrosamente bien en la práctica.

**Cómo piensa:** para cada clase, estima qué tan probable es ver esas características, y elige la clase más probable. Es rapidísimo porque solo cuenta frecuencias.

**Cuándo sí:** clasificación de **texto** (spam, análisis de sentimiento), como línea base veloz, y cuando tiene muchísimas variables.

**Cuándo no:** cuando las variables están **fuertemente correlacionadas** y esa dependencia importa mucho; ahí la suposición "ingenua" le pasa factura.

```python
from sklearn.naive_bayes import GaussianNB
modelo = GaussianNB()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

---

## 2.4 Los modelos no supervisados

Aquí **no hay respuesta conocida**. No le decimos al modelo qué es correcto; le pedimos que **descubra estructura** por su cuenta. Dos tareas clave: **agrupar** (clustering) y **resumir variables** (reducción de dimensión).

### 2.4.1 K-means (agrupamiento)

**Idea en una frase:** separa los datos en *k* grupos juntando lo que se parece.

**Analogía:** usted tiene una base de clientes y quiere formar **3 segmentos** para campañas distintas, sin saber de antemano cuáles son. K-means los agrupa por similitud (consumo, frecuencia, edad...) y le entrega 3 grupos con "perfiles" diferenciados.

**Cómo piensa:** usted elige cuántos grupos quiere (*k*). El algoritmo coloca *k* "centros", asigna cada punto al centro más cercano, recalcula los centros como el promedio de su grupo, y repite hasta que se estabiliza.

```{admonition} El talón de Aquiles: ¿cuántos grupos?
:class: warning
Usted **debe decidir el número de grupos *k*** antes de empezar, y rara vez lo sabe. Hay ayudas (el "método del codo", el "coeficiente de silueta"), pero **ninguna descubre la verdad**: son pistas. La decisión final la toma el conocimiento del negocio. Además, K-means asume grupos más o menos redondos y de tamaño parecido; si sus grupos tienen formas raras, fallará.
```

```python
from sklearn.cluster import KMeans
modelo = KMeans(n_clusters=3, n_init=10, random_state=42)
grupos = modelo.fit_predict(X)   # a qué grupo va cada punto
```

### 2.4.2 Agrupamiento jerárquico

**Idea en una frase:** arma un árbol de grupos, uniendo poco a poco lo más parecido.

**Analogía:** un árbol genealógico al revés. Empieza con cada punto solo, une los dos más cercanos, luego los siguientes, y así hasta formar un gran grupo. El resultado es un *dendrograma* (diagrama de árbol) que usted "corta" a la altura que quiera para obtener más o menos grupos.

**Cuándo sí:** cuando quiere **ver la estructura jerárquica** y decidir el número de grupos *después*, mirando el dendrograma. Bueno para datos pequeños.

**Cuándo no:** con muchos datos es costoso en memoria y tiempo.

```python
from sklearn.cluster import AgglomerativeClustering
modelo = AgglomerativeClustering(n_clusters=3)
grupos = modelo.fit_predict(X)
```

### 2.4.3 DBSCAN

**Idea en una frase:** agrupa por **densidad**: los grupos son zonas donde hay muchos puntos juntos, y lo que queda aislado se marca como "ruido".

**Analogía:** mire un mapa nocturno de una ciudad desde un avión. Las zonas muy iluminadas son los grupos (barrios); las luces sueltas en el campo son **anomalías**. DBSCAN piensa así.

**Cuándo sí:** cuando no sabe cuántos grupos hay, los grupos tienen **formas irregulares**, o quiere **detectar atípicos** automáticamente. No le pide el número de grupos.

**Cuándo no:** cuando los grupos tienen densidades muy distintas, o en muchas dimensiones (le cuesta definir "cercanía").

```python
from sklearn.cluster import DBSCAN
modelo = DBSCAN(eps=0.5, min_samples=5)
grupos = modelo.fit_predict(X)   # el grupo -1 significa "ruido/atípico"
```

### 2.4.4 PCA (análisis de componentes principales) — reducción de dimensión

**Idea en una frase:** resume muchas variables en unas pocas, perdiendo la menor información posible.

**Analogía:** usted tiene un objeto 3D y quiere tomarle **una foto** (2D) que lo represente lo mejor posible. Hay un ángulo que muestra "casi todo" y otro que lo aplasta y oculta información. PCA busca ese **mejor ángulo**: las direcciones donde los datos más varían.

**Cómo piensa:** encuentra nuevas "variables artificiales" (componentes) que son combinaciones de las originales y que capturan la mayor variación de los datos. Las primeras 2 o 3 suelen resumir gran parte de todo.

**Cuándo sí:** para **visualizar** datos de muchas variables en 2D, para **acelerar** otros modelos, o para quitar redundancia. Es no supervisado: **no usa la etiqueta**.

**Cuándo no:** cuando necesita interpretar variables **originales** (los componentes son mezclas y pierden significado directo), o cuando las relaciones importantes no son de "varianza".

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reducida = pca.fit_transform(X)                 # ahora X tiene solo 2 columnas
print(pca.explained_variance_ratio_.sum())        # cuánta información conservó
```

---

## 2.5 Un vistazo (ligero) a las redes neuronales

No podemos cerrar un recorrido de modelos sin nombrar a las **redes neuronales**, el motor del *deep learning* y de la IA actual (incluidos los modelos de lenguaje). Aquí solo la **idea**; la profundidad queda para un módulo dedicado, porque entrenar redes grandes sí necesita hardware potente, y la regla de este curso es que todo corra en su portátil.

**Idea en una frase:** son muchas "mini-decisiones" (neuronas) conectadas en capas, que juntas aprenden patrones muy complejos.

**Analogía:** una línea de producción con varias estaciones. La primera estación detecta cosas simples (bordes, en una imagen); la siguiente combina eso en formas; la siguiente, en objetos. Cada capa construye sobre la anterior, pasando de lo simple a lo abstracto.

**Cómo piensa:** cada neurona hace una pequeña suma ponderada (como una mini regresión) y decide si "se activa". Apilando capas de neuronas, la red aprende a representar relaciones que un modelo lineal jamás capturaría. Aprende ajustando todos esos pesos para reducir el error, con un método que, en intuición, es "bajar una montaña en la niebla dando pasos hacia donde el terreno baja" (descenso del gradiente).

```{admonition} ¿Cuándo brilla y cuándo es exagerado?
:class: tip
Las redes neuronales dominan en **datos no estructurados**: imágenes, audio, texto, video. Pero en **datos tabulares** (tablas de Excel, bases de clientes), los ensambles de árboles (Random Forest, XGBoost, LightGBM) **siguen siendo iguales o mejores**, más rápidos y más fáciles de explicar. No use un cañón para matar una mosca: para su tabla de clientes, empiece por un árbol o un bosque, no por una red.
```

Para que vea que una red pequeña **también corre en su PC**, scikit-learn trae una (`MLPClassifier`) que entrena en segundos con datos pequeños. La usamos en los ejemplos desarrollados.

---

```{admonition} Sigue en el Módulo 2B
:class: tip
Ya sabe qué hace cada modelo. En el **Módulo 2B** los ponemos lado a lado (ventajas y desventajas), aprendemos a elegir y vemos con qué librería de Python se programa cada cosa.
```

## Referencias (Módulo 2A)

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (pp. 3146–3154). Curran Associates.
