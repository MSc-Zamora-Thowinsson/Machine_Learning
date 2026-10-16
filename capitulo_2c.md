---
kernelspec:
  name: python3
  display_name: "Python 3"
---

# Módulo 2C. Práctica guiada, ejercicios y evaluación

> **Subtítulo:** Ejemplos que corren tal cual, plantillas para completar y ejercicios para Colab y VS Code.

```{admonition} Antes de empezar
:class: important
Esta parte cierra el **Módulo 3** (viene de 2A y 2B). Para ejecutar los ejemplos necesita el entorno preparado en el **Módulo 3B, sección 3.8** (Colab o VS Code con scikit-learn, pandas, NumPy y Matplotlib). Todos los ejemplos usan datasets diminutos integrados en scikit-learn: corren en segundos, sin GPU y sin internet.
```

```{admonition} Objetivos de aprendizaje (parte 2C)
:class: note
Al terminar esta parte usted será capaz de:

6. Ejecutar los ejemplos desarrollados y completar los ejercicios "por desarrollar".
7. Resolver ejercicios prácticos de modelado tanto en **Google Colab** como en **VS Code**, y leer críticamente sus resultados.
```

---

## 3.9 Ejemplos DESARROLLADOS (listos para correr en clase)

Estos ejemplos **funcionan tal cual**, con datasets integrados (sin internet) y son pequeños (corren en segundos en cualquier portátil). Úselos para demostrar en vivo.

### Ejemplo 1 — Clasificación: comparar varios modelos en un mismo problema

Objetivo didáctico: mostrar que distintos modelos dan resultados distintos y que **compararlos de forma justa** es parte del trabajo.

```{code-cell} python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

SEMILLA = 42

# 1) Datos: 569 tumores, 30 características; 0 = maligno, 1 = benigno
datos = load_breast_cancer(as_frame=True)
X, y = datos.data, datos.target

# 2) Separar entrenamiento y prueba, conservando la proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEMILLA
)

# 3) Definir los modelos. Los que necesitan escalado van dentro de un Pipeline
#    (StandardScaler + modelo). Así el escalado se aprende SOLO con entrenamiento.
modelos = {
    "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "k-NN (k=5)":          make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Árbol (prof.=4)":     DecisionTreeClassifier(max_depth=4, random_state=SEMILLA),
    "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
    "SVM (RBF)":           make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "Naive Bayes":         GaussianNB(),
}

# 4) Comparar con validación cruzada (5 particiones) SOLO en entrenamiento
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
filas = []
for nombre, modelo in modelos.items():
    puntajes = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1")
    filas.append({"Modelo": nombre, "F1 medio": puntajes.mean(), "Desv.": puntajes.std()})

tabla = pd.DataFrame(filas).sort_values("F1 medio", ascending=False).round(3)
tabla
```

```{code-cell} python
# 5) Tomamos el mejor según la tabla y lo evaluamos UNA sola vez en prueba
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

mejor_nombre = tabla.iloc[0]["Modelo"]
mejor_modelo = modelos[mejor_nombre].fit(X_train, y_train)
y_pred = mejor_modelo.predict(X_test)

print("Modelo ganador:", mejor_nombre)
print(classification_report(y_test, y_pred, target_names=datos.target_names))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=datos.target_names, cmap="Blues")
plt.title(f"Matriz de confusión — {mejor_nombre}")
plt.show()
```

```{admonition} Qué debe observar el estudiante
:class: important
1. Varios modelos quedan muy cerca: no siempre gana el más complejo.
2. Usamos **F1**, no solo exactitud, porque en diagnóstico un **falso negativo** (decir "benigno" a un tumor maligno) es gravísimo. La métrica se elige según el problema, no por comodidad.
3. El conjunto de **prueba se tocó una sola vez**, al final. Si lo usa para elegir el modelo, se engaña a sí mismo.
```

### Ejemplo 2 — Regresión: lineal vs. Random Forest

```{code-cell} python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

SEMILLA = 42
X, y = load_diabetes(return_X_y=True, as_frame=True)  # progresión de diabetes (número)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)

modelos = {
    "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest":    RandomForestRegressor(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
}
for nombre, m in modelos.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    print(f"{nombre:18s} | MAE = {mean_absolute_error(y_te, pred):6.2f} | R2 = {r2_score(y_te, pred):.3f}")
```

```{admonition} Lectura del resultado
:class: note
Aquí la **regresión lineal suele empatar o ganarle** al Random Forest, porque este dataset es pequeño y bastante lineal. Es la prueba viva de la regla de oro: **lo complejo no siempre gana**. Primero lo simple.
```

### Ejemplo 3 — No supervisado: K-means + PCA para visualizar

```{code-cell} python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

SEMILLA = 42
vino = load_wine()
X = StandardScaler().fit_transform(vino.data)   # K-means y PCA exigen escalado

# 1) Agrupar en 3 (sabemos que hay 3 tipos de vino, pero K-means NO lo sabe)
km = KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit(X)
print("Coeficiente de silueta:", round(silhouette_score(X, km.labels_), 3))

# 2) Reducir a 2 dimensiones SOLO para poder dibujarlo
Z = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(6, 5))
plt.scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="viridis", alpha=0.8)
plt.xlabel("Componente principal 1"); plt.ylabel("Componente principal 2")
plt.title("Grupos de K-means proyectados con PCA")
plt.show()
```

### Ejemplo 4 — Una red neuronal pequeña que SÍ corre en su portátil

```{code-cell} python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

SEMILLA = 42
# 1797 imágenes de dígitos 8x8 (diminutas): perfectas para una red pequeña en CPU
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)

red = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=SEMILLA)
)
red.fit(X_tr, y_tr)
print("Exactitud de la red:", round(accuracy_score(y_te, red.predict(X_te)), 3))
```

```{admonition} El puente conceptual
:class: tip
Esta red tiene **una capa oculta de 64 neuronas** y entrena en segundos. Es el mismo principio de los modelos gigantes de hoy, solo que a escala de juguete. Así el estudiante "toca" una red neuronal sin necesitar GPU.
```

---

## 3.10 Ejemplos POR DESARROLLAR (esqueletos para el estudiante)

Estos **no están completos a propósito**. El estudiante rellena los `# TODO`. Son plantillas seguras (plain Python, no se ejecutan al construir el libro).

### Plantilla A — Comparar modelos en un dataset propio

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
# TODO 1: importar los 3 modelos que quiera comparar

# TODO 2: cargar sus datos (CSV) y separar X (variables) e y (objetivo)
# df = pd.read_csv("mis_datos.csv")
# X = df.drop(columns=["objetivo"])
# y = df["objetivo"]

# TODO 3: separar entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO 4: entrenar cada modelo y comparar con cross_val_score
# TODO 5: elegir el mejor y evaluarlo UNA vez en prueba
# TODO 6: escribir en 3 líneas qué modelo eligió y por qué
```

### Plantilla B — Segmentación de clientes con K-means

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# TODO 1: cargar datos de clientes (p. ej., edad, ingreso, frecuencia de compra)
# TODO 2: ESCALAR las variables (K-means lo exige)
# TODO 3: probar k = 2, 3, 4, 5 y calcular la silueta de cada uno
# TODO 4: elegir k mirando la silueta Y el sentido de negocio
# TODO 5: describir el "perfil" de cada grupo (promedios por grupo)
```

---

## 3.11 Ejercicios prácticos (Colab y VS Code)

Cada ejercicio indica dónde hacerlo. Los marcados **"sin internet"** corren con datasets de scikit-learn; los de **datos abiertos de Colombia** requieren descargar un archivo (hágalos en casa, no en clase en vivo).

```{admonition} Formato de entrega sugerido
:class: note
Entregue un notebook (`.ipynb`) con: (1) una celda de chequeo de versiones, (2) el código comentado, (3) una celda de **conclusiones en texto** (markdown) de máximo 5 líneas. El razonamiento vale tanto como el código.
```

**Ejercicio 1 — Baseline obligatorio (Colab o VS Code, sin internet).**
Con `load_wine`, entrene una regresión logística como línea base y luego un Random Forest. Compare F1 con validación cruzada. ¿Justifica el bosque su mayor complejidad? Responda con números, no con intuición.

**Ejercicio 2 — El efecto de NO escalar (Colab o VS Code, sin internet).**
Con `load_breast_cancer`, entrene un k-NN **sin** `StandardScaler` y **con** `StandardScaler`. Reporte la diferencia de F1 y explique por qué el escalado cambia tanto el resultado de k-NN pero casi nada el de un árbol.

**Ejercicio 3 — Sobreajuste visible (VS Code recomendado, sin internet).**
Con `load_digits`, entrene árboles de decisión con `max_depth` = 1, 3, 5, 10, 20 y None. Grafique la exactitud en **entrenamiento** vs. **prueba** contra la profundidad. Marque en qué punto empieza el sobreajuste.

**Ejercicio 4 — Elegir *k* en K-means (Colab o VS Code, sin internet).**
Con `load_iris`, calcule la silueta para *k* de 2 a 8 y grafíquela. ¿Coincide el mejor *k* con las 3 especies reales? Explique por qué puede no coincidir (pista: dos especies se solapan).

**Ejercicio 5 — Boosting en acción (Colab recomendado; requiere `pip install xgboost`).**
Sobre el Ejercicio 1, agregue un `XGBClassifier`. Compárelo con el Random Forest. ¿Ganó? ¿Por cuánto? ¿Vale la pena la complejidad extra para este dataset pequeño?

**Ejercicio 6 — Datos abiertos de Colombia (VS Code, requiere internet/descarga).**
Descargue un conjunto de `datos.gov.co` o del **DANE** (por ejemplo, resultados **Saber 11** del ICFES, o indicadores socioeconómicos municipales). Formule **un** problema supervisado (p. ej., predecir el puntaje global) o uno no supervisado (agrupar municipios). Documente las fases 1 a 3 de CRISP-DM (Módulo 1) y señale **un riesgo de sesgo** concreto de esos datos.

```{admonition} Reto de pensamiento crítico (sin código)
:class: important
Un banco quiere un modelo para aprobar créditos. Su equipo logra 92 % de exactitud con XGBoost, pero el modelo es una caja negra. La competencia usa una regresión logística con 89 % de exactitud, totalmente explicable. **¿Cuál recomienda y por qué?** No hay una única respuesta correcta; lo que se evalúa es que considere precisión, interpretabilidad, regulación (en Colombia, la **Ley 1581 de 2012** de datos personales y la **Circular Externa de la SFC** sobre modelos) y el costo de un error.
```

---

## 3.12 Errores comunes (y cómo evitarlos)

```{admonition} Los cinco tropiezos que más veremos en clase
:class: warning
1. **No escalar cuando toca.** k-NN, SVM, K-means, PCA y redes **exigen** `StandardScaler`. Árboles y bosques, no. Olvidarlo arruina el resultado en silencio.
2. **Fuga de información (*data leakage*).** Escalar o imputar usando *todos* los datos antes de dividir. Solución: hágalo dentro de un `Pipeline`, que aprende las transformaciones solo con el entrenamiento.
3. **Usar la exactitud con clases desbalanceadas.** Si el 99 % de las transacciones son legítimas, un modelo que dice "todo legítimo" acierta 99 % y es inútil. Use precisión, recall, F1 o AUC.
4. **Mirar el conjunto de prueba muchas veces.** Si lo usa para decidir, deja de ser una estimación honesta de generalización. Úselo **una sola vez**, al final.
5. **Saltar directo al modelo complejo.** Sin una línea base simple, usted no sabe si su XGBoost realmente aporta o si un modelo de 3 líneas igualaba el resultado.
```

---

## 3.13 Resumen del módulo

- Un **modelo** es la regla aprendida de los datos; el **algoritmo** es la receta que la produce.
- **Supervisado** (hay respuesta): modelos lineales, k-NN, árboles, ensambles (Random Forest, boosting), SVM y Naive Bayes. **No supervisado** (no hay respuesta): K-means, jerárquico, DBSCAN (agrupar) y PCA (resumir).
- Para **tablas**, los ensambles de árboles (Random Forest, XGBoost, LightGBM) son hoy el estándar de oro; las redes neuronales brillan sobre todo en imágenes, texto y audio.
- La elección de modelo depende de la **pregunta, los datos, la interpretabilidad y el tiempo**, no de la moda. Empiece simple, mida, y suba la complejidad solo si se justifica.
- Cada librería tiene su papel: **pandas** carga y limpia, **NumPy** calcula, **Matplotlib/seaborn** grafican, **scikit-learn** modela y evalúa, **statsmodels** infiere, **XGBoost/LightGBM** exprimen precisión.
- Escalar cuando toca, evitar la fuga de información, elegir bien la métrica y respetar el conjunto de prueba son hábitos que separan a un aficionado de un profesional.

---

## 3.14 Glosario

```{list-table}
:header-rows: 1
:widths: 25 75

* - Término
  - Definición breve
* - **Modelo**
  - Regla aprendida de los datos que transforma entradas en una predicción.
* - **Algoritmo**
  - Procedimiento para aprender el modelo a partir de los datos.
* - **Entrenar (`fit`)**
  - Ajustar el modelo con los datos de entrenamiento.
* - **Predecir (`predict`)**
  - Usar el modelo entrenado sobre datos nuevos.
* - **Escalar**
  - Poner las variables en una escala común (p. ej., media 0 y desviación 1).
* - **Pipeline**
  - Cadena que une preprocesamiento y modelo para evitar fugas de información.
* - **Validación cruzada**
  - Rotar particiones del entrenamiento para estimar el desempeño de forma robusta.
* - **Ensamble**
  - Combinar muchos modelos (árboles) para mejorar la predicción.
* - **Hiperparámetro**
  - Ajuste que usted fija antes de entrenar (p. ej., `k` en k-NN, `max_depth`).
* - **Silueta**
  - Medida de qué tan bien separados están los grupos en clustering.
* - **Componente principal**
  - Variable artificial que PCA crea para resumir la variación de los datos.
```

---

## 3.15 Autoevaluación

1. Con sus palabras: ¿cuál es la diferencia entre un algoritmo y un modelo?
2. La regresión logística, ¿sirve para regresión o para clasificación? ¿Por qué el nombre confunde?
3. ¿Qué tienen en común y en qué se diferencian Random Forest y XGBoost?
4. Mencione tres modelos que **exigen** escalar las variables y explique por qué.
5. En K-means, ¿quién decide el número de grupos y por qué eso es un problema?
6. ¿Cuándo preferiría `statsmodels` sobre `scikit-learn`?
7. Un modelo tiene 99 % en entrenamiento y 71 % en prueba. ¿Qué pasa y qué tres cosas haría?
8. ¿Por qué la exactitud es engañosa en detección de fraude con 0,3 % de casos positivos?
9. Para datos tabulares de clientes, ¿empezaría con una red neuronal? Justifique.
10. ¿Qué es la fuga de información y cómo la evita un `Pipeline`?

---

## Referencias (Módulo 3C)

Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
