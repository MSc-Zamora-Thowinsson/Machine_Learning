---
kernelspec:
  name: python3
  display_name: "Python 3"
---

# Módulo 2B. Comparación de modelos y librerías de Python

> **Subtítulo:** Ventajas y desventajas, cómo elegir, y la caja de herramientas de Python.

```{admonition} De dónde venimos
:class: tip
Esta parte continúa el **Módulo 2A**, donde definimos cada familia de modelos. Aquí los comparamos y conocemos las librerías. Se mantiene la regla didáctica: intuición primero, código al final, todo en un portátil normal.
```

```{admonition} Objetivos de aprendizaje (parte 2B)
:class: note
Al terminar esta parte usted será capaz de:

3. Comparar modelos con criterios prácticos: interpretabilidad, cantidad de datos, velocidad y tipo de problema.
4. Elegir una familia de modelos razonable para un problema real y justificar la elección.
5. Identificar para qué sirve cada librería del ecosistema de datos (NumPy, pandas, Matplotlib, seaborn, scikit-learn, statsmodels, XGBoost, LightGBM) y dejar listo el entorno en Colab o VS Code.
```

---

## 2.6 Comparación de modelos: ventajas y desventajas

Esta es la tabla que más va a consultar en su vida profesional. Léala por filas (cada modelo) y por columnas (cada criterio).

```{list-table} Comparación práctica de modelos clásicos
:header-rows: 1
:widths: 16 10 12 12 25 25

* - Modelo
  - Tipo
  - Interpretable
  - Necesita escalar
  - Ventajas
  - Desventajas
* - **Regresión lineal**
  - Regresión
  - Alta
  - No (ayuda)
  - Simple, rápida, explica el efecto de cada variable
  - Solo capta relaciones rectas; sensible a atípicos
* - **Regresión logística**
  - Clasificación
  - Alta
  - No (ayuda)
  - Da probabilidades, rápida, buena línea base
  - Fronteras solo lineales; floja en relaciones complejas
* - **k-NN**
  - Ambos
  - Media
  - **Sí (obligatorio)**
  - Fácil de entender; sin fase de entrenamiento
  - Lento al predecir; sufre con muchas variables
* - **Árbol de decisión**
  - Ambos
  - **Muy alta**
  - No
  - Reglas claras y explicables; capta no linealidad
  - Sobreajusta fácil si no se controla
* - **Random Forest**
  - Ambos
  - Media-baja
  - No
  - Muy robusto "de fábrica"; pocas sorpresas
  - Más difícil de interpretar; modelo pesado
* - **Gradient Boosting (XGBoost/LightGBM)**
  - Ambos
  - Baja
  - No
  - Suele dar la **mejor precisión** en tablas
  - Requiere ajustar parámetros; riesgo de sobreajuste
* - **SVM**
  - Ambos
  - Baja
  - **Sí (obligatorio)**
  - Fuerte con muchas variables y pocos datos
  - Lento con muchos datos; sin probabilidades directas
* - **Naive Bayes**
  - Clasificación
  - Media
  - No
  - Rapidísimo; excelente para texto
  - Supone independencia (casi siempre falsa)
* - **K-means**
  - No superv.
  - Media
  - **Sí**
  - Simple y veloz para segmentar
  - Hay que fijar *k*; asume grupos redondos
* - **DBSCAN**
  - No superv.
  - Media
  - **Sí**
  - Halla formas raras y atípicos; no fija *k*
  - Difícil con densidades mixtas
* - **PCA**
  - No superv.
  - **Sí**
  - Baja
  - Resume y permite visualizar
  - Componentes poco interpretables
* - **Red neuronal (MLP)**
  - Ambos
  - Muy baja
  - **Sí**
  - Capta patrones muy complejos
  - Necesita más datos; "caja negra"; más lenta
```

```{admonition} Cómo interpretar la columna "Interpretable"
:class: note
"Interpretable" = qué tan fácil es explicarle a un jefe, un auditor o un juez **por qué** el modelo decidió lo que decidió. En crédito, salud y sector público, la interpretabilidad **no es un lujo, es un requisito legal y ético**. A veces conviene sacrificar un poco de precisión por un modelo que se pueda explicar.
```

### 2.6.1 ¿Cómo elijo? Un diagrama de decisión

No existe un "mejor modelo" universal (eso lo vimos en el Módulo 1 con el teorema *No Free Lunch*). Pero sí hay una ruta sensata para empezar:

```{mermaid}
flowchart TD
    A["¿Tengo la respuesta correcta\nen mis datos?"] -->|No| B["No supervisado"]
    A -->|Sí| C["Supervisado"]
    B --> B1["¿Quiero agrupar?"] -->|Sí| B2["K-means / DBSCAN"]
    B --> B3["¿Quiero resumir variables\no visualizar?"] -->|Sí| B4["PCA"]
    C --> D["¿La respuesta es un número\no una categoría?"]
    D -->|Número| E["Regresión"]
    D -->|Categoría| F["Clasificación"]
    E --> G["Empieza con Regresión lineal\n→ si se queda corta: Random Forest / XGBoost"]
    F --> H["Empieza con Regresión logística\n→ si se queda corta: Random Forest / XGBoost"]
    H --> I["¿Necesito explicar\ncada decisión?"] -->|Sí| J["Árbol de decisión"]
```

```{tip}
**Flujo profesional real:** (1) monte primero un modelo simple como **línea base** (lineal o logística). (2) Mida su desempeño. (3) Pruebe un **Random Forest**. (4) Solo si vale la pena, pase a **XGBoost/LightGBM** y ajuste parámetros. El 80 % de los problemas tabulares se resuelven bien en los pasos 1 a 3. scikit-learn publica además un "mapa" oficial para elegir estimador (ver referencias).
```

---

## 2.7 Las librerías de Python: su caja de herramientas

Un modelo no vive solo: necesita cargar datos, limpiarlos, graficarlos y evaluarse. Cada tarea tiene su librería. Piense en ellas como las herramientas de un taller: cada una hace una cosa bien.

```{list-table} Para qué sirve cada librería
:header-rows: 1
:widths: 18 30 30 24

* - Librería
  - ¿Para qué sirve?
  - Analogía
  - La usa cuando...
* - **NumPy**
  - Cálculo numérico con arreglos y matrices
  - La calculadora científica del ecosistema
  - necesita operar con números rápido
* - **pandas**
  - Cargar y manipular tablas (filas y columnas)
  - El Excel programable de Python
  - carga un CSV, filtra, agrupa, limpia
* - **Matplotlib**
  - Gráficos base (líneas, barras, dispersión)
  - El lápiz y papel para dibujar datos
  - quiere control total de una figura
* - **seaborn**
  - Gráficos estadísticos bonitos en pocas líneas
  - Matplotlib con "modo fácil" activado
  - quiere explorar relaciones rápido
* - **scikit-learn**
  - Los modelos clásicos y todo el flujo de ML
  - La navaja suiza del machine learning
  - entrena, evalúa o preprocesa modelos
* - **statsmodels**
  - Modelos estadísticos con inferencia (p-valores, intervalos)
  - El estadístico con bata de laboratorio
  - necesita explicar e inferir, no solo predecir
* - **XGBoost / LightGBM**
  - Gradient boosting de alto rendimiento
  - Los autos de carrera del boosting
  - busca la máxima precisión en tablas
```

```{admonition} scikit-learn vs. statsmodels: la pregunta decide
:class: important
Conecta con las "dos culturas" de Breiman (Módulo 1):
- Si su objetivo es **predecir** lo mejor posible → **scikit-learn**.
- Si su objetivo es **explicar e inferir** (¿este coeficiente es significativo?, ¿cuál es el intervalo de confianza del efecto?) → **statsmodels**.

Una regresión lineal existe en ambas, pero `statsmodels` le entrega la tabla con p-valores y errores estándar que un economista o un epidemiólogo necesita; `scikit-learn` le entrega el motor de predicción y validación.
```

---

## 2.8 Preparación del entorno (Colab y VS Code)

Antes de los ejemplos, deje su entorno listo. Elija su camino.

### 2.8.1 Google Colab (cero instalación)

1. Entre a `colab.research.google.com` con su cuenta de Google.
2. Cree un notebook nuevo. **Ya trae** NumPy, pandas, Matplotlib, seaborn y scikit-learn instalados.
3. Si va a usar XGBoost o LightGBM, por si acaso ejecute en una celda:

```bash
!pip install -q xgboost lightgbm
```

```{admonition} Ventajas y límites de Colab (datos reales, octubre de 2026)
:class: note
- **A favor:** nada que instalar, librerías listas, se comparte como un documento de Google.
- **Límites del plan gratuito:** ~12 GB de RAM, CPU por defecto (la GPU T4 aparece *si hay disponible*, no garantizada), sesiones de hasta ~12 horas y desconexión por inactividad (~90 min). Para este curso sobra: nuestros datasets son diminutos.
```

### 2.8.2 VS Code (en su propio PC)

1. Instale **Python 3.11 o superior** y la extensión **Python** de Microsoft en VS Code (más la extensión **Jupyter**).
2. Abra una terminal en la carpeta de su proyecto y cree un entorno virtual (así no desordena su sistema):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

3. Instale las librerías dentro del entorno:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels xgboost lightgbm jupyter
```

4. Cree un archivo `.ipynb` (notebook) o un `.py`. En un notebook, VS Code le pedirá **seleccionar el kernel**: elija el intérprete de su `.venv`.

```{admonition} La celda de "chequeo" que debe correr siempre primero
:class: tip
Tanto en Colab como en VS Code, empiece sus notebooks verificando versiones. Si algo falla luego, aquí sabrá por qué.
```

```{code-cell} python
import sys, numpy, pandas, sklearn, matplotlib
print("Python     :", sys.version.split()[0])
print("NumPy      :", numpy.__version__)
print("pandas     :", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib :", matplotlib.__version__)
```

---

```{admonition} Sigue en el Módulo 2C
:class: tip
Con el entorno listo y los modelos comparados, en el **Módulo 2C** pasamos a la práctica: ejemplos desarrollados que corren tal cual, plantillas para completar y ejercicios para Colab y VS Code.
```

## Referencias (Módulo 2B)

Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science, 16*(3), 199–231. https://doi.org/10.1214/ss/1009213726

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (pp. 3146–3154). Curran Associates.

McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 56–61). https://doi.org/10.25080/Majora-92bf1922-00a

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In *Proceedings of the 9th Python in Science Conference* (pp. 92–96). https://doi.org/10.25080/Majora-92bf1922-011

Waskom, M. L. (2021). seaborn: Statistical data visualization. *Journal of Open Source Software, 6*(60), 3021. https://doi.org/10.21105/joss.03021
