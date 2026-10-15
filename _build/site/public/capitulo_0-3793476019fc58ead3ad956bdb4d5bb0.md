# Preparación del entorno de trabajo

## Introducción

Antes de comenzar a desarrollar modelos de *Machine Learning*, es necesario disponer de un entorno de trabajo correctamente configurado. En este capítulo se prepararán las herramientas que utilizaremos durante el curso, procurando que los procedimientos sean reproducibles tanto en **Windows**, **macOS** y **Linux**.

El entorno estará basado principalmente en **Python**, utilizando **Conda** para la gestión de entornos y dependencias, **Visual Studio Code** como editor de desarrollo y **Jupyter** para la ejecución interactiva de código y análisis de datos.

## Objetivos

```{admonition} Objetivos de aprendizaje
:class: important
Al finalizar este capítulo, el estudiante estará en capacidad de:

- Instalar y verificar las herramientas necesarias para trabajar con Python.
- Configurar Visual Studio Code para trabajar con Python y Jupyter.
- Crear y administrar un entorno virtual con Conda.
- Instalar las principales bibliotecas utilizadas en Machine Learning.
- Verificar que el entorno de trabajo funciona correctamente.
- Preparar una estructura de trabajo reproducible para los ejercicios y proyectos del curso.
```

## 0.1. Ecosistema tecnológico de trabajo

El desarrollo de proyectos de *Machine Learning* requiere un entorno computacional que permita integrar programación (`Scripting`), gestión de dependencias, experimentación, documentación y control de versiones. En este libro se utilizará un ecosistema de herramientas de código abierto y de uso ampliamente extendido en ciencia de datos, con el propósito de garantizar que los procedimientos desarrollados sean **reproducibles, organizados y transferibles** entre diferentes equipos y sistemas operativos.

### Python

#### Introducción a Python

Todo empezó a finales de diciembre de 1989. Guido van Rossum, un programador holandés que trabajaba en el CWI (Centrum Wiskunde & Informatica, un centro de investigación en Ámsterdam), se encontró con unos días libres durante las fiestas y decidió buscar un proyecto de "hobby" para entretenerse [@severance2015guido]. Lo que quería era crear algo que fuera fácil de leer, que manejara errores de forma elegante y que pudiera comunicarse con el sistema operativo Amoeba, en el que estaba trabajando por aquel entonces [@swaroop2017byte]. Se inspiró en ABC, un lenguaje anterior que conocía bien, pero que tenía sus limitaciones. Así que se puso a escribir código y, sin saberlo, estaba poniendo la primera piedra de algo enorme.

El nombre, por cierto, no tiene nada que ver con el animal. Van Rossum era fanático de los Monty Python, el grupo de comedia británico, y quiso rendirles un homenaje con un nombre corto, un poco irreverente y fácil de recordar [@cwi2015python25]. La primera versión pública, la 0.9.0, vio la luz el 20 de febrero de 1991 a través de un grupo de noticias llamado alt.sources. Ya desde esa versión temprana estaban presentes cosas que hoy nos resultan familiares: clases, herencia, manejo de excepciones, funciones y los tipos de datos básicos como listas y diccionarios. No era un juguete; era una herramienta funcional desde el principio.

La comunidad empezó a formarse rápido. En 1994 se creó comp.lang.python, el foro de discusión que se convirtió en el punto de encuentro de los primeros usuarios. Ese mismo año llegó la versión 1.0, que añadió herramientas de programación funcional como lambda, map, filter y reduce. Guido seguía al mando, pero el desarrollo se volvía cada vez más colaborativo.

#### Por qué todo el mundo lo usa

En la actualidad prácticamente todos lo usan. Google, Meta, Amazon, Netflix, NASA, Pixar, JP Morgan Chase... la lista es interminable. Pero no se trata solo de gigantes tecnológicos. Marketers que automatizan la recolección de datos de la competencia, periodistas que hacen periodismo de datos, especialistas de recursos humanos que analizan métricas de personal, ingenieros de control de calidad que automatizan pruebas... todos han encontrado en Python una herramienta accesible para resolver problemas concretos.

Un estudio publicado en 2025 en la revista IEEE *Transactions on Parallel and Distributed Systems* lo resume sin rodeos: "Python se ha convertido en el lenguaje de facto para la computación científica" [@ziogas2025datacentric]. Y no es una exageración. Ese mismo paper señala que la demanda de soporte de Python en computación de alto rendimiento (HPC) se ha disparado, precisamente porque su ecosistema científico, construido alrededor de `NumPy`, es enormemente productivo. Campos que van desde el *Machine Learning* hasta la *climatología* y el *transporte cuántico* ya han adoptado Python como su lenguaje de elección para nuevos desarrollos.

#### El presente: Python como lengua franca de la IA

Si hay un campo donde Python se ha vuelto absolutamente dominante, ese es la inteligencia artificial. El propio *Guido van Rossum* lo explicó con una idea sencilla: la gente que hoy escribe código para IA empezó en el aprendizaje automático (*Machine Learning*), y en ese mundo Python ya era el estándar[@venegas2026pythonia]. Cuando una tecnología alcanza una masa crítica de usuarios y bibliotecas, se vuelve cada vez más atractiva para nuevos proyectos, y eso refuerza su posición [@venegas2026pythonia]. Es un círculo virtuoso.

Hoy, más del 80% de los profesionales de **machine learning** usan Python, respaldados por herramientas como `TensorFlow`, `PyTorch`, `scikit-learn` y todo el ecosistema de `Hugging Face`. El 71% de las ofertas de empleo para ingenieros de IA exigen conocimientos de Python. No es que sea el único lenguaje que sirve para IA, pero sí es el que tiene las bibliotecas más maduras, la documentación más completa y la comunidad más grande. Y eso, para quien empieza, marca una diferencia enorme.

Pero Python no vive solo de la IA. En **ciencia de datos**, bibliotecas como `Pandas`, `NumPy` y `Matplotlib` son el pan de cada día para convertir datos crudos en información útil. En desarrollo `web`, `Django` y `FastAPI` compiten cara a cara con frameworks de otros lenguajes. En automatización, es el rey indiscutible: `scripts` para renombrar archivos, enviar correos, extraer datos de webs, lo que se te ocurra. Y en educación, es el lenguaje con el que muchísimas universidades enseñan a programar por primera vez.

```{admonition} En resumen
:class: important
**[Python](https://www.python.org/)** será el lenguaje de programación principal. Su sintaxis relativamente accesible y, especialmente, su amplio ecosistema de bibliotecas científicas permiten abordar de manera integrada tareas de manipulación de datos, análisis estadístico, visualización y construcción de modelos de *Machine Learning*.
```

A lo largo del libro se incorporarán progresivamente bibliotecas según áreas de aplicación como:

1. **Librería para el desarrollo web:**

- **Django:** Es un framework web de alto nivel para Python que sigue el patrón MVT (Model-View-Template). Está diseñado para desarrollar aplicaciones web robustas, seguras y escalables de forma rápida. Es considerado un framework "baterías incluidas" porque incorpora autenticación, panel de administración, ORM, seguridad y más. Consultar [djangoproject.com](https://www.djangoproject.com/)

**¿Para qué sirve?**

- Desarrollo de aplicaciones web complejas.
- Sistemas empresariales.
- Portales web y e-commerce.
- Gestión de usuarios y autenticación.
- APIs REST mediante extensiones como Django REST Framework.

- **Flask:** es un microframework web ligero y flexible. Proporciona únicamente las herramientas esenciales para crear aplicaciones web, permitiendo al desarrollador elegir las librerías adicionales que necesite.

**¿Para qué sirve?**

- Aplicaciones web pequeñas y medianas.
- Prototipos rápidos.
- APIs REST.
- Microservicios.

- Proyectos educativos y de aprendizaje.
- FastAPI
- Pyramid

2. **Librerías de procesamiento de datos**

- **NumPy:** Para cálculo numérico y operaciones matriciales. Consultar [numpy.org](https://numpy.org/es/)
- **Pandas:** Para manipulación, transformación y análisis de datos estructurados. Consultar [pandas.pydata.org](https://pandas.pydata.org/)
- **SciPy:** Integración, optimización y estadística. Consultar [scipy.org](https://scipy.org/es/)

3. **Librerías de visualización de datos**

- **Matplotlib:** Grafícos, histogramas, barras y más. Consultar [matplotlib.org](https://matplotlib.org/)
-**Seaborn:** Para gráficos avanzados, regresiones y mapas de calor. Consultar [seaborn.pydata.org](https://seaborn.pydata.org/)
- **Ploty:** Gráficos interactivos y dashboards. 

- **scikit-learn**, para preprocesamiento, modelado, validación y evaluación de algoritmos de *Machine Learning*.
- **XGBoost, LightGBM y CatBoost**, para modelos avanzados basados en *gradient boosting*.
- **SHAP**, para interpretación y explicación de las predicciones generadas por los modelos.

Las bibliotecas especializadas se instalarán progresivamente a medida que sean requeridas, evitando incorporar dependencias innecesarias desde el inicio.

## Anaconda

Anaconda es una distribución de **software libre y de código abierto** de los lenguajes Python y R, ampliamente utilizada en *ciencia de datos,* *aprendizaje automático* (Machine Learning), procesamiento de grandes volúmenes de información, análisis predictivo y cómputos científicos. Fue lanzada el 6 de septiembre de 2012 y es desarrollada por Anaconda, Inc.

En el núcleo de Anaconda se encuentra conda, un gestor de paquetes y entornos de código abierto que se encarga de instalar software, resolver dependencias y mantener los proyectos aislados entre sí, de modo que actualizar un proyecto nunca rompa otro [@anaconda_docs]. Conda es multiplataforma (Windows, macOS y Linux), admite múltiples lenguajes (Python, R, C/C++, Ruby, etc.) y no requiere privilegios de administrador para instalar software o crear entornos.

La Anaconda Distribution (la versión completa) instala conda, Python, Anaconda Navigator (una interfaz gráfica de escritorio) y **cientos de paquetes populares de ciencia de datos y machine learning** en un entorno base preconstruido. Está diseñada para usuarios que desean un entorno completo y listo para usar desde el primer día. Según la documentación oficial de Anaconda, la distribución completa incluye **más de 600 paquetes** y requiere aproximadamente 9.7 GB de espacio en disco. Otras fuentes cifran el espacio necesario en **~4.4 GB** o **~3 GB** como mínimo para la descarga e instalación.

### Miniconda y Conda

Se utilizará **Miniconda** como distribución ligera y **Conda** como gestor de entornos y paquetes. Su función principal será mantener aisladas las dependencias utilizadas en el libro.

Un entorno independiente permite especificar versiones concretas de Python y de sus bibliotecas, reduciendo conflictos entre proyectos y favoreciendo la **reproducibilidad computacional**.

En este libro trabajaremos dentro de un entorno específico para Machine Learning, de manera que las instalaciones realizadas durante el curso no alteren otros proyectos existentes en el equipo.

### Visual Studio Code

**Visual Studio Code (VS Code)** será el entorno principal de desarrollo. Desde esta aplicación será posible trabajar con archivos Python (`.py`), documentos Markdown (`.md`), notebooks de Jupyter (`.ipynb`) y archivos de configuración del proyecto.

Mediante sus extensiones para Python y Jupyter, VS Code permitirá integrar en una misma interfaz la escritura de código, ejecución de notebooks, administración de archivos, terminal, control de versiones y documentación.

### Jupyter

**Jupyter** proporcionará el entorno interactivo utilizado para experimentar con código y desarrollar análisis reproducibles. Los notebooks permiten integrar en un mismo documento:

- código ejecutable;
- resultados numéricos;
- tablas;
- visualizaciones;
- ecuaciones matemáticas;
- explicaciones metodológicas.

Esta estructura resulta especialmente apropiada para la enseñanza de estadística, ciencia de datos y *Machine Learning*, porque permite documentar simultáneamente el procedimiento analítico y sus resultados.

### Jupyter Book 2 y MyST Markdown

**Jupyter Book 2**, basado en **MyST**, será utilizado para construir y organizar este libro. Permitirá integrar capítulos escritos en Markdown con notebooks ejecutables, expresiones matemáticas, referencias, figuras, tablas y otros recursos científicos.

MyST Markdown amplía las capacidades tradicionales de Markdown y facilita la construcción de documentos técnicos estructurados y posteriormente publicables en formato web.

La separación entre las herramientas es importante: **VS Code será el entorno de desarrollo, Jupyter facilitará la experimentación computacional y Jupyter Book/MyST proporcionará la estructura editorial del libro**.

### Git y GitHub

**Git** se utilizará como sistema de control de versiones. Permitirá registrar los cambios realizados durante el desarrollo del proyecto y recuperar versiones anteriores cuando sea necesario.

**GitHub** se utilizará como repositorio remoto para almacenar y respaldar el proyecto. Posteriormente también permitirá publicar y distribuir el libro mediante servicios asociados al ecosistema de GitHub.

## 0.2. Arquitectura general del entorno

El flujo tecnológico utilizado a lo largo del libro puede resumirse de la siguiente manera:

**Python → Conda → VS Code/Jupyter → Jupyter Book + MyST → Git → GitHub**

Cada componente cumple una función diferente dentro del proceso. Esta separación permite construir un entorno de trabajo modular en el que el análisis, la documentación, la gestión de dependencias y el control de versiones puedan mantenerse de forma organizada y reproducible.

En las siguientes secciones se realizará la instalación y configuración del entorno de manera diferenciada para **Windows** y **macOS**, verificando cada componente antes de comenzar con el desarrollo de los modelos de Machine Learning.

## 0.3. Instalación y configuración de entornos

La preparación adecuada del entorno computacional constituye un requisito fundamental para desarrollar proyectos reproducibles de *Machine Learning*. En macOS se utilizará **Miniconda** para administrar Python y sus dependencias mediante entornos virtuales independientes. Esta estrategia reduce los conflictos entre proyectos y evita modificar innecesariamente la configuración de Python del sistema.

### 0.3.1. ¿Qué es un entorno?

En Machine Learning, el término "entorno" (o environment) suele referirse a dos conceptos muy distintos dependiendo del contexto en el que te encuentres:

1. **Entorno de Desarrollo y Ejecución (*Software Environment*)**

Este es el uso más común en el trabajo diario de modelado predictivo, estadística aplicada y despliegue de aplicaciones. Se refiere al ecosistema aislado de software que configuras para construir, entrenar o desplegar un modelo.

- **¿Qué lo compone?** La versión exacta del intérprete (como Python 3.10 o R 4.3) y las versiones específicas de todas las librerías necesarias para que tu código funcione (por ejemplo, XGBoost 2.0, CatBoost 1.2, pandas, Streamlit).

- **¿Por qué es indispensable?** Previene conflictos de dependencias y garantiza la reproducibilidad. Asegura que un script que funciona en tu computadora hoy, funcionará exactamente igual en un servidor de producción o en la máquina de otro investigador mañana, sin que una actualización inesperada rompa el código.

2. **Entorno en Aprendizaje por Refuerzo (Reinforcement Learning)**

Dentro de esta rama específica de la inteligencia artificial, el entorno tiene una definición algorítmica formal.

- **¿Qué es?** Es el "mundo" dinámico o el sistema simulado con el que interactúa un agente de IA para aprender.

- **¿Cómo funciona?** El entorno define las reglas del juego. Recibe las acciones del modelo, actualiza su estado interno según esas acciones, y le devuelve al modelo una recompensa (positiva o negativa) junto con el nuevo estado. El objetivo del modelo es aprender la política óptima para maximizar esa recompensa.

- **Herramientas comunes:** Entornos virtuales (`venv` en **Python**, `renv` en **R**), Anaconda/Miniconda, o contenedores de Docker.

### 0.3.1.1. Entorno en Conda

Un entorno de Conda es un espacio de trabajo aislado (un directorio en tu computadora) que contiene una versión específica de un lenguaje de programación y una colección exacta de paquetes y dependencias. Es creado y gestionado por Conda, un sistema de gestión de paquetes y entornos de código abierto.

Aunque cumple la misma función general que un entorno virtual tradicional de Python (como venv), Conda tiene diferencias técnicas fundamentales que lo hacen especialmente potente para el trabajo en Machine Learning y estadística aplicada:

- **Es agnóstico del lenguaje:** A diferencia de `venv` (que solo sirve para **Python**) o `renv` (**para R**), Conda puede gestionar entornos que contengan Python, R, o incluso ambos al mismo tiempo. Puedes tener un entorno con Python 3.10 para entrenar modelos con LightGBM y otro separado con R 4.3 para análisis estadísticos.

- **Gestiona dependencias binarias externas:** Las librerías pesadas de Machine Learning (como **XGBoost, CatBoost o TensorFlow**) a menudo dependen de librerías compiladas en C o C++ a nivel de sistema. Mientras que `pip` a veces tiene problemas instalando estos binarios complejos, Conda resuelve y empaqueta todas estas dependencias externas de forma nativa.

- **Instala el intérprete en sí:** Con Conda, puedes especificar qué versión exacta de *Python* o *R* quieres al crear el entorno, y Conda la descargará e instalará por ti. Con los entornos virtuales estándar, ya debes tener instalada la versión de Python en tu sistema base.

### 0.3.2 Identificación de la arquitectura del equipo

Antes de instalar cualquier herramienta es necesario identificar la arquitectura del procesador.
Los equipos Mac modernos utilizan procesadores **Apple Silicon**, mientras que generaciones anteriores utilizaron procesadores Intel.

Abra la aplicación **Terminal** de macOS y ejecute:

>[!Note] Para abrir la terminal en Mac, presiona las teclas Comando (⌘) + barra espaciadora para abrir `Spotlight`, escribe **Terminal** y presiona Enter.

```bash
uname -m
```

![Terminal en Mac](images/Terminal-Mac.png)

El resultado permitirá identificar la arquitectura del equipo:

| Resultado | Arquitectura | Tipo de equipo |
| --- | --- | --- |
| `arm64` | ARM64 | Apple Silicon: M1, M2, M3, M4, M5 o posteriores |
| `x86_64` | x86-64 | Mac con procesador Intel |

> [!Important] No se debe instalar una distribución diseñada para una arquitectura diferente. En equipos Apple Silicon debe utilizarse software compilado para arm64 siempre que esté disponible.

### 0.3.3. Instalación de Miniconda

Miniconda es una distribución ligera que incorpora Python, Conda y los componentes esenciales para administrar entornos de trabajo independientes.
Descargue desde su sitio oficial Miniconda y seleccione el instalador correspondiente a la arquitectura identificada anteriormente. 

#### Instalar Miniconda en Mac

1. Visita la página [de descarga de Miniconda](https://continuumio-docs.readthedocs-hosted.com/miniconda/install/#quick-command-line-install) y luego selecciona la **64-bit (Apple silicon) Graphical Installeropción** correspondiente. Se descargará un `.pkg`  instalador. 

![Paso 1](images/mac-images/1.png)

2. Una vez descargado, ejecute el instalador doble clic sobre él. Clic en *Continuar.*

![Paso 2](images/mac-images/2.png)

3. Lea los términos y condiciones y haga clic en *Aceptar*.

![Paso 3](images/mac-images/3.png)

4. Confirme la ubicación de instalación y haga clic en *Continuar*

![Paso 4](images/mac-images/4.png)

5. Espere a que finalice la instalación y, a continuación, haga clic en *Cerrar*

![Paso 5](images/mac-images/5.png)

6. Abra una nueva ventana de *terminal*. Si la instalación se realizó correctamente, la terminal se iniciará sin errores y mostrará tu `base` entorno activado por defecto.

![Paso 6](images/mac-images/6.png)

7. Ejecute el siguiente comando para aceptar los [Términos de Servicio (ToS)](https://www.anaconda.com/legal) de anaconda

````bash
conda tos accept
````

![Paso 7](images/mac-images/7.png)

Una vez finalizada la instalación, cierre y abra nuevamente la Terminal. Compruebe que Conda está disponible mediante:

````bash
conda --version
````

También puede consultar información detallada de la instalación:

````bash
conda info
````

Si ambos comandos se ejecutan correctamente, Conda está disponible en el sistema.

### 0.3.4. Creación del entorno de Machine Learning

Las bibliotecas utilizadas durante el curso no se instalarán directamente en el entorno base. Se creará un entorno independiente denominado ml_machine.

Ejecute:

````bash
conda create -n ml_machine python=3.12
````

Cuando Conda solicite confirmación, escriba:

````bash
y
````

y presione **Enter**.

Active el nuevo entorno:

````bash
conda activate ml_machine
````

La Terminal deberá mostrar el nombre del entorno al comienzo de la línea:

````texto sin formato
(ml_machine)
````

Luego Compruebe la versión de Python:

````bash
python --version
````

Verifique también la ubicación del intérprete:

````bash
which python
````

La ruta obtenida deberá corresponder al entorno `ml_machine`.

### 0.3.5. Instalación de Jupyter

Con el entorno `ml_machine` activado, instale Jupyter y el kernel de Python:

````bash
conda install jupyter ipykernel
````

Registre posteriormente el entorno como kernel de Jupyter:

````bash
python -m ipykernel install --user --name ml_machine --display-name "Python (ml_machine)"
````

Compruebe la instalación:

````bash
jupyter --version
````

El kernel `Python (ml_machine)` podrá seleccionarse posteriormente desde los notebooks utilizados durante el curso.

### 0.3.6. Instalación de las bibliotecas fundamentales

El entorno inicial incorporará las principales bibliotecas necesarias para análisis de datos, visualización y construcción de modelos básicos:

````bash
conda install numpy pandas matplotlib scikit-learn
````

Estas bibliotecas cumplen funciones diferentes dentro del flujo de trabajo:

- NumPy: cálculo numérico y operaciones matriciales.
- pandas: manipulación, transformación y análisis de datos.
- Matplotlib: construcción de visualizaciones.
- scikit-learn: preprocesamiento, entrenamiento, validación y evaluación de modelos de Machine Learning.
Las bibliotecas especializadas, como XGBoost, LightGBM, CatBoost y SHAP, se incorporarán posteriormente cuando sean necesarias.

### 0.3.7. Configuración de Visual Studio Code

**Visual Studio Code (VS Code)** será el entorno principal de desarrollo utilizado en este libro.

Una vez instalado, abra el panel Extensions e instale las extensiones oficiales:

- Python
- Jupyter

Posteriormente, abra la paleta de comandos mediante:

````
⇧ + ⌘ + P
````

Busque:

> `Python: Select Interpreter`

Seleccione el intérprete correspondiente al entorno:

> `ml_machine`

Cuando trabaje con un notebook `(.ipynb)`, seleccione igualmente el kernel:

> `Python (ml_machine)`

De esta manera, VS Code y Jupyter utilizarán el mismo entorno y las mismas dependencias del proyecto.

### 0.3.8. Verificación integral del entorno

Con `ml_machine` activo, ejecute:

````bash
python -c "import numpy, pandas, matplotlib, sklearn; print('Entorno configurado correctamente')"
````

Si la configuración es correcta, la Terminal mostrará:

`Entorno configurado correctamente`

También puede realizar una prueba desde un notebook:

> `Nota Copie y pegue el siguiente bloque de código dentro de Vs code`

````python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
````

La ejecución sin errores confirma que Python y las bibliotecas científicas fundamentales están disponibles en el entorno.

### 0.3.9. Gestión básica del entorno

Cada vez que se inicie una nueva sesión de trabajo deberá activarse el entorno:

````bash
conda activate ml_machine
````

Para salir del entorno:

````bash
conda deactivate
````

Para consultar los entornos disponibles:

````bash
conda list
````

El uso de un entorno independiente permite mantener controladas las dependencias del proyecto y facilita su mantenimiento.

### 0.3.10. Reproducibilidad del entorno

La reproducibilidad constituye un principio fundamental en estadística aplicada, ciencia de datos y *Machine Learning*. Un análisis debería poder ejecutarse nuevamente bajo condiciones computacionales equivalentes.

Conda permite exportar las dependencias del entorno mediante:

````bash
conda env export > environment.yml
````

El archivo `environment.yml` puede utilizarse posteriormente para reconstruir el entorno:

````bash
conda env create -f environment.yml
````

De esta manera, otro usuario puede reproducir una configuración compatible sin instalar manualmente cada dependencia.

#### 0.3.10.1 Comprobación final

Antes de continuar con los siguientes capítulos, verifique que:

- Conda funciona correctamente.
- El entorno ml_machine puede activarse.
- Python se ejecuta desde ml_machine.
- Jupyter reconoce el kernel Python (ml_machine).
- Visual Studio Code utiliza el intérprete correspondiente al entorno.
- NumPy, pandas, Matplotlib y scikit-learn pueden importarse sin errores.

Una vez superadas estas comprobaciones, el entorno de macOS estará preparado para desarrollar los ejercicios y proyectos de *Machine Learning* incluidos en este libro.
