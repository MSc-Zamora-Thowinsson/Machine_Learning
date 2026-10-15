# Capítulo 0. Preparación del entorno de trabajo
# Capítulo 0. Preparación del entorno de trabajo

## Introducción

Antes de comenzar a desarrollar modelos de *Machine Learning*, es necesario disponer de un entorno de trabajo correctamente configurado. En este capítulo se prepararán las herramientas que utilizaremos durante el curso, procurando que los procedimientos sean reproducibles tanto en **Windows** como en **macOS**.

El entorno estará basado principalmente en **Python**, utilizando **Conda** para la gestión de entornos y dependencias, **Visual Studio Code** como editor de desarrollo y **Jupyter** para la ejecución interactiva de código y análisis de datos.

## Objetivos de aprendizaje

Al finalizar este capítulo, el estudiante estará en capacidad de:

- Instalar y verificar las herramientas necesarias para trabajar con Python.
- Crear y administrar un entorno virtual con Conda.
- Configurar Visual Studio Code para trabajar con Python y Jupyter.
- Instalar las principales bibliotecas utilizadas en Machine Learning.
- Verificar que el entorno de trabajo funciona correctamente.
- Preparar una estructura de trabajo reproducible para los ejercicios y proyectos del curso.

## 0.1. Ecosistema tecnológico de trabajo

El desarrollo de proyectos de *Machine Learning* requiere un entorno computacional que permita integrar programación, gestión de dependencias, experimentación, documentación y control de versiones. En este libro se utilizará un ecosistema de herramientas de código abierto y de uso ampliamente extendido en ciencia de datos, con el propósito de garantizar que los procedimientos desarrollados sean **reproducibles, organizados y transferibles** entre diferentes equipos y sistemas operativos.

### Python

**Python** será el lenguaje de programación principal. Su sintaxis relativamente accesible y, especialmente, su amplio ecosistema de bibliotecas científicas permiten abordar de manera integrada tareas de manipulación de datos, análisis estadístico, visualización y construcción de modelos de *Machine Learning*.

A lo largo del libro se incorporarán progresivamente bibliotecas como:

- **NumPy**, para cálculo numérico y operaciones matriciales.
- **pandas**, para manipulación, transformación y análisis de datos estructurados.
- **Matplotlib**, para visualización de datos.
- **scikit-learn**, para preprocesamiento, modelado, validación y evaluación de algoritmos de *Machine Learning*.
- **XGBoost, LightGBM y CatBoost**, para modelos avanzados basados en *gradient boosting*.
- **SHAP**, para interpretación y explicación de las predicciones generadas por los modelos.

Las bibliotecas especializadas se instalarán progresivamente a medida que sean requeridas, evitando incorporar dependencias innecesarias desde el inicio.

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

## 0.3. Instalación y configuración del entorno en macOS

La preparación adecuada del entorno computacional constituye un requisito fundamental para desarrollar proyectos reproducibles de *Machine Learning*. En macOS se utilizará **Miniconda** para administrar Python y sus dependencias mediante entornos virtuales independientes. Esta estrategia reduce los conflictos entre proyectos y evita modificar innecesariamente la configuración de Python del sistema.

### 0.3.1. Identificación de la arquitectura del equipo

Antes de instalar cualquier herramienta es necesario identificar la arquitectura del procesador.
Los equipos Mac modernos utilizan procesadores **Apple Silicon**, mientras que generaciones anteriores utilizaron procesadores Intel.

Abra la aplicación **Terminal** de macOS y ejecute:

```bash
uname -m
```

El resultado permitirá identificar la arquitectura del equipo:

| Resultado | Arquitectura | Tipo de equipo |
|---|---|---|
| `arm64` | ARM64 | Apple Silicon: M1, M2, M3, M4, M5 o posteriores |
| `x86_64` | x86-64 | Mac con procesador Intel |

> [!Important] No se debe instalar una distribución diseñada para una arquitectura diferente. En equipos Apple Silicon debe utilizarse software compilado para arm64 siempre que esté disponible.

### 0.3.2. Instalación de Miniconda

Miniconda es una distribución ligera que incorpora Python, Conda y los componentes esenciales para administrar entornos de trabajo independientes.
Descargue Miniconda desde su sitio oficial y seleccione el instalador correspondiente a la arquitectura identificada anteriormente.
Una vez finalizada la instalación, cierre y abra nuevamente la Terminal.
Compruebe que Conda está disponible mediante:

````bash
conda --version
````

También puede consultar información detallada de la instalación:

````bash
conda info
````

Si ambos comandos se ejecutan correctamente, Conda está disponible en el sistema.

### 0.3.3. Creación del entorno de Machine Learning

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

### 0.3.4. Instalación de Jupyter

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

### 0.3.5. Instalación de las bibliotecas fundamentales

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

### 0.3.6. Configuración de Visual Studio Code

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

### 0.3.7. Verificación integral del entorno

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

### 0.3.8. Gestión básica del entorno

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

### 0.3.9. Reproducibilidad del entorno

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

#### 0.3.10. Comprobación final

Antes de continuar con los siguientes capítulos, verifique que:

- Conda funciona correctamente.
- El entorno ml_machine puede activarse.
- Python se ejecuta desde ml_machine.
- Jupyter reconoce el kernel Python (ml_machine).
- Visual Studio Code utiliza el intérprete correspondiente al entorno.
- NumPy, pandas, Matplotlib y scikit-learn pueden importarse sin errores.

Una vez superadas estas comprobaciones, el entorno de macOS estará preparado para desarrollar los ejercicios y proyectos de *Machine Learning* incluidos en este libro.
