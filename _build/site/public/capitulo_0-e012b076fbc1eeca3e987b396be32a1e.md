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

Antes de realizar la instalación es necesario identificar la arquitectura del procesador. Abra la aplicación **Terminal** y ejecute:

```bash
uname -m
```
El resultado permitirá identificar la arquitectura del equipo:
| Resultado | Arquitectura | Tipo de equipo |
|---|---|---|
| `arm64` | ARM64 | Mac con Apple Silicon |
| `x86_64` | x86-64 | Mac con procesador Intel |

Los equipos con procesadores Apple Silicon M1, M2, M3, M4, M5 y generaciones posteriores utilizan arquitectura ARM64. La arquitectura identificada determinará qué versión de Miniconda debe instalarse.

### 0.3.2. Instalación de Miniconda

Miniconda es una distribución ligera que incorpora Python, Conda y los componentes esenciales para administrar entornos de trabajo independientes.
Descargue Miniconda desde su sitio oficial y seleccione el instalador correspondiente a la arquitectura identificada anteriormente.
Una vez finalizada la instalación, cierre y abra nuevamente la Terminal.
Compruebe que Conda está disponible mediante:

````bash
conda --version
````