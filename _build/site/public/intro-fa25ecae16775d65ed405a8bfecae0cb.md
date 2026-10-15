# Machine Learning

## Aprendizaje automático aplicado al análisis de datos

```{figure} images/Portada.jpeg
:width: 520px
:align: center
:alt: Zamora Thowinsson

Zamora Thowinsson
```

### Introducción

El aprendizaje automático se ha convertido en una herramienta importante para el análisis de datos y la solución de problemas en diferentes campos del conocimiento. Hoy encontramos aplicaciones de Machine Learning en salud, economía, administración, ingeniería, mercadeo, ciencias sociales y muchas otras áreas. Sin embargo, aprender Machine Learning implica mucho más que ejecutar algoritmos o escribir unas cuantas líneas de código.

Este curso parte precisamente de esa idea. El propósito es comprender cómo se construye un proceso de aprendizaje automático, desde el momento en que se plantea un problema y se estudian los datos disponibles hasta la selección, entrenamiento, evaluación e interpretación de un modelo.

El recorrido comienza con la preparación del entorno de trabajo y el uso de las herramientas que necesitaremos durante el curso. Trabajaremos principalmente con **Python** y algunas de sus bibliotecas más utilizadas para análisis de datos y Machine Learning. A medida que avancemos, estas herramientas se incorporarán de manera progresiva, procurando entender para qué sirven y por qué las utilizamos.

Una parte fundamental del curso estará dedicada al conocimiento de los datos. Antes de pensar en un algoritmo tendremos que explorar la información, identificar problemas de calidad, estudiar las variables, reconocer valores faltantes, detectar comportamientos atípicos y realizar las transformaciones que sean necesarias. Un buen modelo difícilmente puede construirse sobre datos que no han sido comprendidos adecuadamente.

Posteriormente abordaremos los principales métodos de **aprendizaje supervisado y no supervisado**. Estudiaremos problemas de regresión y clasificación, técnicas de agrupamiento y diferentes algoritmos utilizados actualmente en Machine Learning. El interés no estará únicamente en obtener una predicción. También tendremos que preguntarnos qué tan bueno es el modelo, cómo se comporta con datos que no ha observado anteriormente y si sus resultados tienen sentido dentro del problema que estamos estudiando.

Por esta razón, la evaluación tendrá un papel central. Trabajaremos con métricas de desempeño, procedimientos de validación y estrategias para comparar modelos. También estudiaremos problemas frecuentes, como el sobreajuste, el desbalance de clases y la selección inadecuada de métricas. Más adelante incorporaremos herramientas de interpretación que permitan comprender mejor la relación entre las variables y las predicciones generadas por los modelos.

El curso tiene un enfoque aplicado. Los conceptos estadísticos, matemáticos y computacionales se introducirán cuando sean necesarios para comprender los procedimientos que estamos utilizando. El código será una herramienta de trabajo, no el objetivo final. Cada ejercicio buscará relacionar la programación con la interpretación de los resultados y con las decisiones que pueden derivarse del análisis.

Los ejemplos permitirán trabajar con situaciones provenientes de diferentes áreas. Esto es especialmente importante porque Machine Learning no pertenece exclusivamente a la informática o a la ingeniería. Un profesional de la salud, un economista, un administrador, un investigador social o un ingeniero pueden enfrentarse a problemas distintos, pero todos necesitan desarrollar la capacidad de formular preguntas adecuadas, analizar datos y evaluar críticamente los resultados de un modelo.

A lo largo de estas páginas avanzaremos paso a paso. Algunos temas requerirán programación, otros estadística y otros una combinación de ambas. No se espera memorizar algoritmos ni fragmentos de código. Lo importante será comprender el proceso, aprender a utilizar las herramientas y desarrollar criterio para decidir cuándo una técnica puede ser útil y cuándo no.

Al finalizar este recorrido, el lector deberá estar en capacidad de abordar un conjunto de datos, explorar sus características, preparar la información, construir diferentes modelos de Machine Learning, evaluar su desempeño e interpretar sus resultados con argumentos técnicos.

Ese será el propósito que acompañará todo el curso: **aprender Machine Learning trabajando con datos, entendiendo lo que hacemos y, sobre todo, comprendiendo por qué lo hacemos.**

### Sobre el Autor

Jesús David Zamora Thowinsson es economista y administrador público, especialista y Magíster en Estadística Aplicada (M.Sc.), y actualmente complementa su formación profesional con estudios en Ingeniería de Software.

Su trayectoria académica y profesional se ha desarrollado alrededor del análisis cuantitativo, la estadística, la ciencia de datos y la docencia. Esta combinación le ha permitido abordar los datos no solamente desde la perspectiva matemática y computacional, sino también desde la comprensión de los fenómenos económicos, administrativos y sociales que les dan origen.
Cuenta con experiencia en el uso de **Python, R, SPSS, Excel avanzado y Power BI**, así como en herramientas orientadas al desarrollo y documentación de proyectos reproducibles, entre ellas **Jupyter, Quarto, LaTeX y GitHub**. Su trabajo reciente se ha concentrado especialmente en el análisis estadístico, la visualización de datos y la aplicación de técnicas de *Machine Learning*, incluyendo modelos de regresión y clasificación, métodos basados en árboles, técnicas de ensamble, redes neuronales y aprendizaje no supervisado.

Como docente, su interés se centra en acercar estos métodos a estudiantes y profesionales provenientes de distintas disciplinas. Su enfoque parte de una premisa fundamental: aprender análisis de datos no consiste únicamente en ejecutar código. Es necesario comprender el problema, conocer los datos, seleccionar adecuadamente los métodos, evaluar los resultados y, finalmente, ser capaz de interpretarlos.

En los últimos años también ha incorporado a su trabajo académico el estudio y uso de la inteligencia artificial generativa, particularmente como herramienta para apoyar procesos de enseñanza, investigación, programación, análisis de información y construcción de recursos educativos.

La convergencia entre economía, administración pública, estadística e ingeniería de software define buena parte de su perfil profesional actual. Desde estas áreas busca integrar el razonamiento estadístico con las herramientas computacionales modernas para desarrollar soluciones aplicadas a problemas reales y, al mismo tiempo, facilitar su enseñanza.

Su filosofía de trabajo puede resumirse en una idea que también orienta este curso: los datos adquieren verdadero valor cuando permiten comprender un problema, sustentar una decisión o generar nuevo conocimiento.

**Redes sociales**

<div class="redes-autor">
<a href="https://www.linkedin.com/in/zamora-thowinsson" target="_blank" rel="noopener"><svg viewBox="0 0 24 24" aria-hidden="true"><rect width="24" height="24" rx="3" fill="#0A66C2"/><circle cx="7" cy="7.2" r="1.7" fill="#fff"/><rect x="5.6" y="9.6" width="2.8" height="9" fill="#fff"/><path fill="#fff" d="M10.6 9.6h2.7v1.3c.5-.9 1.6-1.5 3-1.5 2.6 0 3.3 1.7 3.3 4v4.8h-2.8v-4.3c0-1.1-.3-2-1.5-2s-1.9.9-1.9 2v4.3h-2.8z"/></svg>LinkedIn</a>
<a href="https://www.youtube.com/@info.zamora" target="_blank" rel="noopener"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="1" y="5" width="22" height="14" rx="4" fill="#FF0000"/><path fill="#fff" d="M10 8.6 15.8 12 10 15.4z"/></svg>YouTube</a>
<a href="https://www.instagram.com/zamorathowinsson" target="_blank" rel="noopener"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="5" fill="none" stroke="#E4405F" stroke-width="2"/><circle cx="12" cy="12" r="4" fill="none" stroke="#E4405F" stroke-width="2"/><circle cx="17.3" cy="6.7" r="1.2" fill="#E4405F"/></svg>Instagram</a>
<a href="https://www.tiktok.com/@msc..zamora.thowi" target="_blank" rel="noopener"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#fff" d="M12.525.02c1.31-.02 2.61-.01 3.91-.02.08 1.53.63 3.09 1.75 4.17 1.12 1.11 2.7 1.62 4.24 1.79v4.03c-1.44-.05-2.89-.35-4.2-.97-.57-.26-1.1-.59-1.62-.93-.01 2.92.01 5.84-.02 8.75-.08 1.4-.54 2.79-1.35 3.94-1.31 1.92-3.58 3.17-5.91 3.21-1.43.08-2.86-.31-4.08-1.03-2.02-1.19-3.44-3.37-3.65-5.71-.02-.5-.03-1-.01-1.49.18-1.9 1.12-3.72 2.58-4.96 1.66-1.44 3.98-2.13 6.15-1.72.02 1.48-.04 2.96-.04 4.44-.99-.32-2.15-.23-3.02.37-.63.41-1.11 1.04-1.36 1.75-.21.51-.15 1.07-.14 1.61.24 1.64 1.82 3.02 3.5 2.87 1.12-.01 2.19-.66 2.77-1.61.19-.33.4-.67.41-1.06.1-1.79.06-3.57.07-5.36.01-4.03-.01-8.05.02-12.07z"/></svg>TikTok</a>
<a href="https://x.com/jthowinsson" target="_blank" rel="noopener"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#fff" d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z"/></svg>X</a>
</div>
