# MNIST-KNN-DecisionTree

Proyecto realizado como trabajo práctico para la materia "Laboratorio de datos" de la Facultad de Ciencias Exactas
y Naturales", UBA. 

## Descripción del proyecto

El objetivo de este trabajo es desarrollar y analizar dos técnicas de machine learning: KNN y árboles de decisión. 
Para ello se trabaja con la base de datos MNIST ampliamente utilizada en el campo del aprendizaje automático, que 
contiene información sobre imágenes de dígitos numéricos manuscritos. Se crearon modelos de clasificación, donde, 
dada una imagen de un dígito escrito a mano, se predijo a qué dígito correspondía.
En el caso del algoritmo KNN se crearon varios modelos predictores considerando varios subconjuntos de 
atributos diferentes; en el caso de los árboles de decisión se crearon varios modelos con árboles de profundidades
diferentes. Cada uno de estos modelos fueron evaluados usando conjuntos de datos de testing y se analizaron sus 
correspondientes performances, las cuales están informadas en `ìnforme.pdf`.
Ver información mas detallada en `ìnforme.pdf`.
Todo fue implementado en Python. 

## Contenido del repositorio 

El repositorio consta de los siguientes archivos: 

- `enunciado.pdf`: aquí está el enunciado del TP provisto por la materia "Laboratorio de datos"
- `digitos.py`: aquí está el código correspondiente a todos los modelos predictores realizados.
- `informe.pdf`: aquí se presenta con detalles el objetivo del trabajo, qué se hizo, cómo se hizo, y además se
muestran los resultados obtenidos con sus correspondientes análisis.
- `MNIST Dataset\`: acá están, en formato comprimido, los archivos `mnist_desarrollo.csv` (utilizado para
entrenar los modelos de machine learning), `mnist_test.csv`que contiene datos sobre dígitos del 0 al 9
utilizados para testear los modelos realizados y `mnist_test_binario.csv`que contiene datos sobre los dígitos
0 y 1 utilizado también para testear los modelos. 
