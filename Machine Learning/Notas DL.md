# Hands On - Parte 2: Neural Networks

## Introduction to ANN with Keras



### Ejercicios

3. Es preferible utilizar la Regresión Logística frente al perceptrón, porque este último converge sólo cuando los datos son linealmente separables, y no será capaz de devolver las probabilidades de clase. Si se cambia la función de activación de activación a logística, o softmax, y entrenamos usando GD o CE, entonces el perceptrón es equivalente a la Regresión Logística.
4. La función de activación logística fue un elemento clave en el entrenamiento de las primeras MLP porque su derivada no se anula en ningún punto. Cuando la función de activación es una función escalón, el gradiente no se puede mover, por lo que no hay pendiente. 
5. Sigmoide, ReLU, tanh, step function. 

## Training DNN

### Problemas con gradientes nulos y divergentes

- Problemas con la inicialización usual de los _weights_ mediante una distribución normal de media 0 y dispersión 1. 
- La función de activación se satura para valores altos de los inputs, llevando a un gradiente muy pequeño. 

__Soluciones propuestas:__

- _Xavier initialization_, que es una distribución normal o uniforme que tiene medias y desviaciones estándar que están calculadas en relación al número de inputs y outputs de la red (ver pág. 437). Similar es la inicialización de LeCun. Hay entonces, inicializaciones que se prefieren, tomando como referencia la función de activación que se va a implementar en el red neuronal.

  <img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\w_init.png" style="zoom:75%;" />

- Funciones de activación que no saturan, y en las que no "mueren" neuronas (ReLU, leaky ReLU, RReLU, PReLU,ELU)
- Otra función que garantiza que no va a haber gradientes nulos/divergentes es la función SELU, la cual garantiza (bajo ciertas condiciones) que la salida de cada capa tiende a preservar una media de 0 y una desviación estándar 1.
- _Batch Normalization_: Se agrega una operación en el modelo justo antes o después de la función de activación de cada hidden layer. Esta operación centra y normaliza cada input, luego escalea el resultado usando dos nuevos parámetros por capa. Esto mejora considerablemente la performance en todas las redes neuronales, y actúa a la vez como regularizador. El entrenamieto resulta ser más lento, pero la convergencia resulta ser más rápida. Tiene unos pocos hiperparámetros, siendo el más importante de ellos el `momentum`, el cual se encarga de hacer la actualización de los movimientos de los vectores que tienen los parámetros para normalizar el batch. Otro es `axis`, que especifica la dirección del tensor a lo largo del cual se hace la normalización. Es una capa muy usada, al punto de que hoy a menudo se omite en los diagramas, asumiendo que una capa de BN se añade después de cada capa.
- _Gradient clipping:_ Otra técnica popular para mitigar el problema de los gradientes que explotan es clipearlos en la backpropagation con algún threshold. Este truco es más común en las RNN.

### Reutilizar Capas previamente entrenadas

Uno siempre debería ser capaz de encontrar una red neuronal existente que lleva a cabo una tarea similar a la que se intena hacer. Este approach es el _transfer learning_.

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\optimizers.png" style="zoom:75%;" />

- Learning rate scheduling
- Regularización
- Dropout

![](C:\Users\Nico\Dropbox\Machine Learning\Figuras\default_dnn.png)

![](C:\Users\Nico\Dropbox\Machine Learning\Figuras\sn_default_dnn.png)

### Ejercicios

1. Los pesos deben ser sampleados independientemente; el objetivo de esto siempre es romper la simetría; si esto no ocurre, el algoritmo de backpropagation no puede converger. 
2. Está oki.
3. 
