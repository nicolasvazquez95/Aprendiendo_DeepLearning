# Notas Hands On

## Capítulo 4: _Training models_

Vamos a empezar mirando el modelo de Regresión Lineal. Discutiremos dos formas de entrenarlo:

- Usando una ecuación cerrada que calcula directamente los parámetros del modelo que mejor fitean el modelo.
- Usando un approach de optimización iterativa llamado Descenso de Gradiente. 

Después vamos a revisar la regresión polinomial, un modelo más complejo que puede fitear datasets no lineales. Finalmente, vamos a mirar dos modelos que son usados comúnmente para clasificación: Logistic Regression y Softmax Regression.

### Linear Regression

El modelo puede ser resumido por la siguiente ecuación $\ref{lin_reg}$ :
$$
\hat{y} = \mathbf{\theta}\cdot\mathbf{x}\tag{4.1}\label{lin_reg},
$$
donde $\theta$ es el vector de parámetros, que contiene el término de bias $\theta_0$ y el de pesos. $\mathbf{x}$ es el vector de features, con $x_0=1$.

Para entrenar el modelo, necesitamos una medida de qué tan bien fitea el modelo la data de training. La forma de medida más común en los modelos de regresión es el error medio cuadrático (RMSE). Entrenar un modelo de regresión lineal significa encontrar $\mathbf{\theta}$ que minimiza RMSE (o MSE).
$$
\operatorname{MSE}(\mathbf{X},h_\theta) = \frac{1}{m}\sum_{i=1}^m\big(\mathbf{\theta^T x^{(i)}-y^{(i)}}\big)^2
\label{mse}\tag{4.2}
$$
 Para encontrar el valor de $\mathbf{\theta}$ que minimiza la función de costo existe una solución cerrada, normalmente conocida como la ecuación normal.
$$
\hat{\mathbf{\theta}} = (\mathbf{X^T X})^{-1}\mathbf{X^T y}\tag{4.3}
$$
__Nota:__ En la mayoría de los algoritmos lo que está implementado no es la inversa, sino la [_pseudoinversa_](https://es.wikipedia.org/wiki/Pseudoinversa_de_Moore-Penrose), que es una generalización para matrices $n\times m$.

El costo computacional de este algoritmo (implementado en [Scikit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)) está alrededor de $\mathcal{O}(n^2)$.

### Descenso de Gradiente

Es un algoritmo de optimización genérico, capaz de encontrar soluciones óptimas en una gran variedad de problemas. La idea general es modificar los parámetros iterativamente para minimizar alguna función de costo.

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\GD.png" style="zoom:50%;" />

Un hiperparámetro importante es el _learning rate_. Si es muy pequeño, el algoritmo va a iterar muchas más veces para converger. En cambio, si es muy alto, podríamos hacer que el algoritmo diverja, al no conseguir que se dirija en la dirección esperada.

La función de costo MSE es convexa (que significa que si tomamos dos puntos de la curva, el segmento de línea que los une nunca cruza la curva). Esto implica que no hay mínimos locales, sino un único mínimo global. Además es una función suave. Bajo esas condiciones, el descenso de gradiente garantiza convergencia arbitrariamente cerca al mínimo global.

Si las features tienen diferentes escalas, el descenso de gradiente puede ser más lento en alguna de las direcciones, haciendo que tarde más en converger.

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\gd_features.png" style="zoom:50%;" />

Para implementar DG, necesitamos calcular el gradiente de la función de costo con respecto a $\theta_j$. 
$$
\frac{\part}{\part\theta_j}\operatorname{MSE}(\mathbf{\theta}) = \frac{2}{m}\sum_{i=1}^m(\mathbf{\theta^T x}^{(i)}-\mathbf{y}^{(i)})\;x_j
\label{ddj_mse}\tag{4.4}
$$
En notación vectorial,
$$
\nabla_\theta\operatorname{MSE}(\mathbf{\theta}) = \frac{2}{m}\mathbf{X^T}(\mathbf{X\theta-y}).
$$
Una vez que tenemos el vector gradiente, la regla de iteración es
$$
\mathbf{\theta}_{n+1} = \mathbf{\theta}_n -\eta\nabla_\theta\operatorname{MSE}(\mathbf{\theta})\label{theta_n+1}\tag{4.5},
$$
donde incorporamos $\eta$ como el _learning rate_ del algoritmo.

### Descenso de gradiente estocástico

El principal problema con el método que hemos estudiado es que necesita el training set completo para calcular el gradiente a cada paso, lo que se vuelve lento. En vez de eso, el descenso de gradiente estocástico elige una instancia de manera aleatoria en el training set a cada paso, y calcula los gradientes basados sólo en esa instancia. Como consecuencia, debido a su naturaleza, el algoritmo va a converger más lento al mínimo, descendiendo únicamente en promedio. Una vez que el algoritmo se detiene, los parámetros finales serán buenos, pero no los óptimos, puesto que una vez cerca del mínimo, los parámetros seguirán cambiando ligeramente.

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\SGD.png" style="zoom:50%;" />

Para sortear las dificultades de que el algoritmo no termina de asentarse en el mínimo, una posible solución es ir ajustando el learning rate a medida que vamos iterando, mediante alguna función (_learning schedule_).

### Descenso de gradiente en mini-batches

Una vez que entendimos los otros dos, este es sencillo: en vez de calcular los gradientes usando todas las instancias, o una única instancia al azar, en este caso se calculan los gradientes de pequeños sets de instancias (_mini batches_). La principal ventaja es la optimización de las operaciones con matrices, especialmente al usar GPUs.

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\3GD.png" style="zoom:50%;" />

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\tabla_gd.png" style="zoom:80%;" />

### Regresión Polinomial

Podemos usar un modelo lineal para fitear datos no lineales! Podemos añadir (por ejemplo) potencias de cada feature como nuevas features, y entrenar el modelo en este nuevo conjunto. Esta técnica es la _regresión polinomial_.

> `PolynomialFeatures(degree=d)` transforma un array de $n$ features en otro que contiene $\frac{(n+d)!}{d!n!}$ features. La explosión es de orden combinatorio!

### Modelos lineales regularizados

Para un modelo lineal, la regularización del modelo (es decir, imponer condiciones de borde sobre los parámetros para prevenir que overfitee) se realiza sobre los pesos. Hay tres maneras comunes en las que se hace, las dejo anotadas sin entrar en detalles (ver libro):

1. Ridge Regression (regularización de Tikhonov): Se añade un término $\alpha\sum_{i=1}^n \theta_i^2$, controlado por el hiperparámetro $\alpha$.
2. Lasso Regression: Similar a Ridge, pero utiliza la norma $\mathcal{l}_1$ del vector de pesos. El término que se añade es entonces $\alpha\sum_{i=1}^n \abs{\theta_i}$
3. Elastic Net: Viene a ser el punto medio entre las dos primeras. El término de regularización es una mezcla de los dos anteriores, controlados por el mix ratio $r$. Cuando $r = 0(1)$, recuperamos Ridge(Lasso).

#### Parada temprana

Una forma de regularizar algoritmos de aprendizaje iterativos es deteniendo el entrenamiento tan pronto como el error de validación alcance un mínimo.

### Regresión Logística

Se usa para estimar la probabilidad de que una instancia dada pertenezca a una clase particular. 

¿Cómo funciona? El modelo computa una suma pesada de las features, pero en vez de devolver el resultado directamente, devuelve ese valor evaluado en la función sigmoide, es decir:
$$
\hat{p} = \sigma(\mathbf{x^T \theta}).
\tag{4.6}
$$
Una vez calculado $\hat{p}$ , el modelo estima a partir de aquí el valor de $\hat{y}$, con un valor de corte de $0.5$.

#### Generalización: Regresión Softmax

La generalización de la regresión logística para soportar clases múltiples directamente, es la regresión _Softmax_ o multinomial.

La idea es simple: dada una instancia $\mathbf{x}$, la regresión calcula un score $s_k(\mathbf{x})$ para cada clase $k$, entonces estima la probabilidad de cada clase al aplicar la función softmax (que es como la función de partición :) ). Luego elige $\hat{y}$ como el máximo entre todos los $\hat{p}_k$. 

Para el entrenamiento, la función de costo elegida para este modelo es la de entropía cruzada. Para dos clases, es equivalente a la de la función logística.
$$
J(\mathbf{\Theta}) = -\frac{1}{m}\sum_i^m\sum_k^Ky_k^{(i)}\log(\hat{p}_k^{(i)})
\label{softmax_costo}\tag{4.7}
$$


### Ejercicios

1. La mejor opción es utilizar descenso de gradiente en minibatches.

2. El descenso de gradiente puede tener problemas de convergencia si las features no están escaleadas de manera homogénea. Hay que usar alguna herramienta de estandarización, como `StandardScaler`.

3. La función de costo de la Regresión Logística es convexa, asique el descenso de gradiente (así como cualquier algoritmo de optimización) garantiza aproximación arbitraria al mínimo global.

4. _Del libro_: Si dejamos correr un tiempo suficiente, y asumimos que $\eta$ no es muy alto, eventualmente todos los algoritmos van a generar modelos similares. Sin embargo, hay que tener en cuenta que tanto el descenso estocástico como el de minibatches no va a converger realmente, si no que va a quedar oscilando alrededor del verdadero mínimo, sin importar cuánto tiempo lo dejemos entrenar. 

5. Si esto ocurre, el modelo está overfiteando sobre el set de entrenamiento. Lo que se puede hacer es incorporar más datos para entrenar, reducir la complejidad del modelo, o imponer condiciones de borde sobre los parámetros. Otra cosa más que se puede hacer es agregar términos a la función de costo que normalicen los parámetros, como es el caso de las regresiones Ridge y Lasso. Eso si el error en el entrenamiento sigue bajando o se mantiene estable. Si el error de entrenamiento también sube, lo que hay que hacer es reducir $\eta$. 

6. Ni idea. _Del libro_: Por su naturaleza aleatoria, el descenso de gradiente con minibatch no garantiza hacer progresos en _todas_ las iteraciones. Si detenemos el entrenamiento muy rápido, quiza lo estamo haciendo antes de que el algoritmo tenga tiempo de encontrar la solución óptima. Una opción es guardar el modelo a intervalos regulares; entonces, si no ha mejorado en un tiempo considerable, podemos revertir al mejor modelo guardado.

7. El descenso de gradiente estocástico es el que itera más rápidamente, por lo tanto es más probable que sea el primero en alcanzar regiones cercanas a la solución óptima. Sin embargo, el único algoritmo que garantiza convergencia es el Descenso de Gradiente al entrenar sobre el conjunto entero de instancias. Para ayudar a la convergencia del DG estocástico y minibatch, lo que se puede hacer es implementar un _learning schedule_.

8. Es casi seguro que el modelo está overfiteando sobre el set de entrenamiento. Puede simplificarse el modelo (reducir el grado del polinomio), aumentar el número de instancias de entrenamiento, o normalizar los pesos del modelo.

9. El modelo está haciendo _underfitting_ sobre el set de entrenamiento. El modelo está sufriendo _high bias_ (muy sesgado). Lo que debería hacerse es reducir el valor de $\alpha$, para reducir el grado de normalización de los pesos. 

10.  

    (a) Un modelo con regularización generalmente funciona mejor que uno sin regularización; es por ello que deberíamos preferir una Ridge Regression antes que una Linear.

    (b) La regresión Lasso usa norma $\mathcal{l_1}$, lo que tiende a llevar los pesos a cero. Esto lleva a modelos dispersos, donde casi todos los pesos son cero excepto los más importantes. Esta es una manera de hacer _feature selection_. Si no estamos seguros, es preferible usar Ridge Regression.

    (c) Elastic Net es preferible a Lasso porque esta última tiene algunos problemas cuando las features tienen mucha correlación, o si hay más features que instancias. 

11. Las combinaciones no son excluyentes, por lo que habría que entrenar dos regresiones logísticas por separado. 

## Capítulo 5: Support Vector Machines

### Ejercicios

1. La idea de la clasificación por SVM es fitear dos rectas paralelas lo más distanciadas posibles entre ambas clases (lo que el libro le dice "la calle más ancha"). El hiperparámetro `C`es el que hace el balance entre alcanzar este objetivo, y separar las clases tan precisamente como sea posible. Otra idea clave es la de usar kernels cuando se entrena en datasets no lineales.
2. El _support vector_ es el vector que sostiene el armado de las líneas paralelas que definen el "margen" entre las clases. El libro dice que son las instancias que están localizadas sobre estas paralelas. El borde de decisión está totalmente determinado por los support vectors. Todas las instancias que no sean support vectors no tienen influencia, y calcular predicciones con el modelo sólo involucra a los support vectors.
3. El _warning_ de la página 220 tiene la respuesta: el escaleo de las figuras permite construir mejores márgenes de decisión. Los SVM son sensitivos al escaleo de las features.
4. No puede devolver probabilidades, por cómo está construido el algoritmo. (Ejercicio 1). De todas formas, podemos pedirle al algoritmo la distancia entre la instancia y el _decision boundary_, que lo podemos usar de score. 
5. Para un set de entrenamiento tan grande, lo más conveniente es el problema primal. La pág 234 dice que el problema dual es más rápido de resolver cuando el número de instancias es menor al número de features. Además,esta pregunta sólo aplica para SVM lineales. Los SVM con kernel sólo pueden usar el problema dual. 
5. Si está haciendo underfit sobre el training set, lo que debe hacerse es aumentar ambos valores.
5. Ni idea. Tampoco entiendo mucho todo esto de los problemas QP.

## Capítulo 6 - Decision Trees

### CART Training Algorithm

Scikit usa el algoritmo _Clasification and Regression Tree_ (CART) para entrenar árboles de decisión. Este algoritmo trabaja dividiendo el training set en dos subsets usando una única feature $k$ y un threshold. ¿Cómo elige el par $(k,t_k)$? Lo que hace es buscar el par que produce los subsets más puros (pesados por su tamaño). La ecuación $\ref{eq:J_DT}$ muestra la función de costo que el algoritmo intenta minimizar:
$$
J(k,t_k) = \frac{m_{\rm{left}}}{m}G_{\rm{left}}+\frac{m_{\rm{right}}}{m}G_{\rm{right}}
\label{eq:J_DT}\tag{6.1}, \rm{donde}\\
\begin{cases}
G_\rm{left/right}\;\text{mide la impureza del subset l/r,}\\
m_\rm{left/right} \;\text{es el número de instancias en el subset l/r}.
\end{cases}
$$
Una vez que CART divide el training set en dos, vuelve a dividir sobre los subsets usando la misma lógica, y así sucesivamente, de forma recursiva. Se detiene una vez que alcanza la máxima profundidad (que es un hiperparámetro), o si no puede encontrar un split que reduzca la impureza. Otros hiperparámetros controlan algunas condiciones adicionales de stop (`min_samples_split`,`min_samples_leaf`,`min_weight_fraction`, y `max_leaf_nodes`).

> Warning: CART es un algoritmo _greedy_ (codicioso). Busca el split óptimo en el nivel superior, y repite el proceso en cada nivel siguiente. Sin embargo, no chequea si el split va a llevar a la menor impureza varios niveles abajo. Un algoritmo de este tipo produce soluciones razonablemente buenas pero que no está garantizado que son óptimas. 
>
> Encontrar el árbol óptimo es un problema NP-completo. 

### Complejidad Computacional

Las predicciones son rápidas, orden $\mathcal{O}(\log_2(m))$, debido a que los árboles generalmente están balanceados. El algoritmo de entrenamiento compara todas las features (excepto que  `max_features` esté seteado) en cada nodo. Comparar todas las features en todas las muestras en cada nodo resulta en un costo $\mathcal{O}(nm\log_2(m))$. Para datasets pequeños, el proceso se puede acelerar ordenando los datos previamente (set `presort=True`), pero no es recomendable en training sets grandes.

### Impureza o Entropía Gini?

Por default, la medida Gini de impureza se utiliza, pero se puede utilizar la entropía en el hiperparámetro `criterion`. La ecuación $\ref{eq:Hi_DT}$ es la definición de entropía para el el $i$-ésimo nodo:
$$
H_i = \sum_{i=1}^n p_{i,k}\log_2(p_{i,k}).
\label{eq:Hi_DT}\tag{6.2}
$$
 Usar uno u otro criterio no hace una gran diferencia: Gini es más rápido de calcular, mientras que la entropía tiende a producir árboles más balanceados.

### Hiperparámetros de regularización

Si el árbol de decisión no tiene condiciones de borde, la estructura se va a adaptar a los datos, y es muy posible que el algortimo overfitee. Estos modelos son no paramétricos, por lo tanto la estructura es libre de ajustarse cercana a los datos. 

Para evitar esto, hay que restringir la libertad del árbol de decisión. Esto es, _regularización_.

- Reducir `max_depth`regularizará el modelo, restringiendo la profundidad del árbol. 
- `min_samples_split`: mínimo número de muestras que debe tener un nodo antes de dividirse.
- `min_samples_leaf`: mínimo número de muestras que un nodo hoja debe tener. 
- `min_weight_fraction_leaf`: Similar a `min_samples_leaf`, pero expresado como fracción del nro. total de instancias pesadas.
- `max_leaf_nodes`: máximo nro. de nodos hoja. 
- `max_features`: máximo nro. de features que son evaluadas para dividir en cada nodo. 

Incrementar los mínimos o reducir los máximos en los hiperparámetros regularizará el modelo.

### Regresión

Los árboles de decisión son similares a los de la clasificación, pero la diferencia principal es que en vez de predecir una clase en cada nodo, predice un valor. 

![](C:\Users\Nico\Dropbox\Machine Learning\Figuras\dtregressor.png)

El algoritmo CART trabaja casi igual que antes, excepto que ahora intenta dividir el training set de manera que se minimice el MSE. 

### Inestabilidad

Los árboles de decisión son simples para entender e interpretar, fácil su uso, versátiles, y poderosos. Sin embargo, tienen algunas limitaciones.  Como se habrá notado, los márgenes de decisión de los árboles de decisión son ortogonales (perpendiculares a algún eje), lo que hace que sean sensibles a rotación de los datos.  Una forma de limitar este problema es usar Principal Component Analysis, que a menudo resulta en una mejor orientación de los datos de entrenamiento.

El principal problema con los árboles de decisión es que son muy sensibles a pequeñas variaciones en los datos de entrenamiento. Los Random Forests limitan esta inestabilidad promediando las predicciones sobre muchos árboles, como vamos a ver en el siguiente capítulo. 

### Ejercicios

1. Los árboles están más o menos balanceados, por lo tanto el costo computacional (es decir, `depth`) es $\mathcal{O}(\log_2(m))$. Por lo tanto, $\text{depth}\approx 20$.
2. La impureza Gini de un nodo es generalmente menor que la de su nodo padre. Sin embargo, es posible para un nodo tener una impureza mayor que su nodo padre, siempre que este incremento esté más que compensado por un descenso en la impureza del otro hijo. 
3. Si el árbol overfitea sobre training set, es una buena idea _reducir_ el parámetro `max_depth`.
4. Los árboles no miran si los datos están escaleados o centrados. Si está underfiteando, escalear las features no va a generar ningún cambio.

5. Si las instancias tienen el mismo nro. de features, el costo computacional de entrenar el arbol es $\mathcal{O}(m\log_2(m))$. El incremento es casi lineal, y el tiempo de entrenamiento va a ser aprox. 10 veces mayor (~10hs).

6. Preordenar los datos acelera el entrenamiento sólo en datasets pequeños (unos pocos miles de instancias). En un dataset de 100000, vamos a ralentizar considerablemente el entrenamiento. 

## Ensemble Learning and Random Forests

Si juntamos las predicciones de un grupo de predictores (clasificadores, regresores), a menudo las predicciones serán mejores que las de los predictores indivuduales por separado. Un grupo de predictores es llamado _ensamble_, y la técnica es llamada _Ensemble Learning, y un algoritmo de ensamble es un _Ensemble method_.

Como ejemplo de ensamble, podemos entrenar un grupo de árboles de decisión, cada uno en un subconjunto random del training set. Luego, trabajamos como en el último ejercicio del Capítulo 6. Este ensamble es un _Random Forest_, y es uno de los algoritmos de machine learning más poderosos disponible.

Las soluciones ganadores en las competiciones de Machine Learning a menudo involucran métodos de ensamble.

En este capítulo vamos a discutir los métodos de ensamble más populares, incluyendo _bagging,boosting,stacking_. 

### Voting Classifiers

Supongamos que hemos entrenado algunos clasificadores, cada uno con alrededor de 80% de precisión. Una manera simple de crear un clasificador aún mejor es agregar las predicciones de cada clasificador y predecir la clase con mayoría de votos. Este _voto democrático_ es un _hard voting classifier_. 

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\hardvotingclf.png" style="zoom:67%;" />

Incluso si cada predictor es débil (es decir, apenas un poco mejor que el _random guessing_), el ensamble aún puede alcanzar alta precisión, dado un número suficiente de predictores débiles suficientemente diversos. Esto se debe a la _ley de los grandes números_. 

> Los métodos de ensamble funcionan mejor cuando los predictores son tan independientes el uno del otro como sea posible. Una manera de conseguir esto es entrenarlos usando algoritmos muy distintos. Esto incrementa la posibilidad de que los errores cometidos sean muy distintos, mejorando la precisión del ensamble en conjunto. 

Se puede hacer _soft voting_, haciendo un promediado de las probabilidades (que devuelven los métodos `predict_proba()` de cada clasificador). A menudo esto consigue una mejor performance, porque da mayor peso a los votos con alto grado de confianza. En el notebook hay un ejemplo en código de esto.

### Bagging y pasting

Una forma de conseguir un conjunto diverso de clasificadores es usar algoritmos de entrenamiento distintos, como discutimos. Otro approach es usar el mismo algoritmo para cada predictor, y entrenarlo sobre subsets aleatorios en el training set. Cuando el sampleo es hecho con reemplazo, el método es llamado _bagging_, y sin reemplazo es llamado _pasting_. Sólo el bagging permite que las instancias de entrenamiento sean sampleadas varias veces por el mismo predictor (esto es _bootstrapping_). 

Cuando todos los predictores están entrenados, el ensamble puede hacer una predicción para una nueva instancia simplemente agregando las de todos los predictores. Usualmente, la función de agregación es la moda estadística. Si bien cada predictor individual tiene bias más altos que los que tendría al entrenar sobre todo el training set, la agregación reduce los bias y la varianza. El resultado neto, generalmente, es que el ensamble tiene similar bias pero varianza más baja que un único predictor entrenado sobre todo el training set original.

> Bootstrapping introduce más diversidad en los subsets, asique el bagging tiene un poco más de sesgo que el pasting; sin embargo, la diversidad extra hace que los predictores estén menos correlacionados, y la varianza se reduce. En total, bagging a menudo resulta en mejores modelos, y es generalmente preferido.

#### Out-of-Bag Evaluation

Con bagging, algunas instancias pueden ser sampleadas varias veces para un predictor dado, y otras no sampleadas. En promedio, el 63% de las instancias son sampleadas por cada predictor (??).  El 37% restante de las instancias que no son sampleadas son instancias _out-of-bag (oob)_ . Como un predictor nunca ve las instancias _oob_ durante el entrenamiento, puede ser evaluado en estas instancias, sin necesidad de un set de validación aparte. En Scikit, esto se configura seteando `oob_score=True` al crear un `BaggingClassifier`.

### Random Patches and Random Subspaces

`Bagging Classifier`soporta sampleo de las features. El sampleo está controlado por dos hiperparámetros: `max_features` y `bootstrap_features`. Trabajan de la misma manera que `max_samples` y `bootstrap`, pero para sampleo de las features. De esta forma, cada predictor es entrenado sobre un subset random de las features.

Esta técnica es útil al lidiar con inputs de muchas dimensiones (imágenes, por ejemplo). Samplear instancias y features es el método _Random patches_. Conservar todas las instancias (seteando `bootstrap=False`, `max_samples=1.0`) y samplear las features (seteando `bootstrap_features=True` y/o `max_features`$\leq$1.0) es el método de _Random subspaces_.

Samplear features resulta en más diversidad de los predictores, intercambiando un poco más de bias por una menor varianza.

### Random Forests

Un _Random Forest_ es un ensamble de árboles de decisión, generalmente entrenado por bagging, típicamente con `max_samples` del tamaño del training set. 

Con pocas excepciones, un `RandomForestClassifier` tiene todos los hiperparámetros del `DecisionTreeClassifier`, mas todos los hiperparámetros del `BaggingClassifier`para controlar el ensamble.

#### Extra-Trees

Cuando está creciendo un árbol en Random Forest, en cada nodo sólo un subset random de las features es considerado para el splitting. Es posible crear árboles aún más random usando thresholds aleatorios para cada feature, en vez de buscar por los mejores thresholds.

Un bosque de este estilo es llamado ensamble de _Extremely Randomized Trees_ (extra-trees). Es difícil decidir a priori si un extra-tree va a funcionar mejor que un random forest, la mejor forma es probarlo y comparar. La clase que implementa esto en Scikit es `ExtraTreesClassifier`.

#### Feature importance

Otra característica importante de los Random Forest es que pueden medir la importancia relativa de cada feature. Scikit mide la importancia de una feature al mirar en qué cantidad reducen la impureza en promedio los nodos que utilizan esa feature (entre todos los árboles en el bosque). Más precisamente, es un promedio pesado, donde el peso de cada nodo es igual al número de instancias asociadas.

Scikit calcula este score automáticamente para cada feature después de entrenar, y luego escalea los resultados para que la suma de todas las 'importancias' sea igual a 1. Se puede acceder a esta info por la variable `feature_importances_`. Esto es re útil para entender cuáles son las features que importan, en particular si necesitamos hacer feature selection.

### Boosting

_Boosting_ es combinar varios _weak learners_ en un _strong learner_. La idea general de los métodos de boosting es entrenar predictores secuencialmente, cada uno tratando de corregir a su predecesor. Los métodos más populares son _AdaBoost_ (adaptative boosting) y _Gradient Boosting_. 

#### AdaBoost

Para un nuevo predictor, una manera de corregir a su predecesor es prestar más atención a las instancias de entrenamiento que el anterior underfiteó. Esto resulta en nuevos predictores que se concentran más y más en los casos difíciles. Esta es la técnica de AdaBoost.

Una vez que todos los predictores son entrenados, el ensamble hace predicciones similar al bagging o pasting, excepto que los predictores tienen diferentes pesos dependiendo su precisión total sobre el training set pesado.

Inicialmente, los pesos para todas las instancias $w^{(i)}$ son seteados a $1/m$. Un primer predictor es entrenado, y su error rate $r_1$ es calculado sobre el training set:
$$
r_j = \frac{\sum_ {\hat{y_j}^{(i)}\neq y^{(i)}}w^{(i)}}{\sum_{i=1}^m w^{(i)}} \tag{7.1}
$$
El peso de cada predictor $\alpha_j$ es calculado por $\ref{eq:alpha_j_ada}$. Mientras más preciso es el predictor, mayor será su peso. En random guessing, el peso será cercano a cero, mientras que si es menos preciso que random guessing, su peso es negativo.
$$
\alpha_j = \eta\log\frac{1-r_j}{r_j}
\label{eq:alpha_j_ada}\tag{7.2}
$$
Luego, el algoritmo AdaBoost actualiza los pesos de las instancias, lo cual aumenta los pesos de las instancias no clasificadas. Los pesos de las instancias mal clasificadas son multiplicadas por $\exp(\alpha_j)$ Luego, los pesos de las instancias son normalizados. 

Scikit utiliza una versión multiclase de AdaBoost llamada SAMME. Cuando los predictores pueden estimar probabilidades de clase, Scikit utiliza SAMME.R, que generalmente funciona mejor. 

> Si el ensamble AdaBoost está overfiteando el training set, se puede reducir el nro de estimadores o regularizar más fuertemente el estimador base.

#### Gradient Boosting

Otro algoritmo popular de boosting es el _Gradient Boosting_. Como AdaBoost, trabaja añadiendo predictores secuecialmente, cada uno corrigiendo a su predecesor. En vez de cambiar los pesos de las instancias a cada iteración, este método intenta que el nuevo predictor fitee los errores residuales del predictor previo.

En Scikit está implementado el clasificador y el regresor (con árboles de decisión) como `GradientBoostingRegressor` y `GradientBoostingClassifier`.



![](C:\Users\Nico\Dropbox\Machine Learning\Figuras\gr_boosting.png)

Hay una librería que tiene implementada una versión optimizada del Gradient Boosting en Python, llamada XGBoost. Hay que probarla!

### Stacking

El último método de ensamble que vamos a discutir en este capítulo es el _stacking_. Está basado en una idea simple: en vez de usar funciones simples (como hard voting) para sumar las predicciones de todos los predictores del ensamble, por qué no entrenamos un modelo que lo haga? A este modelo le dicen _blender_, y es el que se encarga de tomar todas las predicciones como inputs y hace la predicción final.

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\stacking.png" style="zoom:75%;" />

Es posible entrenar varios blenders de esta manera, para obtener modelos multicapa. 

<img src="C:\Users\Nico\Dropbox\Machine Learning\Figuras\blender_multilayer.png" style="zoom:75%;" />

Scikit no posee una implementación directa del stacking, pero no es difícil de programar, y además existen versiones open source, como [Deslib](https://github.com/scikit-learn-contrib/DESlib).

### Ejercicios

1. Generalmente, combinarlos en un ensamble dará mejores resultados, siempre que los modelos entrenados tengan un algoritmo distinto detrás. Hay aun más posibilidades de mejora si los modelos han sido entranados sobre instancias distintas del training set. 
2. Los hard voting classifiers hacen la agregación de los resultados calculando la moda sobre las predicciones de los modelos del ensamble, mientras que los soft voting classifiers hacen una ponderación sobre los scores (calculados con los métodos `predict_proba()`). Esto a menudo consigue mejores resultados, puesto que los modelos más precisos tienen más peso en la decisión. 
3. En bagging y pasting si, puesto que el entrenamiento de los modelos no es secuencial (random forest es bagging/pasting también). En cambio, el boosting es un ensamble que mejora los modelos de manera secuencial, asique no sería posible paralelizar el trabajo. En el stacking, la posibilidad de paralelizar es parcial, puesto que los predictores en una capa son independientes de las otras, aunque para entrenar una capa, todos los predictores de las capas anteriores tienen que haber sido entrenados previamente. 
4. La evaluación "out-of-bag" permite realizar una validación de cada uno de los predictores del ensamble sin necesidad de separar instancias en un conjunto aparte (test set), ya que utiliza las instancias que no fueron utilizadas en entrenar a ese predictor.
5. Que usa thresholds aleatorios para dividir un nodo cuando entrena el árbol de decisión. Dado que una buena parte del tiempo del entrenamiento del árbol se utiliza en decidir cuál es el mejor threshold, los Extra-trees son más rápidos que los Random Forest regulares.
6. Se pueden modificar los hiperparámetros del modelo indivdual para quitar regularización, o bien se puede aumentar el nro. de estimadores del ensamble.
7. No estoy de acuerdo con el libro, supongo que más que $\eta$ deberíamos estar mirando otros parámetros primero. El libro dice que deberíamos achicarlo, pero también dice en el capítulo que al hacer esto incrementamos el nro. de estimadores del ensamble, y esto seguro aumenta el riesgo de overfit (incluso hay una gráfica que lo demuestra). La solución seguramente sea hacer early stopping, porque probablemente tenemos demasiados predictores.

## Dimensionality Reduction

### Ejercicios

1. Las principales motivaciones para la reducción de dimensiones son:

   - Acelerar un algoritmo de entrenamiento, reduciendo ruido y features redundantes, optimizando la performance.
   - Visualizar los datos y obtener insights sobre las features más importantes.
   - Ahorrar espacio.

   Los principales problemas asociados son:

   - Algo de información se pierde.
   - Puede ser computacionalmente costoso.
   - Añade complejidad a los pipelines.
   - Las features transformadas a menudo son difíciles de interpretar.

2. _"The curse of dimensionality"_  se refiere a los problemas que aparecen al trabajar al aumentar las dimensiones del espacio de las features. Al hacer esto, los vectores tienden a ser cada vez más dispersos, haciendo difícil el entrenamiento de los algoritmos, incrementando el riesgo de overfitting, y dificultando hallar patrones en los datos. Para datos de muchas dimensiones, son necesarios muchos más datos.

3. Una vez que la dimensión del dataset ha sido reducida, es casi imposible revertir perfectamente la operación debido a que algo de la información original siempre se pierde en la transformación. Algunos algoritmos pueden reconstruir el dataset original con bastante similaridad (PCA) mientras otros no (t-SNE).

4. PCA puede ser usado para reducir la dimensionalidad de la mayoría de los datasets, incluso si son no lineales. Si toda la información del dataset es importante, reducir la dimensionalidad con PCA ocasionará mucha pérdida de información.

5. Depende del dataset. Puede ser un número entre 1 y 950. Es necesario graficar la varianza explicada en función del nro. de dimensiones para darnos una diea de la dimensionalidad intrínseca del dataset.

6. Regular PCA es el default, pero sólo funciona si el dataset cabe en la memoria. Si esto no ocurre, Incremental PCA es la solución, pero es más lento que regular PCA, asique si no hay problemas de memoria lo mejor es utilizar el PCA regular. Randomized PCA es útil cuando queremos reducir considerablemente la dimensionalidad, y además el dataset cabe en la memoria. Por último, Kernel PCA es útil en datasets no lineales.

7. Intuitivamente, un algoritmo de reducción de dimensionalidad funciona bien si elimina muchas dimensiones del dataset sin mucha pérdida de información. Una forma de medir esto es aplicar la transormación inversa y medir el error de reconstrucción. No todos los algoritmos proveen una transformación inversa.

8. Tiene sentido usar dos algoritmos diferentes de reducción encadenados. Un ejemplo común es usar PCA para descartar dimensiones irrelevantes, y aplicar LLE luego. El resultado es similar a aplicar sólo LLE, sólo que es más rápido.
