# Python pour le Machine Learning : SKLEARN make_scorer

[Retour README](../README.md)

[toc]

------

## 1. Intro

Je vais vous montrer comment utiliser la fonction make_scorer qui nous vient de Scikit-Learn. Cette fonction est extrêmement utile car elle vous permet de développer vos propres métriques pour les utiliser dans des algorithmes de cross-validation ou des algorithmes comme GridSearchCV. Croyez-moi, développer ses propres métriques pour évaluer son modèle de machine learning est quelque chose qui arrive très souvent dans le monde professionnel. 

En effet, quand vous travaillez avec un client, surtout dans les secteurs industriels, il arrive très souvent que votre client se fiche un peu de votre coefficient de détermination ou de votre erreur quadratique moyenne. Lui, il vous fournit un projet avec un cahier des charges, dans lequel il y a des contraintes que vous devez respecter. Et parmi ces contraintes, on va trouver des mesures de performances qui sont spécifiques au projet sur lequel vous travaillez.

Pour vous donner un exemple typique que j'ai déjà vécu à plusieurs reprises, on pourrait imaginer que notre client nous fournisse le dataset suivant, que j'ai ici généré avec une fonction. 


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
np.random.seed(0)

m = 100
X = np.linspace(0, 4, m).reshape((m,1))
y = 2 + X**1.3 + np.random.randn(m,1)
y = y.ravel()

plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
```


    Text(0, 0.5, 'y')


![png](./.../images/Python_SKLEARN_Fig_000601.png)
    


Les détails ne sont pas très importants, et à partir de ce dataset, on développe un modèle de régression linéaire qui nous donne le résultat suivant que j'ai enregistré sous une variable y_pred.


```python
from sklearn.linear_model import LinearRegression
```


```python
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, c = 'r', lw = 3)
```


    [<matplotlib.lines.Line2D at 0x289902900d0>]


![png](./.../images/Python_SKLEARN_Fig_000602.png)
    


Et comme on l'a vu la dernière fois, ce qu'on pourrait faire, c'est utiliser le module metrics de Scikit-Learn, extraire les erreurs quadratiques moyennes et calculer notre erreur entre y et y_pred. 


```python
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y, y_pred)
```


    0.8739397321740953

Mais cette mesure n'intéresse pas vraiment notre client. Car lui, ce qu'il attend, c'est un modèle qui respecte la condition suivante: toutes les valeurs de prédiction doivent être dans une zone de tolérance de ± 20% par rapport à nos vraies valeurs y. 

C'est quelque chose que vous pouvez tout à fait rencontrer dans un projet avec un client, surtout dans l'industrie automobile, aérospatiale, ou médicale. 


```python
plt.figure(figsize=(9,6))
plt.scatter(X, y)
plt.plot(X, y_pred, c = 'r', lw = 3)
plt.plot(X, y + y*0.2, c = 'g', ls = '--')
plt.plot(X, y - y*0.2, c = 'g', ls = '--')
```


    [<matplotlib.lines.Line2D at 0x289903298e0>]


![png](./.../images/Python_SKLEARN_Fig_000603.png)
    


C'est très fréquent ce genre de choses. Cela voudrait dire que sur nos données, il faut que nos prédictions soient toujours entre les deux marges vertes.

## 2. Fil conducteur

Donc, pour réussir à développer un modèle et l'évaluer avec les métriques de notre client, ça va être assez simple. On va devoir suivre le fil conducteur. 
![Python_Seaborn_Fig_000085.png](../images/Python_Seaborn_Fig_000085.png)

Tout d'abord, il faut se rappeler qu'une métrique dans Scikit-Learn, c'est juste une fonction que vous pouvez très bien définir vous-même. Cette fonction prend deux arguments : un argument y et un argument y_pred. Mais une fois qu'on a défini cette fonction, la question qu'on va se poser est : comment est-ce qu'on peut l'utiliser dans une cross-validation ou bien dans un algorithme d'optimisation comme GridSearchCV? 

Eh bien, pour ça, il faut transformer notre fonction en un scoreur en utilisant la fonction make_scorer. Après quoi, on peut l'utiliser dans des algorithmes comme GridSearchCV.

Donc, pour commencer, je vous propose de créer une fonction qu'on va appeler, par exemple, custom_metric pour une mesure personnalisée, la mesure de notre client, dans laquelle on va faire passer deux arguments : un argument y et un argument y_pred. 

Dans cette fonction, on va calculer la proportion de nos prédictions qui rentrent dans notre marge des 20%. 

On va utiliser cette mesure de y ± 20% en écrivant quelque chose comme : lorsque y_pred est inférieur à y + 20% et lorsque y_pred est supérieur à y - 20% de y. Tout ça va nous donner un tableau avec une opération logique "et" dont on va faire la somme de tous les éléments vrais du tableau. 

À ce stade, on peut simplement retourner ce résultat.


```python
def custom_metric(y, y_pred):
    return np.sum((y_pred < y + y * 0.2) & (y_pred > y - y * 0.2))
```

En utilisant notre fonction custom_metric sur y et y_pred, cela nous donne donc 63 points. 


```python
custom_metric(y, y_pred)
```


    63

Sachant que j'ai créé 100 points ici, on a donc 63 points qui rentrent dans le score de notre client. 

Pour aller plus loin, ce qu'on va faire, c'est qu'on va obtenir une proportion en divisant cela par le nombre d'éléments qu'il y a dans y, c'est-à-dire 100. 


```python
def custom_metric(y, y_pred):
    return np.sum((y_pred < y + y * 0.2) & (y_pred > y - y * 0.2))/y.size
```


```python
custom_metric(y, y_pred)
```


    0.63

Donc, on va obtenir un pourcentage, c'est-à-dire 0,63, donc 63% de nos prédictions sont conformes à la métrique de notre client. 

Maintenant, on pourrait se dire: "Chouette! Il n'y a plus qu'à passer cette fonction custom_metric dans cross-validation ou GridSearchCV." 

Mais comme on vient de le voir, il faut d'abord passer ça dans notre fonction make_scorer.

Donc, on va importer cette fonction depuis le module metrics de scikit-learn. Pour cela, on utilise : from sklearn.metrics import make_scorer. 

Cette fonction, comme vous pouvez le voir, est très simple à utiliser. Il suffit de passer la fonction que l'on souhaite transformer en scoreur, ainsi qu'un autre argument qui définit si un score élevé est préférable ou non. Tout ceci est précisé avec l'argument greater_is_better, qui peut être égal à True ou False.
![Python_Seaborn_Fig_000086.png](../images/Python_Seaborn_Fig_000086.png)


```python
from sklearn.metrics import make_scorer
```

Nous allons créer quelque chose qui s'appelle custom_score, qui sera égal à make_scorer de notre custom_metric. Il suffit ensuite de préciser que greater_is_better est égal à True car, avec la métrique que nous avons créée, plus on s'approche de 100%, meilleur est notre modèle.


```python
custom_score = make_scorer(custom_metric, greater_is_better=True)
```

À présent, notre scoreur est prêt. On peut l'utiliser dans une fonction de cross-validation. Pour cela, nous importons cross_val_score depuis sklearn.model_selection.

Ensuite, on utilise cross_val_score avec notre modèle linéaire, en passant X, y et en spécifiant trois splits de cross-validation. Il ne reste plus qu'à préciser que scoring est égal à notre custom_score et non pas notre custom_metric - la différence est cruciale. En effet, l'utilisation directe de custom_metric engendrerait une erreur.


```python
from sklearn.model_selection import cross_val_score
```


```python
cross_val_score(LinearRegression(), X, y, cv = 3, scoring = custom_score)
```


    array([0.08823529, 0.42424242, 0.3030303 ])

Lorsqu'on exécute cela, on obtient des scores plutôt bas. Cependant, c'est attendu car nous développons un modèle de régression linéaire basique. Voyons comment améliorer ce modèle avec un autre type d'algorithme et obtenir de meilleurs scores en utilisant GridSearchCV.

Je vous propose d'importer le module de Support Vector Machine (SVM) - SVR. Nous allons créer un modèle simple avec un kernel linéaire et un degré égal à trois. Puis, nous définirons une grille d'hyperparamètres gamma allant de 0,1 à 1 par pas de 0,005. 

Il nous reste à importer GridSearchCV depuis le même module sklearn.model_selection. Avec GridSearchCV, nous allons créer une grille dans laquelle nous passerons notre modèle, nos paramètres, définirons trois splits de cross-validation, et utiliserons notre scoreur custom_score. 

Une fois cela fait, on entraîne notre grille avec .fit(X, y).


```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
```


```python
model = SVR(kernel = 'rbf', degree = 3)
params = {'gamma' : np.arange(0.1, 1, 0.05)}

grid = GridSearchCV(model, param_grid = params, cv = 3, scoring = custom_score )
grid.fit (X, y)
```

Une fois notre modèle entraîné, il ne reste plus qu'à enregistrer notre meilleur modèle dans une variable best_model en utilisant grid.best_estimator_. 

Pour évaluer notre modèle, on utilise best_model.predict(X) pour obtenir y_pred, que l'on passe à notre custom_metric avec y pour obtenir une performance de 64%.


```python
best_model = grid.best_estimator_
```


```python
y_pred = best_model.predict(X)
custom_metric(y, y_pred)
```


    0.64

Si on veut visualiser ce nouveau modèle, on peut simplement copier-coller le code que nous avions précédemment pour afficher les résultats.


```python
plt.figure(figsize=(9,6))
plt.scatter(X, y)
plt.plot(X, y_pred, c = 'r', lw = 3)
plt.plot(X, y + y*0.2, c = 'g', ls = '--')
plt.plot(X, y - y*0.2, c = 'g', ls = '--')
```


    [<matplotlib.lines.Line2D at 0x289903def70>]


![png](./../images/Python_SKLEARN_Fig_000604.png)
    


Pour résumer, grâce à la fonction make_scorer, nous avons pu convertir la métrique fournie par notre client en un scoreur utilisable avec GridSearchCV. 

Avec ce scoreur, nous avons entraîné plusieurs versions de notre modèle et sélectionné celle qui offrait la meilleure performance selon la métrique attendue par notre client.

## 3. Exercice : création d'un scoreur personnalisé avec make_scorer

**Objectif** : Dans cet exercice, vous allez créer votre propre fonction pour calculer la Root Mean Square Error (RMSE) que nous avons introduite dans la dernière vidéo. Ensuite, vous la convertirez en scoreur en utilisant la fonction make_scorer de scikit-learn, et vous l'utiliserez pour évaluer vos modèles.

**Étapes** :

1. Création de la fonction RMSE : Commencez par définir une fonction qui calcule la RMSE. Elle prendra en entrée les valeurs réelles et les prédictions, et renverra la RMSE.

    Astuce : La formule de la RMSE est $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{\text{réel},i} - y_{\text{prédit},i})^2}
$

2. Conversion de la fonction RMSE en scoreur : À l'aide de la fonction make_scorer de scikit-learn, convertissez votre fonction RMSE en un scoreur.

    Astuce : Gardez à l'esprit que pour la RMSE, des valeurs plus basses sont meilleures. Ainsi, lors de l'utilisation de make_scorer, définissez greater_is_better à False.

3. Évaluation de vos modèles :
    Appliquez ce scoreur personnalisé pour évaluer vos différents modèles de machine learning. Par exemple, vous pouvez l'utiliser avec la fonction cross_val_score pour évaluer la performance de votre modèle avec validation croisée.

Lorsque vous aurez terminé cet exercice, vous aurez non seulement renforcé votre compréhension de la RMSE et de la création de scoreurs personnalisés, mais vous serez également en mesure d'utiliser votre scoreur personnalisé pour évaluer la performance de vos modèles de machine learning.

