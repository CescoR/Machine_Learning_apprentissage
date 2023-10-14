# Python pour le Machine Learning : Pandas / Time Series

[Retour README](../README.md)

[toc]

------

## 1. Intro

Nous avons ici l'une des techniques de trading les plus populaires. 

![Python_Pandas_Fig](../images/Python_Pandas_Fig_000016.png)

Pourtant, je vais vous expliquer pourquoi vous ne devez jamais l'utiliser sur du Bitcoin, au risque de perdre tout votre argent.Nous allons voir comment utiliser $Pandas$ pour travailler sur des problèmes de $time\ series$. Cela va typiquement inclure l'étude du climat, l'analyse de la bourse ou tout autre phénomène qui évolue avec le temps. 

En réalité, Pandas a même été spécifiquement développé pour aborder ce type de problème, donc nous y trouverons une multitude de fonctionnalités pour travailler sur des time series.

## 2. DateTimeIndex

Pour commencer, nous allons nous rendre sur le site Yahoo Finance pour télécharger des données sur le Bitcoin. 

Pour ce faire, c'est très simple : il suffit de rechercher "BTC-EUR", puis dans l'historique, tout ce que nous avons à faire est de cliquer sur "Maximum". Ainsi, nous obtenons la date la plus ancienne, et nous cliquons ensuite ici pour télécharger nos données : BTC-EUR.csv

Comme vous le savez, nous importons Pandas, puis nous créons un DataFrame "bitcoin". Nous utilisons la fonction $read\_csv$ car cette fois-ci, nous disposons d'un fichier CSV. 

Nous utilisons ensuite la fonction $head$ pour observer le début de notre DataFrame. 

Si nous souhaitons observer l'évolution d'une des valeurs, comme la valeur "Close", nous utiliserons la fonction $plot$. Nos données sont alors chargées. Cependant, il est regrettable que sur ce graphique, nous ne trouvions pas de date en abscisse. Nous aimerions voir, par exemple, 2017, 2018, 2019.


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
bitcoin = pd.read_csv('datasets/BTC-EUR.csv')
bitcoin.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-09-30</td>
      <td>296.389008</td>
      <td>309.562134</td>
      <td>294.327698</td>
      <td>306.417480</td>
      <td>306.417480</td>
      <td>27484400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-10-01</td>
      <td>306.799957</td>
      <td>310.224304</td>
      <td>301.894867</td>
      <td>303.949768</td>
      <td>303.949768</td>
      <td>20782347</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-10-02</td>
      <td>304.245300</td>
      <td>304.692535</td>
      <td>294.376831</td>
      <td>296.054932</td>
      <td>296.054932</td>
      <td>17189755</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-10-03</td>
      <td>296.140961</td>
      <td>299.354065</td>
      <td>285.944061</td>
      <td>287.264862</td>
      <td>287.264862</td>
      <td>24691330</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-10-04</td>
      <td>287.568512</td>
      <td>291.240112</td>
      <td>260.396301</td>
      <td>262.777466</td>
      <td>262.777466</td>
      <td>37743907</td>
    </tr>
  </tbody>
</table>

```python
bitcoin['Close'].plot(figsize=(9, 6))
```


    <Axes: >


![png](../images/Python_Pandas_Fig_000040.png)
    


Ne vous inquiétez pas, c'est normal. Pour l'instant, nous n'avons pas indiqué à Pandas que nous voulions travailler sur une base temporelle. Donc, dans notre DataFrame "bitcoin", nous avons un index par défaut (0,1,2,3) que nous retrouvons à la fois dans notre DataFrame (dans la colonne tout à gauche) et sur notre graphique Matplotlib. 


```python
bitcoin.index
```


    RangeIndex(start=0, stop=3257, step=1)

Pour changer cela et débloquer les fonctionnalités de time series, nous devons définir un nouveau type d'index : le DateTimeIndex. Pour ce faire, nous écrivons deux choses dans notre fonction $read\_csv$. 

Pour commencer, nous indiquons que la colonne d'index doit être celle qui contient nos différentes dates. Ce changement est visible dans notre DataFrame et sur notre graphique, car nous voyons maintenant nos différentes dates. 

Mais ce n'est pas encore fini, car il faut désormais indiquer à Pandas que cette colonne d'index doit être interprétée comme une date. Pour cela, nous écrivons dans notre fonction read_csv : $parse\_dates=True$.
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000017.png)

Maintenant, si nous exécutons tout notre code, nous obtenons un index de type DateTimeIndex.


```python
bitcoin = pd.read_csv('datasets/BTC-EUR.csv', index_col='Date', parse_dates=True)
bitcoin.head()

bitcoin['Close'].plot(figsize=(9, 6))
```


    <Axes: xlabel='Date'>

![png](./../images/Python_Pandas_Fig_000041.png)
    Nous sommes prêts à commencer notre analyse de time series. Avec ce nouvel index, nous pourrons réaliser toutes les opérations que vous pourriez imaginer avec des dates et le temps, car Pandas comprend les notions de jours, semaines, mois et années. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000018.png)

Par exemple, pour observer l'évolution du Bitcoin uniquement en 2019, nous écrivons $bitcoin['2019']$ pour la colonne "Close" et nous l'affichons avec Matplotlib. Cela nous donne l'évolution du Bitcoin en 2019. 


```python
bitcoin.loc['2019']['Close'].plot(figsize=(9, 6))
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000042.png)
    


Si nous voulons voir l'évolution du Bitcoin en septembre 2019, nous écrivons $bitcoin['2019-09']$. Nous observons que le Bitcoin a subi une chute importante ce mois-là. Comme mentionné précédemment, nous pouvons aussi faire du slicing sur des dates. 

Par exemple, pour voir l'évolution du Bitcoin entre 2017 et 2019, nous écrivons $bitcoin['2017':'2019']$.


```python
bitcoin.loc['2015':'2021'].plot(figsize=(9, 6))
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000043.png)
    


De plus, cela fonctionne également avec la méthode $lo$c. Donc si vous préférez utiliser cette méthode, c'est à votre choix, vous obtiendrez le même résultat. 

Si vous avez des fichiers avec des dates formatées différemment, ne vous inquiétez pas, car Pandas s'adapte à de nombreux formats. 

Vous pouvez utiliser des tirets, des slashes, des points, des virgules, des espaces, etc. Si Pandas ne parvient pas à convertir une date, vous pouvez effectuer la conversion vous-même avec la fonction $to\_datetime$.
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000019.png)

Enfin, une dernière chose à propos de cet index DateTime est que nous pouvons non seulement y inclure des dates (année, mois, jour), mais aussi des heures, minutes, secondes et même millisecondes. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000020.png)

Ainsi, vous pourriez avoir un dataset avec des températures pour chaque minute de la journée et analyser ce dataset sans problème. 

Vous pourriez même avoir une machine qui collecte des données toutes les millisecondes et analyser ces données avec les mêmes fonctions que nous verrons dans la suite de cette vidéo.

## 3. Pandas resample() et agg()

Je vais vous présenter la première fonction qui est très utile : c'est la fonction $resample$. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000022.png)

Cette fonction permet de regrouper nos données selon une fréquence temporelle. Par exemple, pour l'année 2019, voici ce que l'on obtient lorsque l'on regroupe nos données par mois.


```python
bitcoin.loc['2019','Close'].resample('M').plot()
```


    Date
    2019-01-31    Axes(0.125,0.11;0.775x0.77)
    2019-02-28    Axes(0.125,0.11;0.775x0.77)
    2019-03-31    Axes(0.125,0.11;0.775x0.77)
    2019-04-30    Axes(0.125,0.11;0.775x0.77)
    2019-05-31    Axes(0.125,0.11;0.775x0.77)
    2019-06-30    Axes(0.125,0.11;0.775x0.77)
    2019-07-31    Axes(0.125,0.11;0.775x0.77)
    2019-08-31    Axes(0.125,0.11;0.775x0.77)
    2019-09-30    Axes(0.125,0.11;0.775x0.77)
    2019-10-31    Axes(0.125,0.11;0.775x0.77)
    2019-11-30    Axes(0.125,0.11;0.775x0.77)
    2019-12-31    Axes(0.125,0.11;0.775x0.77)
    Freq: M, Name: Close, dtype: object


![png](./../images/Python_Pandas_Fig_000044.png)
    


On observe différents paquets de données : on a des données pour janvier, pour février, pour mars, etc. 

Et qui dit groupe de données, dit statistique. On pourrait chercher la moyenne pour chaque mois ou bien l'écart type pour chaque mois. Vous avez saisi l'idée.

Si on choisit la moyenne, on obtiendra le graphique suivant. 


```python
bitcoin.loc['2019','Close'].resample('M').mean().plot()
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000045.png)
    


Voici donc l'évolution du bitcoin en 2019 en faisant la moyenne sur chaque mois. C'est simple à comprendre et on peut faire la même chose en suivant les semaines en utilisant W (pour week) ou bien on peut le faire toutes les deux semaines en écrivant "2W". On obtient ainsi la moyenne du bitcoin toutes les deux semaines.


```python
bitcoin.loc['2019','Close'].resample('W').mean().plot()
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000046.png)
    

```python
bitcoin.loc['2019','Close'].resample('2W').mean().plot()
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000047.png)
    


Juste avant, j'ai parlé de l'écart type. Il serait intéressant de voir à quel point le bitcoin était volatile en 2019. On pourrait écrire std et observer les résultats. 


```python
bitcoin.loc['2023','Close'].resample('2W').std().plot()
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000048.png)
    


On voit que le bitcoin est très volatile en ce moment, alors qu'il était plutôt stable en début d'année. Avec la fonction resample, on obtient des résultats intéressants.

Dans la pratique, on aime bien afficher toutes ces courbes sur un seul et même graphique. 

Ça peut sembler impressionnant, mais en réalité, c'est simple. On affiche nos données brutes, puis on affiche la moyenne de chaque mois. 

Avec quelques options matplotlib, on peut créer un style agréable.


```python
plt.figure(figsize=(12, 8))
bitcoin.loc['2023', 'Close'].plot()
bitcoin.loc['2023', 'Close'].resample('M').mean().plot(label='moyenne par mois', lw=3, ls=':', alpha=0.8)
bitcoin.loc['2023', 'Close'].resample('W').mean().plot(label='moyenne par semaine', lw=2, ls='--', alpha=0.8)
plt.legend()
plt.show()
```

  ![png](./../images/Python_Pandas_Fig_000049.png)
 À présent, une fonction qu'on utilise très souvent en complément de la fonction $resample$ est la fonction $aggregate$ (ou $agg$). 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000024.png)

Cette fonction nous permet de rassembler, dans un seul tableau, plusieurs statistiques que l'on aimerait appliquer après avoir utilisé $resample$. 

Par exemple, on reprend notre échantillonnage pour les semaines, puis on écrit $agg$, entre crochets, on indique les différentes statistiques que l'on veut, comme la moyenne, l'écart type, le minimum et le maximum. 


```python
m = bitcoin['Close'].resample('W').agg(['mean', 'std', 'min', 'max'])

plt.figure(figsize=(12, 8))
m['mean']['2023'].plot(label='moyenne par semaine')
plt.fill_between(m.index, m['max'], m['min'], alpha=0.2, label='min-max par semaine')

plt.legend()
plt.show()
```

  ![png](./../images/Python_Pandas_Fig_000050.png)
Cela nous donne un tableau contenant nos différentes statistiques. 

Encore une fois, avec un peu de manipulation sur matplotlib, on peut créer des graphiques très intéressants.

J'ai enregistré mon tableau aggregate dans une variable $m$. 

Ensuite, je crée un graphique où j'affiche la moyenne de ce tableau pour chaque semaine. J'utilise également la fonction $fill\_between$ de $matplotlib$ pour créer une zone d'incertitude entre le maximum et le minimum pour chaque semaine. C'est vraiment aussi simple que ça. Je pense que vous avez compris la puissance de cette fonction $resample$ ; elle est extrêmement pratique.

Maintenant, nous allons voir comment calculer des moyennes mobiles ($moving\ averages$) avec pandas. 

Si vous ne savez pas ce qu'est une moyenne mobile, c'est une technique qui permet de calculer une moyenne sur une fenêtre glissante de valeurs. Plutôt que de faire la moyenne de toutes les valeurs, comme celles que nous aurions pour 2019 par exemple, nous ferons la moyenne sur une fenêtre définie de valeurs.

## 4. Pandas rolling(): Moyenne mobile, pandas ewm() : Exponential weigthed function

On définit une fenêtre, par exemple de sept jours. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000025.png)

On calcule la moyenne, puis on se décale d'un jour dans cette fenêtre et on recalcule la moyenne, et ainsi de suite. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000026.png)
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000027.png)
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000028.png)

Donc, en fait, on utilise une fonction qui permet d'effectuer un roulement. Dans pandas, on appelle cette fonction $rolling$. À l'intérieur de cette fonction qui effectue le roulement, on définira une taille pour notre fenêtre. Par exemple, on écrit $rolling$ et à l'intérieur, on spécifie $window = 7$. Cela nous donne une fonction de roulement. 


```python
bitcoin.loc['2023','Close'].rolling(window=7)
```


    Rolling [window=7,center=False,axis=0,method=single]

Mais ce que nous voulons, c'est utiliser cette fonction pour calculer une moyenne, ou peut-être une variance, ou même un écart type. 

Il y a tellement de possibilités. Donc, par exemple, on va calculer la moyenne, et on va l'afficher avec matplotlib, obtenant ainsi un certain résultat. 

```python
bitcoin.loc['2023','Close'].rolling(window=7).mean().plot()
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000051.png)
    


En superposant cela sur notre signal original, on obtient le graphique suivant. 


```python
plt.figure(figsize=(12, 8))
bitcoin.loc['2019', 'Close'].plot()
bitcoin.loc['2019', 'Close'].rolling(window=7).mean().plot(label='moving average', lw=3, ls=':', alpha=0.8)
plt.legend()
plt.show()
```

   ![png](./../images/Python_Pandas_Fig_000052.png)
Si vous regardez ce graphique de plus près, par exemple pour le mois de septembre. 


```python
plt.figure(figsize=(12, 8))
bitcoin.loc['2019-09', 'Close'].plot()
bitcoin.loc['2019-09', 'Close'].rolling(window=7).mean().plot(label='moving average', lw=3, ls=':', alpha=0.8)
plt.legend()
plt.show()
```

   ![png](./../images/Python_Pandas_Fig_000053.png)
Vous remarquerez quelque chose d'ennuyeux. Notre moyenne mobile ne commence pas dès le début de notre signal. C'est simplement parce que nous avons pris une fenêtre de sept jours, regardé tous les nombres, et finalement placé la moyenne à la fin. 

Mais ce que l'on pourrait faire est de placer cette moyenne au centre de cette fenêtre en écrivant center=True. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000029.png)

Cela ne change pas le calcul de la moyenne, mais redécale toutes les valeurs au centre de leur fenêtre, comme vous pouvez le voir. 

Pour comparer les deux méthodes, il suffirait de copier-coller cette ligne et, dans la seconde ligne, écrire $center=True$. 


```python
plt.figure(figsize=(12, 8))
bitcoin.loc['2019-09', 'Close'].plot()
bitcoin.loc['2019-09', 'Close'].rolling(window=7).mean().plot(label='non centre', lw=3, ls=':', alpha=0.8)
bitcoin.loc['2019-09', 'Close'].rolling(window=7, center=True).mean().plot(label='centre', lw=3, ls=':', alpha=0.8)
plt.legend()
plt.show()
```


  ![png](./../images/Python_Pandas_Fig_000054.png)
    


Le résultat serait légèrement différent, et peut-être préférable. 

Mais on pourrait se dire que la moyenne mobile n'est peut-être pas la meilleure technique pour cette manipulation. Et vous auriez raison. Il y a une autre technique, un peu plus adaptée, appelée moyenne mobile exponentielle en français. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000030.png)


Elle est définie par la formule suivante où ${x}_t$ est la valeur de $x$ à un instant $t$, c'est-à-dire la valeur à une date précise, et α est un facteur de lissage.

Grâce à cette formule, les valeurs perdent progressivement du poids avec le temps. C'est précisément notre facteur de lissage, $\alpha$, qui est compris entre 0 et 1, qui permet de définir la façon dont les valeurs vont perdre de leur importance avec le temps. 

Cette formule peut aussi être exprimée de cette manière, ce qui met en relief la fonction $ewm$ que l'on trouve dans pandas. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000032.png)

Dans cette fonction, tout ce que nous avons à faire est de définir une valeur pour notre paramètre alpha. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000033.png)

Cependant, il n'y a ni paramètre window ni paramètre center dans cette fonction. Tout ce que l'on spécifie, c'est alpha. Nous pourrions essayer un paramètre alpha égal à, disons, 0,6 pour commencer.

 Ainsi, en affichant le tout, on obtient le graphique suivant, dans lequel on observe que $ewm$ suit un peu mieux la tendance du Bitcoin que nos moyennes mobiles précédentes.


```python
plt.figure(figsize=(12, 8))
bitcoin.loc['2019-09', 'Close'].plot()
bitcoin.loc['2019-09', 'Close'].rolling(window=7).mean().plot(label='non centre', lw=3, ls=':', alpha=0.8)
bitcoin.loc['2019-09', 'Close'].rolling(window=7, center=True).mean().plot(label='centre', lw=3, ls=':', alpha=0.8)
bitcoin.loc['2019-09', 'Close'].ewm(alpha=0.6).mean().plot(label='ewm', lw=3, ls=':', alpha=0.6)
plt.legend()
plt.show()  
```

  ![png](./../images/Python_Pandas_Fig_000055.png)
    Maintenant, en utilisant une simple boucle $for$, nous pouvons facilement comparer différentes valeurs de $\alpha$ pour voir ce que cela donne pour notre fonction exponentielle. 

En l'occurrence, tout ce que j'ai eu à faire, c'est de créer un tableau $numpy$ avec la fonction $arange$ et d'itérer à travers ce tableau pour des valeurs entre 0,2 et 1, d'intégrer ces différentes valeurs dans ma fonction $ewm$ et, accessoirement, d'ajouter une légende avec la fonction $format$ qui me permet d'inclure ma valeur d' $\alpha$ dans cette légende.


```python
plt.figure(figsize=(12, 8))
bitcoin.loc['2019-09', 'Close'].plot()
for i in np.arange(0.2, 1, 0.2):
    bitcoin.loc['2019-09', 'Close'].ewm(alpha=i).mean().plot(label=f'ewm {i}', ls='--', alpha=0.8)
plt.legend()
plt.show()
```

  ![png](./../images/Python_Pandas_Fig_000056.png)
Voilà, nous avons passé en revue les fonctions les plus utiles pour travailler avec des séries temporelles. À présent, je vais vous montrer comment comparer deux séries en les assemblant ensemble. 

Dans ce contexte, nous allons essayer de comparer le Bitcoin avec une autre crypto-monnaie très connue, l'Ethereum.

## 5. Pandas merge() : inner, outer, etc…

Donc, nous revoilà sur Yahoo Finance. Cette fois-ci, nous allons rechercher les données pour Ethereum. 

Cependant, il y a un petit souci : contrairement à Bitcoin, les données historiques d'Ethereum ne commencent qu'en 2015, car Ethereum est plus récent que Bitcoin. 

Lorsque nous voudrons fusionner ces deux ensembles de données, cela pourrait poser problème. Typiquement, quand on a deux tableaux numériques de dimensions différentes et qu'on utilise la méthode $concat$ dessus, cela n'est pas possible. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000034.png)

Mais avec Pandas, il n'y a aucun problème. On peut assembler des dataframes et des séries avec plusieurs méthodes, notamment $inner$ et $outer$, qui ressemblent aux méthodes SQL.
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000035.png)

Ne vous inquiétez pas, nous allons voir cela en détail. Mais pour l'instant, téléchargeons nos données. 

Nous allons les importer dans Pandas comme auparavant, en écrivant $index\_col=date$ et $parse\_dates=True$. 

Cette fois-ci, j'ai créé un dataframe Ethereum. En regardant son évolution en 2019, on observe une tendance similaire à celle du Bitcoin.


```python
ethereum = pd.read_csv('datasets/ETH-EUR.csv', index_col='Date', parse_dates=True)
ethereum.loc['2019']['Close'].plot()
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000057.png)
    


Pour fusionner ces deux dataframes, nous allons utiliser $pd.merge()$. Nous passerons nos dataframes Bitcoin et Ethereum en arguments, définirons une colonne commune (dans ce cas, "date") pour effectuer la fusion, et enfin choisirons la méthode de fusion. 

Il existe plusieurs méthodes : $inner$, $outer$, $left$, $right$, etc. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000036.png)

Si nous utilisons $inner$, nous obtenons un dataframe où les colonnes "open" du Bitcoin et d'Ethereum sont respectivement nommées $Open\_x$ et $Open\_y$. 

Pour rendre ces noms plus clairs, nous pouvons utiliser l'argument suffixes, comme $suffixes=('\_btc', '\_eth')$.


```python
pd.merge(bitcoin, ethereum, on='Date', how='inner', suffixes=('_btc','_eth'))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_btc</th>
      <th>High_btc</th>
      <th>Low_btc</th>
      <th>Close_btc</th>
      <th>Adj Close_btc</th>
      <th>Volume_btc</th>
      <th>Open_eth</th>
      <th>High_eth</th>
      <th>Low_eth</th>
      <th>Close_eth</th>
      <th>Adj Close_eth</th>
      <th>Volume_eth</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-30</th>
      <td>8358.750000</td>
      <td>9118.615234</td>
      <td>7735.415039</td>
      <td>8601.109375</td>
      <td>8601.109375</td>
      <td>6984946875</td>
      <td>363.833099</td>
      <td>392.427094</td>
      <td>338.259155</td>
      <td>375.789215</td>
      <td>375.789215</td>
      <td>1599462094</td>
    </tr>
    <tr>
      <th>2017-12-01</th>
      <td>8571.692383</td>
      <td>9279.042969</td>
      <td>8138.558105</td>
      <td>9231.726563</td>
      <td>9231.726563</td>
      <td>5705374608</td>
      <td>374.188110</td>
      <td>397.517883</td>
      <td>359.442780</td>
      <td>392.413147</td>
      <td>392.413147</td>
      <td>1049608828</td>
    </tr>
    <tr>
      <th>2017-12-02</th>
      <td>9233.998047</td>
      <td>9521.575195</td>
      <td>9172.427734</td>
      <td>9314.997070</td>
      <td>9314.997070</td>
      <td>4322062491</td>
      <td>392.674744</td>
      <td>400.571136</td>
      <td>384.097076</td>
      <td>389.813263</td>
      <td>389.813263</td>
      <td>793716868</td>
    </tr>
    <tr>
      <th>2017-12-03</th>
      <td>9321.810547</td>
      <td>10004.808594</td>
      <td>9154.656250</td>
      <td>9550.607422</td>
      <td>9550.607422</td>
      <td>5573810526</td>
      <td>390.028595</td>
      <td>407.334839</td>
      <td>380.827637</td>
      <td>392.925934</td>
      <td>392.925934</td>
      <td>835490040</td>
    </tr>
    <tr>
      <th>2017-12-04</th>
      <td>9544.028320</td>
      <td>9823.207031</td>
      <td>9341.481445</td>
      <td>9823.207031</td>
      <td>9823.207031</td>
      <td>5167615957</td>
      <td>393.095459</td>
      <td>400.385406</td>
      <td>382.154785</td>
      <td>396.228210</td>
      <td>396.228210</td>
      <td>847349807</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-08-26</th>
      <td>24228.587891</td>
      <td>24234.349609</td>
      <td>23941.740234</td>
      <td>24129.414063</td>
      <td>24129.414063</td>
      <td>11492414331</td>
      <td>1537.474609</td>
      <td>1543.697021</td>
      <td>1515.694336</td>
      <td>1531.206421</td>
      <td>1531.206421</td>
      <td>4999541315</td>
    </tr>
    <tr>
      <th>2023-08-27</th>
      <td>24129.011719</td>
      <td>24184.732422</td>
      <td>24070.322266</td>
      <td>24093.095703</td>
      <td>24093.095703</td>
      <td>5590389230</td>
      <td>1531.198975</td>
      <td>1532.913330</td>
      <td>1522.663086</td>
      <td>1525.066162</td>
      <td>1525.066162</td>
      <td>2244790090</td>
    </tr>
    <tr>
      <th>2023-08-28</th>
      <td>24092.890625</td>
      <td>24238.449219</td>
      <td>24052.923828</td>
      <td>24163.882813</td>
      <td>24163.882813</td>
      <td>6403428781</td>
      <td>1525.033936</td>
      <td>1537.072021</td>
      <td>1524.691895</td>
      <td>1535.163696</td>
      <td>1535.163696</td>
      <td>2430369670</td>
    </tr>
    <tr>
      <th>2023-08-29</th>
      <td>24163.810547</td>
      <td>24243.509766</td>
      <td>23932.878906</td>
      <td>24115.007813</td>
      <td>24115.007813</td>
      <td>10163610214</td>
      <td>1535.095947</td>
      <td>1535.811646</td>
      <td>1506.128784</td>
      <td>1526.422852</td>
      <td>1526.422852</td>
      <td>4485247941</td>
    </tr>
    <tr>
      <th>2023-08-30</th>
      <td>25512.494141</td>
      <td>25523.169922</td>
      <td>24923.080078</td>
      <td>24951.197266</td>
      <td>24951.197266</td>
      <td>15584089088</td>
      <td>1591.720703</td>
      <td>1591.753662</td>
      <td>1553.628052</td>
      <td>1559.392578</td>
      <td>1559.392578</td>
      <td>4778068992</td>
    </tr>
  </tbody>
</table>
<p>2100 rows × 12 columns</p>
Avec la méthode $inner$, nous constatons que nos données commencent en 2017. Cela signifie que nous avons écarté toutes les données de Bitcoin datant d'avant 2017. 

Si, par contre, nous utilisons $outer$, alors le dataframe résultant contiendra toutes les données, y compris celles du Bitcoin à partir de 2014. Bien sûr, pour les lignes correspondant à des dates avant 2017, il n'y aura pas de données pour Ethereum.


```python
pd.merge(bitcoin, ethereum, on='Date', how='outer', suffixes=('_btc','_eth'))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_btc</th>
      <th>High_btc</th>
      <th>Low_btc</th>
      <th>Close_btc</th>
      <th>Adj Close_btc</th>
      <th>Volume_btc</th>
      <th>Open_eth</th>
      <th>High_eth</th>
      <th>Low_eth</th>
      <th>Close_eth</th>
      <th>Adj Close_eth</th>
      <th>Volume_eth</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-09-30</th>
      <td>296.389008</td>
      <td>309.562134</td>
      <td>294.327698</td>
      <td>306.417480</td>
      <td>306.417480</td>
      <td>27484400</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-10-01</th>
      <td>306.799957</td>
      <td>310.224304</td>
      <td>301.894867</td>
      <td>303.949768</td>
      <td>303.949768</td>
      <td>20782347</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-10-02</th>
      <td>304.245300</td>
      <td>304.692535</td>
      <td>294.376831</td>
      <td>296.054932</td>
      <td>296.054932</td>
      <td>17189755</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-10-03</th>
      <td>296.140961</td>
      <td>299.354065</td>
      <td>285.944061</td>
      <td>287.264862</td>
      <td>287.264862</td>
      <td>24691330</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-10-04</th>
      <td>287.568512</td>
      <td>291.240112</td>
      <td>260.396301</td>
      <td>262.777466</td>
      <td>262.777466</td>
      <td>37743907</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-08-26</th>
      <td>24228.587891</td>
      <td>24234.349609</td>
      <td>23941.740234</td>
      <td>24129.414063</td>
      <td>24129.414063</td>
      <td>11492414331</td>
      <td>1537.474609</td>
      <td>1543.697021</td>
      <td>1515.694336</td>
      <td>1531.206421</td>
      <td>1531.206421</td>
      <td>4.999541e+09</td>
    </tr>
    <tr>
      <th>2023-08-27</th>
      <td>24129.011719</td>
      <td>24184.732422</td>
      <td>24070.322266</td>
      <td>24093.095703</td>
      <td>24093.095703</td>
      <td>5590389230</td>
      <td>1531.198975</td>
      <td>1532.913330</td>
      <td>1522.663086</td>
      <td>1525.066162</td>
      <td>1525.066162</td>
      <td>2.244790e+09</td>
    </tr>
    <tr>
      <th>2023-08-28</th>
      <td>24092.890625</td>
      <td>24238.449219</td>
      <td>24052.923828</td>
      <td>24163.882813</td>
      <td>24163.882813</td>
      <td>6403428781</td>
      <td>1525.033936</td>
      <td>1537.072021</td>
      <td>1524.691895</td>
      <td>1535.163696</td>
      <td>1535.163696</td>
      <td>2.430370e+09</td>
    </tr>
    <tr>
      <th>2023-08-29</th>
      <td>24163.810547</td>
      <td>24243.509766</td>
      <td>23932.878906</td>
      <td>24115.007813</td>
      <td>24115.007813</td>
      <td>10163610214</td>
      <td>1535.095947</td>
      <td>1535.811646</td>
      <td>1506.128784</td>
      <td>1526.422852</td>
      <td>1526.422852</td>
      <td>4.485248e+09</td>
    </tr>
    <tr>
      <th>2023-08-30</th>
      <td>25512.494141</td>
      <td>25523.169922</td>
      <td>24923.080078</td>
      <td>24951.197266</td>
      <td>24951.197266</td>
      <td>15584089088</td>
      <td>1591.720703</td>
      <td>1591.753662</td>
      <td>1553.628052</td>
      <td>1559.392578</td>
      <td>1559.392578</td>
      <td>4.778069e+09</td>
    </tr>
  </tbody>
</table>
<p>3257 rows × 12 columns</p>
Le choix entre $inner$, $outer$, $left$, ou $right$ dépend des besoins spécifiques. Ici, nous allons retourner à la méthode $inner$ car elle est plus pratique pour notre analyse. Nommons ce nouveau dataframe btc_eth.


```python
btc_eth = pd.merge(bitcoin, ethereum, on='Date', how='inner', suffixes=('_btc', '_eth'))
```


```python
btc_eth.loc['2019-09',['Close_btc', 'Close_eth']].plot(figsize=(12,8))
```


    <Axes: xlabel='Date'>

![png](./../images/Python_Pandas_Fig_000058.png)
    En visualisant les données avec Matplotlib, nous constatons que les échelles des deux cryptomonnaies sont différentes, rendant la comparaison difficile. Une solution est d'utiliser des sous-graphiques (ou subplots) en spécifiant $subplot=True $ dans notre fonction d'affichage. 


```python
btc_eth.loc['2019-09', ['Close_btc', 'Close_eth']].plot(subplots=True, figsize=(12, 8))
```


    array([<Axes: xlabel='Date'>, <Axes: xlabel='Date'>], dtype=object)


![png](./../images/Python_Pandas_Fig_000059.png)



## 6. Pandas corr()

Quand on regarde ce graphique, on peut se dire que l'Ethereum et le Bitcoin sont deux cryptomonnaies qui sont très bien corrélées. 

Pour calculer cette corrélation, c'est très simple on va écrire $.corr()$. 


```python
btc_eth[['Close_btc', 'Close_eth']].corr()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close_btc</th>
      <th>Close_eth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Close_btc</th>
      <td>1.000000</td>
      <td>0.928583</td>
    </tr>
    <tr>
      <th>Close_eth</th>
      <td>0.928583</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
Et là, on obtient une matrice de corrélation dans pandas. Donc, on peut voir qu'on a une corrélation de 93 %, ce qui est très élevé. Maintenant, si on veut observer ces corrélations, ainsi que d'autres corrélations dans un graphique, on utilisera Seaborn. 

Ça y est, on a fait le tour des fonctions les plus utiles pour travailler sur des time series.

## 7. Exercice : Stratégie de Trading !

Il va s'agir de mettre en place la "stratégie de la tortue", qui est une technique de trading ancienne, afin de décider quand acheter ou vendre du Bitcoin en fonction de la valeur de "close" par rapport au minimum ou au maximum des 28 derniers jours. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000037.png)

Pour écrire ce code, ce que vous allez devoir utiliser, ce sont les fonctions $rolling$ qui vont vous permettre de calculer le minimum et le maximum sur les 28 derniers jours. 

Puis, avec du $boolean\ indexing$, lorsque la valeur de "close" est supérieure au maximum des 28 derniers jours, c'est un signe qu'il faut acheter. À l'inverse, lorsqu'elle est inférieure au minimum des 28 derniers jours, c'est un signe qu'il faut vendre nos actions.

En bonus, vous pouvez également utiliser la méthode $shift$, qui vous permet de décaler vos fenêtres d'autant de jours que vous le souhaitez, afin de prendre vos décisions un jour à l'avance. Cependant, ce n'est pas nécessaire pour cet exercice.

Attention, si je vous donne cet exercice, c'est dans un simple but pédagogique et je vous déconseille fortement de faire des investissements en utilisant cette technique, d'autant plus que le Bitcoin est extrêmement volatile. 

On peut facilement observer les variations au jour le jour en utilisant la fonction $diff$, ce qui nous donne le graphique suivant. 


```python
bitcoin.loc['2019']['Close'].diff().plot()
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000060.png)
    


Et là, on observe tout de suite qu'on a des variations extrêmes dans le cours du Bitcoin. Typiquement, si on examine cette région autour des mois de juin et juillet, on trouve des variations si extrêmes qu'elles mettent en péril notre stratégie de trading. 


```python
bitcoin.loc['2019-06':'2019-07']['Close'].diff().plot()
```


    <Axes: xlabel='Date'>


![png](./../images/Python_Pandas_Fig_000061.png)
    


On pourrait atteindre ce point et penser que c'est un bon moment pour acheter du Bitcoin parce qu'il est supérieur aux 28 jours qui précèdent. 

Et le lendemain, on perd tout notre argent à cause de la volatilité du Bitcoin. Donc, encore une fois, si je vous donne cet exercice, c'est simplement parce que c'est un bon exercice pratique et amusant. 

Mais surtout, n'utilisez jamais cette méthode en vrai, et surtout pas dans la finance, car c'est réservé aux professionnels.

![Python_Pandas_Fig](../images/Python_Pandas_Fig_000038.png) 

Dans cet exercice, il s'agit d'examiner le cours du Bitcoin, ici en bleu, de calculer le minimum et le maximum des 28 derniers jours avec une fonction "rolling". 

Lorsque le cours du Bitcoin dépasse notre bande maximum, comme c'est le cas ici, on prend la décision d'acheter du Bitcoin. 

De la même manière, lorsque le cours du Bitcoin est inférieur au minimum des 28 derniers jours, comme c'est le cas ici, on prend la décision de vendre du Bitcoin. 

Encore une fois, ce n'est pas du tout une bonne stratégie de trading, et je ne vous la conseille surtout pas pour le Bitcoin. En revanche, c'est un bon exercice pour pratiquer les fonctions "rolling".

Pour obtenir ce résultat, on commence par créer une copie de notre DataFrame "Bitcoin". Ainsi, si les choses tournent mal, on a toujours de quoi faire un backup. 


```python
data = bitcoin.copy()
```

Ensuite, on crée une colonne "buy" et une colonne "sell", et on les initialise avec des zéros. C'est ce qui nous donne cette ligne horizontale que vous voyez ici. 
![Python_Pandas_Fig](../images/Python_Pandas_Fig_000039.png) 

```python
data['Buy'] = np.zeros(len(data))
data['Sell'] = np.zeros(len(data))
```

Ensuite, on crée une colonne "rolling_max" et une colonne "rolling_min". On utilise la fonction "rolling" avec une fenêtre de 28 jours, et dans cette fenêtre, on cherche à chaque fois le maximum et le minimum de notre signal "close". Mais attention, il est crucial de décaler notre signal d'un jour $.shift(1)$, sinon notre signal bleu resterait toujours entre son minimum et son maximum. Du coup, on n'achèterait jamais de Bitcoin, ni on ne vendrait jamais de Bitcoin, on n'effectuerait aucune action. Il est donc essentiel de décaler toutes nos fenêtres d'un jour vers la droite.


```python
data['RollingMax'] = data['Close'].shift(1).rolling(window=28).max()
data['RollingMin'] = data['Close'].shift(1).rolling(window=28).min()
```

Une fois que ces fenêtres sont créées, on utilise du boolean indexing. 

Cela permet d'écrire la valeur 1 à l'intérieur de la colonne "buy" lorsque le maximum est inférieur à notre signal "close" et d'écrire -1 dans la colonne "sell" lorsque, à l'inverse, le minimum est supérieur à notre signal "close".


```python
data.loc[data['RollingMax'] < data['Close'], 'Buy'] = 1
data.loc[data['RollingMin'] > data['Close'], 'Sell'] = -1
```

La cellule suivante permet simplement de créer notre graphique avec Matplotlib. Ici, j'utilise la méthode orientée objet pour partager le même axe X, ce qui me permet de faire des zooms sur les deux graphiques simultanément.


```python
start ='2023'
end='2023'
fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)
#plt.figure(figsize=(12, 8))
#plt.subplot(211)
ax[0].plot(data['Close'][start:end])
ax[0].plot(data['RollingMin'][start:end])
ax[0].plot(data['RollingMax'][start:end])
ax[0].legend(['close', 'min', 'max'])
ax[1].plot(data['Buy'][start:end], c='g')
ax[1].plot(data['Sell'][start:end], c='r')
ax[1].legend(['buy', 'sell'])
```


    <matplotlib.legend.Legend at 0x220cbcf3eb0>


![png](./../images/Python_Pandas_Fig_000062.png)
    

