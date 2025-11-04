# Machine Learning apprentissage

<a name="toc"/>

[Introduction](#introduction)

[1. Sujets d'études sur le Machine Learning](#sujet)

- [Section 1 : Rappel de probabilité et de Statistique](#section1)

- [Section 2 : L'apprentissage statistique](#section2)
- [Section 3 : L'incertitude dans le Machine Learning](#section3)
- [Section 4 : Régression linéaire](#section4)
- [Section 5 : Classification](#section5)
- [Section 6 : Les méthodes de rééchantillonnage](#section6)
- [Section 7 : Sélection de Modèle Linéaire](#section7)
- [Section 8 : Extensions au modèle linéaire](#section8)
- [Section 9 : Méthodes basées sur les arbres](#section9)
- [Section 10 : Deep Learning](#section10)
- [Section 11 : ...](#section11)
- [Section 12 : MachineLearnia Python pour le ML et le DL](#section12)
- [Section 13 : MachineLearnia Projet Python : Covid19](#section13)
- [Section 14 : ...](#section14)
- [Section .. : ...](#section15)
- [Section .. : ...](#section16)

[2. Laboratoire sur le Machine Learning](#lab)

- [Regression Linéaire](#lrl)
- [Neurone artificiel](#dlna)
- ...
- ...

[Appendice](#app)

------

<a name="introduction"/>

## Introduction

[Retour TOC](#toc)

Une première étape, avant de se lancer dans les algorithmes du Machine Learning, est l'apprentissage de certaines notions de probabilités et de statistiques. Le Machine Learning utilise des algorithmes qui aident la machine à apprendre. Par exemple, ces algorithmes servent à reconnaître des visages sur des images ou à prédire les préférences de choix de filmes des utilisateurs de streaming tel que Netflix ou YouTube. Ces prédictions sont  basées sur les caractéristiques (ou des variables ou des features) de comportement de visionnage des utilisateurs, elles peuvent être le type de filme ou le temps passé à les regarder.

Ces algorithme de Machine Learning sont indispensables pour identifier toutes les solutions basées sur les nombreuses variables en entrées (pouvant être de quelques milliers) car il serait très compliqué de traiter cette quantité de caractéristiques en programmation classique c'est à dire que cela demanderait le développement de millions (voir de milliards) de combinaisons en fonction du nombre de variables utilisées en source (il faudrait développer une condition sur chaque cas possible). 

[[9]( https://en.wikipedia.org/wiki/Transistor_count)] Avec la montée en puissance de calcul des machines durant ces 40 dernières années, par exemple le nombre de transistor dans les microprocesseurs.

<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 50%;"
    src=".\images\Moore's_Law_Transistor_Count_1970-2020.png" 
    alt="Moore's law">
</img>

[[10]( https://www.i-scoop.eu/big-data-action-value-context/data-age-2025-datasphere/)] et l'augmentation du volume d'information disponible pour tout traitement, par exemple le Big Data. 

<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 40%;"
    src=".\images\Data_Evolution.png" 
    alt="Data_Evolution">
</img>

Les scientifiques ont compris qu'en utilisant, entre autres, la boîte à outils des probabilités et la boîte à outils des statistiques qu'il devenait possible de donner à une machine la capacité d'apprendre sans la programmer de façon explicite.

Voici une liste des problèmes les mieux adaptés pour les algorithme de Machine Learning

1.	Classer les nombres en nombres premiers et non premiers.
2.	Détecter une fraude potentielle dans les transactions par carte de crédit.
3.	Déterminer le temps qu'il faut à un objet qui tombe pour toucher le sol.
4.	Déterminer le cycle optimal des feux de signalisation dans un carrefour très fréquentée.

Avant d'aborder les probabilités, il est important de préciser que la technique d'apprentissage la plus courante s'inspire de l'apprentissage supervisée, c'est à dire que nous fournissons aux algorithmes des données d'apprentissages (Dataset) qui sont utilisées pour créer des modèles. 
Par exemple, l'idée est de fournir à un algorithme, un tableau de données contenant deux variables $X$ et $Y$ et, ensuite, cet algorithme doit déterminer la relation qui relie la variable $X$ à $Y$, c'est à dire $Y=f(X)$ + $\epsilon$ avec $\epsilon$  est un terme d'erreur aléatoire et indépendant de $X$ , c'est une erreur irréductible qui peut contenir des variables non mesurées pour prédire $Y$. 

C'est une des raisons principales de l'utilisation des probabilités. **Quelle est le niveau d'incertitude de ma fonction (de mon modèle)? Quel modèle utiliser avec des informations incomplètes?** Les probabilités nous donne les outils pour quantifier l'incertitude des événements et pour raisonner de manière sensée (mathématique), c'est à dire que la gestion de l'incertitude ne doit pas être due à la chance ou au hasard (attention, à ne pas confondre avec le hasard des variables aléatoire). **La probabilité quantifie la vraisemblance qu'un évènement va se produire et fournit les outils nécessaires pour gérer l'incertitude.**

 De manière générale, **l'apprentissage statistique supervisé consiste à construire un modèle statistique pour prédire ou estimer une sortie en fonction d'une ou plusieurs entrées**. 

A contrario, **avec l'apprentissage statistique non supervisé, il y a des entrées mais pas de sortie supervisée**; néanmoins, nous pouvons apprendre des relations et des structures à partir de ces données. 



------

<a name="sujet"/>

## 1. Sujets d'études sur le Machine Learning ##

[Retour TOC](#toc)

L'apprentissage du Machine Learning nécessite l'étude de plusieurs domaines. Ces domaines sont repris ci-dessous avec un lien sur le document explicatif.



------

<a name="section1"/>

### Section 1 : Rappel de probabilité et de Statistique ###

[Retour TOC](#toc)

------

**La théorie des probabilités** est une branche des mathématiques qui traitent des propriétés de certaines structures modélisant des phénomènes où le "hasard" intervient. 

Cette introduction à la probabilité se trouve sur ce lien [Rappel de probabilité et de statistique](./docs/Rappel_Probabilite_et_Statistique.md) .

Un complément à la première partie qui fait référence au cours de Samuel Leong [[18]](https://see.stanford.edu/materials/aimlcs229/cs229-prob.pdf) [Théorie des probabilités Samuel Long](./docs/Probabilite_Samuel_Long.md) .

------

<a name="section2"/>

### Section 2 : L'apprentissage statistique [[6](https://www.statlearning.com/)] ###

[Retour TOC](#toc)

------

Par essence, l'apprentissage statistique fait référence à un ensemble d'approches permettant d'estimer $f$. Dans ce chapitre, nous présentons certains des concepts théoriques clés qui interviennent dans l'estimation de $f$ , ainsi que des outils permettant d'évaluer les estimations obtenues.

Avec $f$ une certaine fonction ﬁxée mais inconnue de $X_1,...,X_p$, et $\varepsilon$ est un terme d'erreur aléatoire, qui est indépendant de $X$ et a une moyenne de zéro. Dans cette formule, $f$ représente l'information systématique que $X$ fournit sur $Y$ .

**Les variables d'entrée** sont généralement désignées par le symbole $X$, avec un indice pour les distinguer. Ainsi, $X_1$ pourrait être le budget de la télévision, $X_2$ celui de la radio et $X_3$ celui des journaux. **Les entrées portent des noms** diﬀérents, tels que **prédicteurs**, **variables indépendantes**, **caractéristiques (features)**, ou parfois juste variables.

**La variable de sortie** - dans ce cas, les ventes - est souvent **appelée la réponse** ou **la variable dépendante**, et est généralement désignée par le symbole $Y$ . 

Plus généralement, supposons que nous observions une réponse quantitative $Y$ et $p$ prédicteurs diﬀérents, $X_1,X_2,..., X_p$. Nous supposons qu'il existe une relation entre $Y$ et $X=\ (X_1,X_2,...,X_p)$, qui peut s'écrire sous la forme très générale suivante $\boxed{Y=f(X)+\ \epsilon}$.

La compréhension de l'utilisation de la probabilité et de la statistique commence par [L'apprentissage statistique](./docs/L_Apprentissage_Statistique.md) .

------

<a name="section3"/>

### Section 3 : L'incertitude dans le Machine Learning [[5]( https://machinelearningmastery.com/uncertainty-in-machine-learning/)]  ###

[Retour TOC](#toc)

------

Le Machine Learning appliqué nécessite de gérer  l'incertitude. Il existe de nombreuses sources d'incertitude dans un  projet de Machine Learning, notamment

- **la variance** des valeurs de données spécifiques,
- **l'échantillon de données** collectées et
- **la nature imparfaite de tout modèle** développé à partir de ces données.

**La gestion de l'incertitude inhérente à le Machine Learning pour la modélisation prédictive peut être réalisée grâce aux  outils et techniques de probabilité**, un domaine spécifiquement conçu pour gérer l'incertitude : [L'incertitude dans le Machine Learning](./docs/L_incerrtitude_Machine_Learning.md)

------

<a name="section4"/>

### Section 4 : Régression linéaire [[6](https://www.statlearning.com/)]  [[19]](https://see.stanford.edu/Course/CS229) ###

[Retour TOC](#toc)

------

La régression linéaire est un outil utile pour prédire une réponse quantitative. Bien qu'elle puisse sembler quelque peu ennuyeuse comparée à certaines des approches d'apprentissage statistique plus modernes, la régression linéaire reste une méthode d'apprentissage statistique utile et largement utilisée. On ne saurait trop insister sur l'importance de bien comprendre la régression linéaire avant d'étudier des méthodes d'apprentissage plus complexes. 

Le détail dans cette section [Régression Linéaire](./docs/Regression_Lineaire.md) 

Ce lien contient les notes [Lectures Notes Régression Linéaire](./docs/Regression_Lineaire_Stanford.md) du cours donné à Stanford par Andrew Ng sur le Machine Learning  [[19]](https://see.stanford.edu/Course/CS229)

------

<a name="section5"/>

### Section 5 : Classification  [[6](https://www.statlearning.com/)] ###

[Retour TOC](#toc)

------

Le processus de classification permet de prédire des réponses qualitatives.

La prédiction d'une réponse qualitative pour une observation peut être qualifiée de classification de cette observation, car elle implique l'attribution de l'observation à une catégorie ou à une classe. 

D'autre part, les méthodes utilisées pour la classification commencent souvent par prédirent la probabilité que l'observation appartient à chacune des catégories d'une variable qualitative. En ce sens, elles se comportent également comme des méthodes de régression.

Dans cette section [Classification](./docs/Classification.md) , nous passons en revue

------

<a name="section6"/>

### Section 6 : Les méthodes de rééchantillonnage  [[6](https://www.statlearning.com/)] ###

[Retour TOC](#toc)

------

Les méthodes de rééchantillonnage sont un outil indispensable des statistiques modernes. Elles consistent à tirer de manière répétée des échantillons d'un ensemble d'apprentissage et à réajuster un modèle d'intérêt sur chaque échantillon afin d'obtenir des informations supplémentaires sur le modèle ajusté. Par exemple, afin d'estimer la variabilité de l'ajustement d'une régression linéaire, nous pouvons tirer à plusieurs reprises différents échantillons des données d'apprentissage, ajuster une régression linéaire à chaque nouvel échantillon, puis examiner dans quelle mesure les ajustements résultants diffèrent. Une telle approche peut nous permettre d'obtenir des informations qui ne seraient pas disponibles en ajustant le modèle une seule fois en utilisant l'échantillon d'entraînement original.

Les approches de rééchantillonnage peuvent être coûteuses en termes de calcul, car elles impliquent l'ajustement de la même méthode statistique plusieurs fois en utilisant différents sous-ensembles de données d'apprentissage. Cependant, grâce aux récents progrès de la puissance de calcul, les exigences de calcul des méthodes de rééchantillonnage ne sont généralement pas prohibitives. 

Nous abordons deux des méthodes de rééchantillonnage les plus couramment utilisées, la validation croisée et le bootstrap. Ces deux méthodes sont des outils importants dans l'application pratique de nombreuses procédures d'apprentissage statistique. Par exemple, la validation croisée peut être utilisée pour estimer l'erreur de test associée à une méthode d'apprentissage statistique donnée afin d'évaluer ses performances, ou pour sélectionner le niveau de flexibilité approprié. Le processus d'évaluation de la performance d'un modèle est connu sous le nom d'évaluation de modèle, tandis que le processus de sélection du niveau de flexibilité approprié pour un modèle est connu sous le nom de sélection de modèle. Le bootstrap est utilisé dans plusieurs contextes, le plus souvent pour fournir une mesure de la précision de l'estimation d'un paramètre ou d'une méthode d'apprentissage statistique donnée. Le détail de cette section sur le lien [Méthodes_de_Rééchantillonnage](./docs/Methodes_Reechantillonage.md)

------

<a name="section7"/>

### Section 7 : Sélection de Modèle Linéaire  [[6](https://www.statlearning.com/)] ###

[Retour TOC](#toc)

------

Le modèle linéaire présente des avantages distincts en termes d'inférence et, sur les problèmes du monde réel, il est souvent étonnamment compétitif par rapport aux méthodes non linéaires. Par conséquent, avant de passer au monde non linéaire, nous examinons dans cette section certaines façons d'améliorer le modèle linéaire simple, en remplaçant l'ajustement des moindres carrés par d'autres procédures d'ajustement. Le détail de cette section sur le lien [Sélection_de_Modèle_Linéaire](./docs/Modele_Lineaire.md)

------

<a name="section8"/>

### Section 8 : Extensions au modèle linéaire  [[6](https://www.statlearning.com/)] ###

Jusqu'à présent, nous nous sommes principalement concentrés sur les modèles linéaires. Les modèles linéaires sont relativement simples à décrire et à mettre en œuvre, et présentent des avantages par rapport à d'autres approches en termes d'interprétation et d'inférence. 

Cependant, la régression linéaire standard peut présenter des limites importantes en termes de pouvoir prédictif. Cela est dû au fait que l'hypothèse de linéarité est presque toujours une approximation, et parfois une mauvaise approximation. 

La section 7 montre que nous pouvons améliorer les moindres carrés en utilisant la régression ridge, le lasso, la régression en composantes principales et d'autres techniques. Dans ce contexte, l'amélioration est obtenue en réduisant la complexité du modèle linéaire, et donc la variance des estimations. 

Mais nous utilisons toujours un modèle linéaire, qui ne peut être amélioré que jusqu'à un certain point ! Dans cette section, nous assouplissons l'hypothèse de linéarité tout en essayant de maintenir autant d'interprétabilité que possible. 

Pour ce faire, nous examinons des extensions très simples des modèles linéaires, comme la régression polynomiale et les fonctions échelon, ainsi que des approches plus sophistiquées telles que les splines, la régression locale et les modèles additifs généralisés.

Le détail de cette section sur le lien [Extensions au modèle linéaire](./docs/Extensions_au_modèle_linéarité.md)

[Retour TOC](#toc)

------

<a name="section9"/>

### Section 9 : Méthodes basées sur les arbres  [[6](https://www.statlearning.com/)] ###

[Retour TOC](#toc)

Dans cette section, nous décrivons les méthodes de régression et de classification basées sur les arbres. Ces méthodes consistent à stratifier ou à segmenter l'espace de prédiction en un certain nombre de régions simples. Afin d'effectuer une prédiction pour une observation donnée, nous utilisons généralement la valeur de réponse moyenne ou modale pour les observations d'apprentissage dans la région à laquelle elle appartient. 

Puisque l'ensemble des règles de division utilisées pour segmenter l'espace prédicteur peut être résumé dans un arbre, ces types d'approches sont connus comme des ***méthodes d'arbre de décision***.

Les méthodes basées sur les arbres sont simples et utiles pour l'interprétation. Cependant, elles ne sont généralement pas compétitives avec les meilleures approches d'apprentissage supervisé, telles que celles présentées aux sections 7 et 8, en termes de précision de prédiction. C'est pourquoi, dans cette section, nous présentons également les arbres de type ***bagging***, ***random forests***, ***boosting*** et ***régression additive bayésienne***. Chacune de ces approches implique la production de plusieurs arbres qui sont ensuite combinés pour produire une seule prédiction consensuelle. Nous verrons que la combinaison d'un grand nombre d'arbres permet souvent d'améliorer considérablement la précision des prédictions, au prix d'une certaine perte d'interprétation.

Le détail de cette section sur le lien [Méthodes basées sur les arbres](./docs/Methodes_basees_sur_les_arbres.md)

------

<a name="section10"/>

### Section 10 : Deep Learning  [[6](https://www.statlearning.com/)] [[4](https://machinelearnia.com/)] ###

Cette section aborde le sujet important de l'apprentissage profond. Au moment de la rédaction de ce document (2020), l'apprentissage profond est un domaine de recherche très actif dans les communautés de l'apprentissage automatique et de l'intelligence artificielle. La pierre angulaire de l'apprentissage profond est le réseau neuronal.

Le détail de cette section sur le lien [Deep Learning](./docs/Deep_Learning.md)

Un autre document contenant une méthode simple d' apprentissage du Deep Learning divisé en plusieurs leçons : [Leçons_Deep_Learning](./docs/Deep_Learning_ABC.md)

[Retour TOC](#toc)

------

<a name="section11"/>

### Section 11 : ... ###

[Retour TOC](#toc)

------

<a name="section12"/>

### Section 12 : Python pour le Machine Learning et le Deep Learning [[4](https://machinelearnia.com/)] ###

[Retour TOC](#toc)

1. **Numpy :**

  Numpy (Numerical Python) est une bibliothèque essentielle pour la programmation scientifique en Python. Avec ses fonctions puissantes de manipulation de tableaux multidimensionnels et ses outils de calcul mathématiques de haut niveau, elle sert de pilier pour de nombreuses autres bibliothèques dans l'écosystème Python lié à l'analyse de données, au Machine Learning (ML) et au Deep Learning (DL). Voici quelques raisons pour lesquelles l'apprentissage de Numpy est essentiel dans ces domaines :

  1. **Efficient de la manipulation des données** : Numpy offre une manipulation rapide et efficace des tableaux de données de toutes dimensions. C'est essentiel dans le ML et DL, où nous traitons souvent des ensembles de données volumineux.
  2. **Compatibilité avec d'autres bibliothèques** : Numpy s'intègre parfaitement avec d'autres bibliothèques importantes comme Pandas pour la manipulation de données, Matplotlib pour la visualisation, et Scikit-learn pour le ML. En DL, des frameworks tels que TensorFlow et PyTorch utilisent également des structures de données similaires à celles de Numpy.
  3. **Calcul mathématique** : Le ML et DL impliquent beaucoup de calculs mathématiques, notamment l'algèbre linéaire, le calcul statistique, et les transformations de Fourier. Numpy offre une grande variété de fonctions intégrées pour réaliser ces calculs de manière optimisée.
  4. **Vectorisation** : Une caractéristique importante de Numpy est la vectorisation qui permet d'effectuer des opérations sur des tableaux entiers sans avoir à écrire des boucles. Cela rend les calculs beaucoup plus rapides et améliore l'efficacité du code, ce qui est particulièrement important pour le traitement de grands volumes de données en ML et DL.
  5. **Transparence et contrôle** : Contrairement à certaines bibliothèques de ML et DL qui cachent les détails de mise en œuvre, l'utilisation de Numpy donne plus de contrôle et de transparence sur la façon dont les opérations sont effectuées. Cela peut être utile pour la personnalisation, le débogage et l'amélioration de la performance des modèles.

  En résumé, apprendre Numpy n'est pas seulement bénéfique, mais essentiel pour toute personne travaillant dans le ML ou le DL. C'est une compétence fondamentale qui vous permettra de manipuler des données, d'implémenter des algorithmes, d'intégrer votre travail avec d'autres bibliothèques, et finalement de créer des solutions efficaces pour des problèmes de ML et DL complexes.

Le document reprenant tous ces points se trouve en cliquant sur le lien [Python_pour_le_machine_learning_Numpy](./docs/Python_pour_la_machine_learning_Numpy.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python_pour_le_machine_learning_Numpy](./codes/Python_pour_la_machine_learning_Numpy.ipynb)

2. **Matplotlib** :

  [Retour TOC](#toc)

  Permettez une petite question : Pourquoi crée-t-on des graphiques dans la vie ? 

  C'est pour visualiser les choses sur lesquelles on travaille, qu'il s'agisse de données ou d'un modèle. C'est pour mieux comprendre le problème sur lequel on travaille. En d'autres termes, un graphique est censé aider à la résolution de problèmes. 

  Pourtant, pour nombre de personnes qui utilisent Matplotlib, c'est l'inverse qui se produit.Beaucoup de personnes vont créer un graphique et dans ce graphique, il y aura des erreurs. Ainsi, au lieu d'aider à résoudre leurs problèmes, ce graphique leur donne un nouveau problème qu'ils doivent d'abord résoudre avant de s'attaquer à leurs vrais problèmes. Il suffit de consulter le premier forum tel que Stack Overflow pour voir le nombre de personnes qui ont des problèmes avec Matplotlib.

  Pourtant, Matplotlib est très simple à utiliser et en principe, aucun bug ne devrait survenir avec ce package. 

  Si les gens rencontrent parfois des problèmes, c'est d'une part parce qu'ils essaient d'ajouter beaucoup trop de détails à leur courbe. Ils perdent du temps à perfectionner leur graphique alors qu'ils devraient se concentrer sur leur problème de machine learning. D'autre part, c'est parce qu'il existe deux méthodes pour créer des graphiques dans Matplotlib.

  Une méthode est orientée objet et l'autre est plus basique. Comme l'indique Matplotlib sur leur site officiel, les gens ont tendance à mélanger ces deux méthodes, et ils ne devraient pas. 

  Il sera expliqué comment créer des graphiques qui ne soient ni trop simples, ni trop sophistiqués. Juste les graphiques parfaits qu'il faut, sans jamais avoir de bug dans ces graphiques. C'est vraiment très simple.

  Comme mentionné précédemment, il existe deux méthodes pour créer des graphiques avec Matplotlib. La méthode la plus simple est d'utiliser une fonction appelée plot qui provient du module Pyplot. 

Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning MatPlotLib](./docs/Python_pour_la_machine_learning_MatPlotLib.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning MatPlotLib](./codes/Python_pour_la_machine_learning_MatPlotLib.ipynb)

3. **SciPy**

  [Retour TOC](#toc)

  Nous allons voir comment faire du calcul scientifique avec $𝑆𝑐𝑖𝑃𝑦$. À l'intérieur de ce package, on retrouve des outils absolument  incroyables pour faire du mathématique, et bizarrement, beaucoup de  data scientists oublient de les utiliser.

  En l'occurrence, on va voir comment :

  - Faire des interpolations
  - S'attaquer à l'optimisation de problème
  - Procéder au traitement du signal, ce qui inclura la Transformée de Fourier, extrêmement puissante pour filtrer des signaux.

Nous terminerons en voyant comment faire du traitement d'image avec $𝑛𝑑𝑖𝑚𝑎𝑔𝑒$. Je téléchargerai même en live une image qui nous vient d'internet pour  que nous puissions faire l'analyse avec différentes techniques et en  retirer des informations intéressantes dans un tableau $𝑛𝑢𝑚𝑝𝑦$.

  Alors, quand on consulte la documentation officielle de $𝑆𝑐𝑖𝑃𝑦$, qui est disponible à cette adresse : https://docs.scipy.org/doc/scipy/reference/index.html, on peut se rendre compte que dans $SciPy$, on a tout un tas de petits  modules qui nous permettent de réaliser des actions scientifiques bien  précises.  Par exemple, on va retrouver un module pour faire de l'algèbre  linéaire ou un autre pour faire des statistiques. En fait, c'est un peu  comme dans $𝑛𝑢𝑚𝑝𝑦$ où nous avions aussi 𝑙𝑖𝑛𝑎𝑙𝑔 et 𝑠𝑡𝑎𝑡𝑠.

On va s'intéresser tout de suite au module $𝑖𝑛𝑡𝑒𝑟𝑝𝑜𝑙𝑎𝑡𝑒$ et $𝑛𝑑𝑖𝑚𝑎𝑔𝑒$.

Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning Scipy](./docs/Python_pour_la_machine_learning_Scipy.md)

Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning Scipy](./codes/Python_pour_la_machine_learning_Scipy.ipynb)

4. **Panda, las bases :  Analyse du Titanic**

  [Retour TOC](#toc)

  Est-ce que vous saviez que vous aviez plus de chances de survivre à bord du Titanic si vous étiez un homme voyageant en troisième classe plutôt qu'un homme voyageant en seconde classe ?

  Dans cette leçon de la série Python spécial "machine learning", vous allez apprendre à utiliser pandas, qui est l'outil le plus important à connaître quand on souhaite travailler avec des données. 

  Si j'ai mentionné une telle chose, c'est parce qu'avec pandas, vous pouvez réaliser tout ce que vous pourriez imaginer avec des données. 

  Vous pouvez charger vos propres données dans Python, puis les manipuler, les nettoyer, les observer et les analyser. 

  Vous pouvez prendre deux datasets et les assembler ensemble. Bref, vous pouvez faire tout ce genre de choses, et tout cela grâce à une structure très simple à comprendre : le DataFrame.

  Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning Pandas Analyse du Titanic](./docs/Python_pour_la_machine_learning_Pandas_Analyse_du_Titanic.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning Pandas Analyse du Titanic](./codes/Python_pour_la_machine_learning_Pandas_Anamyse_du_Titanic.ipynb)

5. **Panda, Time series**

  [Retour TOC](#toc)

  Nous avons ici l'une des techniques de trading les plus populaires. 

<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 50%;"
    src=".\images\Python_Pandas_Fig_000016.png" 
    alt="Python_Pandas_Fig">
</img>

- Pourtant, je vais vous expliquer pourquoi vous ne devez jamais l'utiliser sur du Bitcoin, au risque de perdre tout votre argent.Nous allons voir comment utiliser $Pandas$ pour travailler sur des problèmes de $time\ series$. Cela va typiquement inclure l'étude du climat, l'analyse de la bourse ou tout autre phénomène qui évolue avec le temps. 

  En réalité, Pandas a même été spécifiquement développé pour aborder ce type de problème, donc nous y trouverons une multitude de fonctionnalités pour travailler sur des time series.

  Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning Pandas Time Series](./docs/Python_pour_la_machine_learning_Pandas_Time_Series.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour la machine learning Pandas Time Series](./codes/Python_pour_la_machine_learning_Pandas_Time_Series.ipynb)

6. **Seaborn**

   [Retour TOC](#toc)

   Vous savez, quand on cherche à résoudre un problème, il faut bien  souvent commencer par visualiser ce problème. Par exemple, en physique,  on cherche à observer des phénomènes, et en "data science", on cherche à visualiser nos données. 

   Et comme vous le savez, pour visualiser des données dans Python, il  existe Matplotlib. Mais bon, Matplotlib, eh bien, c'est Matplotlib quoi. 

   Déjà, c'est peu esthétique, car ça ressemble à ça,  

![Python_Seaborn_Fig](./images/Python_Seaborn_Fig_000001.png)

- alors qu'en réalité, c'est superbe. 

  Et quand vous vous débrouillez pour sortir un graphique à peu près  sympathique, il faut écrire une tonne de code, et nous, on n'est pas là  pour ça. 

  Notre job, c'est de résoudre des problèmes. C'est pour cette raison  qu'il existe Seaborn, construit sur la base de Matplotlib et de Pandas,  et qui permet de réaliser une visualisation de données très poussée en  écrivant seulement une ligne de code. 

  Je répète, avec Seaborn, vous pouvez créer ce genre de graphiques en écrivant simplement une seule ligne de code.  

![Python_Seaborn_Fig](./images/Python_Seaborn_Fig_000002.png)

Le document reprenant tous ces points se trouve en cliquant sur le lien  [Python pour le machine learning SEABORN](./docs/Python_pour_la_machine_learning_SEABORN.md)

Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning SEABORN](./codes/Python_pour_la_machine_learning_SEABORN.ipynb)

7. **SKLEARN**

   [Retour TOC](#toc)

- **KNN, LinearRegression et SUPERVISED LEARNING**

  Est-ce que vous pensez que vous auriez survécu au naufrage du Titanic ? 

  Nous allons développer un modèle de machine learning pour prédire  quelles étaient vos chances de survie, en prenant en compte votre âge,  votre sexe, et la classe dans laquelle vous auriez voyagé. 

  Nous allons utiliser Sklearn pour faire de l'apprentissage supervisé  c'est à dire comment estimer le prix d'un appartement, prédire le cours  de la bourse, détecter un objet sur une photo, ou même calculer vos  chances de survie lors d'une catastrophe telle que celle du Titanic. 

  Mais avant tout, voyons brièvement ce qu'est le machine learning

  

  Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning SKLEARN KNN LinearRegression SUPERVISED LEARNING](./docs/Python_pour_la_machine_learning_SKLEARN_KNN_LinearRegression_SUPERVISED_LEARNING.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning SKLEARN KNN LinearRegression SUPERVISED LEARNING](./codes/Python_pour_la_machine_learning_SKLEARN_KNN_LinearRegression_SUPERVISED_LEARNING.ipynb)

- **Train_test_split, Cross Validation, GridSearchCV**

  [Retour TOC](#toc)

  Je vais vous dévoiler les techniques pour entraîner un modèle, l'optimiser et l'évaluer avec la bonne méthodologie. 

  Nous découvrirons comment créer un Trainset et un Testset à l'aide de la fonction $train\_test\_split()$. 

  Ensuite, nous aborderons la validation d'un modèle grâce à la technique de $cross-validation$. 

  Enfin, nous explorerons comment améliorer un modèle en utilisant $GridSearchCV$ et les courbes d'apprentissage.

  

  Le document reprenant tous ces points se trouve en cliquant sur le lien  [Python pour le machine learning SKLEARN Train test split Cross Validation GridSearchCV](./docs/Python_pour_la_machine_learning_SKLEARN_Train_test_split_Cross_Validation_GridSearchCV.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning SKLEARN Train test split Cross Validation GridSearchCV](./codes/Python_pour_la_machine_learning_SKLEARN_Train_test_split_Cross_Validation_GridSearchCV.ipynb)

- **Metrics Regression**

  [Retour TOC](#toc)

  Nous allons parler de métriques, et plus précisément de métriques de  régression. En effet, beaucoup d'entre vous se demandent quelle est la  différence entre la RMSE, la MAE, le coefficient R carré, et dans  quelles situations utiliser l'un plutôt que l'autre. 

  

  Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning SKLEARN Metrics Regression](./docs/Python_pour_la_machine_learning_SKLEARN_Metrics_Regression.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning SKLEARN Metrics Regression](./codes/Python_pour_la_machine_learning_SKLEARN_Metrics_Regression.ipynb)

- **Make Scorer**

  [Retour TOC](#toc)

  Je vais vous montrer comment utiliser la fonction make_scorer qui  nous vient de Scikit-Learn. Cette fonction est extrêmement utile car  elle vous permet de développer vos propres métriques pour les utiliser  dans des algorithmes de cross-validation ou des algorithmes comme  GridSearchCV. Croyez-moi, développer ses propres métriques pour évaluer  son modèle de machine learning est quelque chose qui arrive très souvent dans le monde professionnel. 

  En effet, quand vous travaillez avec un client, surtout dans les  secteurs industriels, il arrive très souvent que votre client se fiche  un peu de votre coefficient de détermination ou de votre erreur  quadratique moyenne. Lui, il vous fournit un projet avec un cahier des  charges, dans lequel il y a des contraintes que vous devez respecter. Et parmi ces contraintes, on va trouver des mesures de performances qui  sont spécifiques au projet sur lequel vous travaillez.

  

  Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning SKLEARN Make Scorer](./docs/Python_pour_la_machine_learning_SKLEARN_Make_Scorer.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning SKLEARN Make Scorer](./codes/Python_pour_la_machine_learning_SKLEARN_Make_Scorer.ipynb)

- **PRE-PROCESSING + PIPELINE**

  [Retour TOC](#toc)

  Le data processing est l'une des étapes les plus importantes pour développer des modèles avec de bonnes performances. 

  Nous allons commencer par voir ce qu'est le data processing. Je vous montrerai les différentes techniques à connaître. 

  Ensuite, nous verrons comment les mettre en œuvre avec Scikit-learn  et comment construire une chaîne de transformation avec la classe  Pipeline de Scikit-learn.

  

  Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning SKLEARN PRE PROCESSING PIPELINE](./docs/Python_pour_la_machine_learning_SKLEARN_PRE_PROCESSING_PIPELINE.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning SKLEARN PRE PROCESSING PIPELINE](./codes/Python_pour_la_machine_learning_SKLEARN_PRE_PROCESSING_PIPELINE.ipynb)

- **Feature Selection**

  [Retour TOC](#toc)

  Le document reprenant tous ces points se trouve en cliquant sur le lien [Python pour le machine learning SKLEARN Feature Selection](./docs/Python_pour_la_machine_learning_SKLEARN_Feature_Selection.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour le machine learning SKLEARN Feature Selection](./codes/Python_pour_la_machine_learning_SKLEARN_Feature_Selection.ipynb)

- **Apprentissage non supervisé**

  [Retour TOC](#toc)

  Je vais vous présenter les bases de l'apprentissage non supervisé, la deuxième branche très connue du machine learning et du deep learning.  Nous allons explorer les trois applications les plus importantes : 

  - le cloud storage, 
  - la détection d'anomalies, et 
  - la réduction de dimension. 

  Le document reprenant tous ces points se trouve en cliquant sur le lien  [Python pour le machine learning SKLEARN Apprentissage Non Suppervisé](./docs/Python_pour_la_machine_learning_SKLEARN_Apprentissage_Non_Suppervisé.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour la machine learning SKLEARN Apprentissage Non Suppervisé](./codes/Python_pour_la_machine_learning_SKLEARN_Apprentissage_Non_Suppervisé.ipynb)

- **Ensemble BAGGING, BOOSTING et STACKING**
  [Retour TOC](#toc)

  Nous allons parler d'ensemble learning, une technique qui consiste à entraîner plusieurs modèles de machine learning pour ensuite considérer l'ensemble de leurs prédictions. 

  Pour cela, il existe trois grandes méthodes : le bagging, le boosting et le stacking. 

  Les algorithmes qui reposent sur ces méthodes, comme l'algorithme de random forest, comptent parmi les plus performants dans le monde du machine learning.

  Le document reprenant tous ces points se trouve en cliquant sur le lien  [Python pour la machine learning Ensemble BAGGING BOOSTING STACKING](./docs/Python_pour_la_machine_learning_Ensemble_BAGGING_BOOSTING_STACKING.md)

  Le code notebook contenant tous ces points se trouve en cliquant sur ce lien [Python pour la machine learning Ensemble BAGGING BOOSTING STACKING](./Python_pour_la_machine_learning_Ensemble_BAGGING_BOOSTING_STACKING.ipynb)

------

<a name="section13"/>

### Section 13 : Projet Python: COVID19 [[4](https://machinelearnia.com/)] ###

Je vous propose de pratiquer tout ce que nous avons vu en travaillant  sur un vrai dataset. 

Je vous ai donc trouvé un dataset très intéressant  qui va vous permettre de pratiquer tout ce que nous avons vu avec Pandas et Scikit-learn. Il s'agit du dataset "Diagnosis of COVID-19 and  Clinical Spectrum" qui est disponible sur Google et qui regroupe les  résultats cliniques de plus de 5000 personnes, indiquant à chaque fois  si la personne souffre de la maladie du COVID-19 ou non.

En général, le travail de data scientist est divisé en trois activités. 

1. La première, c'est l'analyse et l'exploration de nos données, ce  qu'on appelle en anglais "exploratory data analysis". Ici, le but est de se familiariser avec le dataset et de comprendre les différentes  variables pour ensuite définir une stratégie de modélisation. 

   Le notebook traitant de cette partie se trouve sur ce lien [Python pour le machine learning Projet Coronavirus and Exploratory Data Analysis](./codes/Python_pour_la_machine_learning_Projet_Coronavirus_and_Exploratory_Data_Analysis.ipynb)

2. Une fois cette stratégie définie, on passe à la deuxième activité :  le preprocessing. C'est ici que l'on transforme le dataset pour qu'il  soit prêt pour le développement de modèles de machine learning. On va  encoder les données, éliminer les valeurs manquantes, sélectionner des  variables, etc.

   Le notebook traitant de cette partie se trouve sur ce lien [Python pour le machine learning Projet Coronavirus and PRE-Traitement](./codes/Python_pour_la_machine_learning_Projet_Coronavirus_and_PRE-Traitement.ipynb)

3. Finalement, on arrive à la troisième activité : la modélisation. Le  but est clair : créer, entraîner, évaluer et améliorer un modèle de  machine learning. On va peut-être aussi comparer ce modèle avec d'autres modèles pour atteindre l'objectif initial.

   Le notebook traitant de cette partie se trouve sur ce lien [Python pour le machine learning Projet Coronavirus Modèle](./codes/Python_pour_la_machine_learning_Projet_Coronavirus_Modèle.ipynb)

[Retour TOC](#toc)

------

<a name="section14"/>

### Section 14 : ... ###

[Retour TOC](#toc)

------



<a name="section15"/>

### Section .. : ... ###

[Retour TOC](#toc)

------



<a name="section16"/>

### Section .. : ... ###

[Retour TOC](#toc)

------

<a name="lab"/>

## 2. Laboratoire sur le Machine Learning ##

[Retour TOC](#toc)

<a name="lrl"/>

### Labs 1 : Regression Linéaire [[4](https://machinelearnia.com/)] ###

[Retour TOC](#toc)

La recette de la régression linéaire :

1. Récolter des données
2. Donner à la machine un modèle linéaire
3. Créer la Fonction Coût
4. Calculer le gradient et utiliser l’algorithme de Gradient Descent avec  le Learning Rate qui prend le nom **d’hyperparamètre** de par son influence sur la performance finale du modèle (s’il est trop grand où trop petit, la fonction le Gradient Descent ne converge pas).

Un explication sur la Fonction coût, ainsi qu'un exemple concret sur la Regression Linéaire se trouve sur ce lien [Labs Regression Linéaire](./labs/Labs_Regression_Lineaire.md) .

<a name="dlna"/>

### Labs 2 : Deep Learning : Programmation d'un neurone artificiel [[4](https://machinelearnia.com/)] ###

[Retour TOC](#toc)

Nous allons développer notre premier programme de neurone artificiel. Et pour ça nous allons implémenter toutes les équations que l'on a vu dans les dernières leçons.

Alors, pour développer notre programme de neurones artificiels, nous allons partir d'un Dataset $(X, y)$ de 100 lignes et de deux colonnes. 

Si on veut, on peut imaginer que ce Dataset représente des plantes avec la longueur et la largeur de leurs feuilles. Et notre but, c'est d'entraîner un neurone artificiel pour reconnaître les plantes toxiques des plantes non toxique grâce à ces données de référence.

Voici l'implémentation. Le code se trouve sur [Labs_2_Deep_Learning_Programmation_neurone_artificiel](./codes/Labs_2_Deep_Learning_Programmation_neurone_artificiel.ipynb)

Une implémentation objet se trouve sur ce lien : [Labs_2_Deep_Learning_Programmation_neurone_artificiel_OO](./codes/Labs_2_Deep_Learning_Programmation_neurone_artificiel_OO.ipynb)

### Labs 3 : Deep Learning : Programmation Chien vs Chat [[4](https://machinelearnia.com/)] ###

[Retour TOC](#toc)

Nous allons développer un programme de vision par ordinateur pour reconnaître une photo de chat ou de chien. Donc ce qu’on aimerait faire, ça serait de fournir des photos de chats et de chiens à notre code pour qu’ils nous retourne un modèle qui soit capable de classer ce genre de photos.

Voici l'implémentation. Le code se trouve sur [Labs_3_Deep_Learning_Chient_vs_Chat](./codes/Labs_3_Deep_Learning_Chient_vs_Chat.ipynb)

### Labs 4 : Deep Learning : Programmer un réseau de neurones à 2 couches [[4](https://machinelearnia.com/)] ###

[Retour TOC](#toc)

Nous allons développer un réseau de neurones à 2 couches.

Voici l'implémentation. Le code se trouve sur [Labs_4_Deep_Learning_Programmation_neurone_artificiel_2_couches](./codes/Labs_4_Deep_Learning_Programmation_neurone_artificiel_2_couches.ipynb)

### Labs 5 : Deep Learning : Réseau de neurones profond [[4](https://machinelearnia.com/)] ###

[Retour TOC](#toc)

Pour développer un réseau de neurones profonds, avec autant de couches que l'on désire à l'intérieur, nous allons repartir des équations qui nous avait permis de créer un réseau de neurones à deux couches. 

Voici l'implémentation. Le code se trouve sur [Labs_5_Reseau_de_neurones_profonds](./codes/Labs_5_Reseau_de_neurones_profonds.ipynb)

------

<a name="app"/>

## Appendice ##

[Retour TOC](#toc)

-A- Notation et algèbre matricielle simple

-B- Calcul des dérivées partielles : descente de gradient

-C- Algèbre Linéaire

Lien vers [Appendice Mathématique](./docs/Appendice_Mathematique.md)
