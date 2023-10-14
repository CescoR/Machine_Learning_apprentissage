# [Appendice](#appendice)  #

[Retour README](../README.md)

<a name="toc"/>

[-A- Notation et algèbre matricielle simple](#a)

[-B- Calcul des dérivées partielles : descente de gradient](#b)

[-C- Algèbre Linéaire](#c)

- [C.1 Concepts et notations de base](#c-1)
  - [C.1.1 Notation de base](#c-1-1)

- [C.2 Multiplication matricielle](#c-2)

  - [C.2.1 Produits vectoriels](#c-2-1)

  - [C.2.2 Produits matrice-vecteur](#c-2-2)

  - [C.2.3 Produits matriciels](#c-2-3)

- [C.3 Opérations et propriétés](#c-3)

  - [C.3.1 Matrice d'identité et matrices diagonales](#c-3-1)

  - [C.3.2 La transposition](#c-3-2)

  - [C.3.3 Matrices symétriques](#c-3-3)

  - [C.3.4 La trace](#c-3-4)

  - [C.3.5 Normes](#c-3-5)

  - [C.3.6 Indépendance linéaire et rang](#c-3-6)

  - [C.3.7 L'inverse](#c-3-7)

  - [C.3.8 Matrices orthogonales](#c-3-8)

  - [C.3.9 Plage et nulspace d'une matrice](#c-3-9)

  - [C.3.10 Le déterminant](#c-3-10)

  - [C.3.11 Formes quadratiques et matrices semi-définies positives](#c-3-11)

  - [C.3.12 Valeurs propres et vecteurs propres](#c-3-12)

  - [C.3.13 Valeurs propres et vecteurs propres des matrices symétriques](#c-3-13)

- [C.4 Calcul matriciel](#c-4)

  - [C.4.1 Le gradient](#c-4-1)

  - [C.4.2 Le hessien](#c-4-2)

  - [C.4.3 Gradients et hessian des fonctions quadratiques et linéaires](#c-4-3)

  - [C.4.4 Les moindres carrés](#c-4-4)

  - [C.4.5 Gradients du déterminant](#c-4-5)

  - [C.4.6 Valeurs propres en tant qu'optimisation](#c-4-6)

<a name="A"/>

## [-A- Notation et algèbre matricielle simple](#a) ##

[Retour TOC](#toc)

[[6](https://www.statlearning.com/)] Le choix de la notation pour un manuel est toujours une tâche difficile. Nous utiliserons $n$ pour représenter le nombre de points de données distincts, ou d'observations, dans notre échantillon, et nous utiliserons $p$ pour désigner le nombre de variables qui sont disponibles pour être utilisées dans les prédictions. 

Par exemple, l'ensemble de données sur les salaires se compose de 11 variables pour 3000 personnes. Nous avons donc $n=3000$ observations et $p=11$ variables (telles que l'année, l'âge, la race, etc.). 

Dans certains exemples, $p$ peut être très grand, de l'ordre de milliers ou même de millions ; cette situation se présente assez souvent, par exemple, dans l'analyse de données biologiques modernes ou de données publicitaires sur Internet.

En général, nous utilisons $x_{ij}$ comme la valeur de la $j^{ème}$ variable pour la $i^{ème}$ observation, avec $i=1,\ 2,...,n$ et $j=1,\ 2,...,p$. 
La variable $i$ est utilisé pour indexer les échantillons ou les observations (de $1\ à\ n$) et $j$ est utilisé pour indexer les variables (de $1\ à\ p$). $X$ dénote une matrice $n \times p$ dont le $i,j^{ème}$ élément est $x_{ij}$. 
C'est-à-dire,       

$$
X = 
 \begin{pmatrix}
  x_{11} & x_{12} & \cdots & x_{1p} \\
  x_{21} & x_{22} & \cdots & x_{2p} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  x_{n1} & x_{n2} & \cdots & x_{np} 
 \end{pmatrix}
$$

Pour les lecteurs qui ne sont pas familiarisés avec les matrices, il est utile de visualiser $X$ comme une feuille de calcul contenant nombres avec $n$ lignes et $p$ colonnes.

Parfois, nous sommes intéressés par les rangs de $X$, que nous écrivons comme $x_1,x_2,\ldots,x_n$. Ici $x_i$ est un vecteur de longueur $p$, contenant les $p$ mesures de variables pour la $i^{ème}$  observation. 
C'est-à-dire que

$$
X = 
 \begin{pmatrix}
  x_{i1}\\
  x_{i1}\\
  \vdots \\
  x_{ip} 
 \end{pmatrix}
$$

Les vecteurs sont représentés par défaut sous forme de colonnes. 
Par exemple, pour les données relatives aux salaires, $x_i$ est un vecteur de longueur 11, composé de l'année, de l'âge, de la race et d'autres valeurs pour le $i^{ème}$ individu.  A d'autres moments, nous sommes plutôt intéressés par les colonnes de $X$, que nous écrivons comme $x_1,\ x_2,...,\ x_p$. 
Chacune des colonnes est un vecteur de longueur $n$. C'est-à-dire,

$$
X = 
 \begin{pmatrix}
  x_{1j}\\
  x_{2j}\\
  \vdots \\
  x_{nj} 
 \end{pmatrix}
$$

Par exemple, pour les données sur les salaires, x_1 contient n=3000 valeurs par an. En utilisant cette notation, la matrice $X$ peut être écrite comme des valeurs pour l'année.

$$
X = 
 \begin{pmatrix}
  x_{1} \ x_{2} \ ... \ x_p\\
 \end{pmatrix},
$$

Ou

$$
X = 
 \begin{pmatrix}
  x_{1}^T\\
  x_{2}^T\\
  \vdots \\
  x_{n}^T 
 \end{pmatrix}
$$

La notation $^T$ désigne la transposition d'une matrice ou d'un vecteur. 
Ainsi, par exemple,

$$
X^T = 
 \begin{pmatrix}
  x_{11} & x_{21} & \cdots & x_{n1} \\
  x_{12} & x_{22} & \cdots & x_{n2} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  x_{1p} & x_{2p} & \cdots & x_{np} 
 \end{pmatrix}
$$

Et

$$
x_i^T=(x_{i1}x_{i2}\ldots x_{ip})
$$

Nous utilisons $y_i$ pour désigner la $i^{ème}$ observation de la variable sur laquelle nous souhaitons faire des prédictions, comme par exemple le salaire. Par conséquent, nous écrivons l'ensemble des n observations sous forme de vecteur comme suit 

$$
y = 
 \begin{pmatrix}
  y_{1}\\
  y_{2}\\
  \vdots \\
  y_{n} 
 \end{pmatrix}
$$

Alors nos données observées consistent en ${(x_1,y_1),\ (x_2,y_2),...,\ (x_n,y_n)}$, où chaque $x_i$ est un vecteur de longueur $p$. (Si $p=1$, alors $x_i$ est simplement un scalaire.) 

Dans ce texte, un vecteur de longueur $n$ sera toujours désigné par une minuscule grasse.

$$
\mathtt{a} =
 \begin{pmatrix}
  a_{1}\\
  a_{2}\\
  \vdots \\
  a_{n} 
 \end{pmatrix}
$$

Cependant, les vecteurs qui ne sont pas de longueur $n$ (tels que les vecteurs attributs de longueur $p$) sont désignés par une police normale minuscule, par exemple $\boldsymbol a$. 

Les scalaires sont également désignés par une police normale minuscule, par exemple $a$. Dans les rares cas où ces deux utilisations de la police normale minuscule conduisent à une ambiguïté, nous précisons quelle utilisation est prévue. 

Les matrices sont désignées par des majuscules en gras, par exemple $\boldsymbol A$. 

Les variables aléatoires sont désignées par une police normale en majuscules, par exemple $A$, quelles que soient leurs dimensions.

Occasionnellement, nous voulons indiquer la dimension d'un objet particulier. Pour indiquer qu'un objet est un scalaire, nous utiliserons la notation $a\in\mathbb{R}$. 

Pour indiquer que c'est un vecteur de longueur $k$, nous utilisons $a\in\mathbb{R}^k$ (ou $a\in\mathbb{R}^n$ s’il est de longueur n). 

Nous indiquerons qu'un objet est une matrice $r\ \times\ s$ en utilisant $A\in\mathbb{R}^{r\ {x\ }s}$   
 
Dans la mesure du possible, nous évitons d'utiliser l'algèbre matricielle. Cependant, dans certains cas, cela devient trop lourd pour l'éviter complètement. Dans ces rares cas, il est important de comprendre le concept de multiplication de deux matrices. 

Supposons que $A\in\mathbb{R}^{r\ \times\ d}$ et $B\in\mathbb{R}^{d\ \times\ s}$. Alors le produit de $A$ et $B$ est dénoté $AB$. 
Le $i,j^{ème}$ élément de $AB$ est calculé en multipliant chaque élément de la $i^{ème}$ ligne de $A$ par l'élément correspondant de la $j^{ème}$ colonne de $B$.

Donc, nous avons

$$
\boxed{AB_{ij}=\sum_{k=1}^{d}{a_{ik}b_{kj}}}
$$

Soit l’exemple, 

$$
A = 
 \begin{pmatrix}
  1 & 2\\
  3 & 4
 \end{pmatrix}
 \ et \ 
\
B = 
 \begin{pmatrix}
  5 & 6\\
  7 & 8
 \end{pmatrix}
$$

Alors

$$
AB = 
 \begin{pmatrix}
  1 & 2\\
  3 & 4
 \end{pmatrix}
 \begin{pmatrix}
  5 & 6\\
  7 & 8
 \end{pmatrix}
  =\begin{pmatrix}
  1 \times 5 + 2\times 7 \hspace{2em}1 \times 6 + 2\times 8 \\
  3 \times 5 + 4\times 7 \hspace{2em}3 \times 6 + 4\times 8 
 \end{pmatrix}
  =\begin{pmatrix}
  19 & 22\\
  43 & 50
 \end{pmatrix}
$$

Notez que cette opération produit une matrice $r\ \times\ s$. Il n'est possible de calculer $AB$ que si le nombre de colonnes de $A$ est égal au nombre de lignes de $B$.



<a name="B"/>

## [-B- Calcul des dérivées partielles: descente de gradient](#b) ##

[Retour TOC](#toc)

Pour implémenter l’algorithme de Gradient Descent, il faut calculer les dérivées partielles de la Fonction de Coût. Pour rappel, en mathématique, la dérivée d’une fonction en un point nous donne la valeur de sa pente en ce point.

Fonction Coût : 




$$
J(a, b) = \frac{1}{2m} \sum_{i=1}^{m} (ax_i + b - y_i)^2
$$




Dérivée selon le paramètre $a$ : 




$$
\frac{\partial J(a,b)}{\partial a} = \frac{1}{m} \sum_{i=1}^{m} (ax_i + b - y_i) \times x_i
$$




Dérivée selon le paramètre 𝒃 :




$$
\frac{\partial J(a,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (ax_i + b - y_i)
$$




Nous somme dans le cas de la dérivée d’une fonction composée : 




$$
(𝑔 ∘ 𝑓)′ = 𝑓′ × 𝑔′ ∘ 𝑓. Avec : 𝒇 = 𝒂𝒙 + 𝒃 − 𝒚\ et\ 𝒈 = (𝒇)^𝟐 .
$$


En dérivant, le carré tombe et se simplifie avec la fraction $\frac{1}{2m}$ pour devenir $\frac{1}{m}$ et $x_i$ apparait en facteur pour la dérivée par rapport à $a$.

<a name="C"/>

## [-C- Algèbre Linéaire](#c) ##

Révision et référence en algèbre linéaire [[19]](https://see.stanford.edu/Course/CS229)

<a name="C-1"/>

### [C.1 Concepts et notations de base](#c-1) ###

[Retour TOC](#toc)

L'algèbre linéaire permet de représenter de manière compacte et d'opérer sur des ensembles d'équations linéaires. Par exemple, considérons le système d'équations suivant :




$$
\begin{aligned}
4 x_1-5 x_2 & =-13 \\
-2 x_1+3 x_2 & =9 .
\end{aligned}
$$


Il s'agit de deux équations et de deux variables. Comme vous l'avez appris en cours d'algèbre au lycée, vous pouvez trouver une solution unique pour $x_1$ et $x_2$ (sauf si les équations sont dégénérées d'une manière ou d'une autre, par exemple si la deuxième équation est simplement un multiple de la première, mais dans le cas ci-dessus, il existe en fait une solution unique). En notation matricielle, nous pouvons écrire le système de manière plus compacte comme suit :




$$
\begin{aligned}
& A x=b \\
& \text { with } A=\left[\begin{array}{cc}
4 & -5 \\
-2 & 3
\end{array}\right], b=\left[\begin{array}{c}
13 \\
-9
\end{array}\right] \text {. } \\
&
\end{aligned}
$$




Comme nous le verrons bientôt, l'analyse des équations linéaires sous cette forme présente de nombreux avantages (y compris un gain de place évident).

<a name="C-1-1"/>

#### [C.1.1 Notation de base](#c-1-1) ####

[Retour TOC](#toc)

Nous utilisons la notation suivante :

- Par $A \in \mathbb{R}^{m \times n}$ nous désignons une matrice avec $m$ lignes et $n$ colonnes, où les entrées de $A$ sont des nombres réels.

- Par $x \in \mathbb{R}^n$, on désigne un vecteur à $n$ entrées. Habituellement, un vecteur $x$ désignera un ***vecteur colonne -*** c'est-à-dire une matrice à $n$ lignes et 1 colonne. 

  Si nous voulons représenter explicitement un ***vecteur ligne*** - une matrice avec 1 ligne et $n$ colonnes - nous écrivons typiquement $x^T$ (ici $x^T$ désigne la transposée de $x$, que nous définirons bientôt).

- L'élément $i$ d'un vecteur $x$ est noté $x_i$ :




$$
x=\left[\begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{array}\right] .
$$




- Nous utilisons la notation $a_{i j}$ (ou $A_{i j}, A_{i, j}$, etc) pour désigner l'entrée de $A$ dans la $i^{ème}$ ligne et la $j^{ème}$ colonne :




$$
A=\left[\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 n} \\
a_{21} & a_{22} & \cdots & a_{2 n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m 1} & a_{m 2} & \cdots & a_{m n}
\end{array}\right] .
$$




- Nous désignons la $j^{ème}$ colonne de $A$ par $a_j$ ou $A_{ :, j}$ :




$$
A=\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
a_1 & a_2 & \cdots & a_n \\
\mid & \mid & & \mid
\end{array}\right] .
$$




- Nous désignons la $i^{ème}$ ligne de $A$ par $a_i^T$ ou $A_{i,:}$ :




$$
A=\left[\begin{array}{ccc}
{-} & a_1^T & {-} \\
{-} & a_2^T & {-} \\
& \vdots & \\
{-} & a_m^T & {-}
\end{array}\right]
$$




- Notez que ces définitions sont ambiguës (par exemple, les $a_1$ et $a_1^T$ dans les deux définitions précédentes ne sont pas le même vecteur). En général, la signification de la notation devrait être évidente à partir de son utilisation.

<a name="C-2"/>

### [C.2 Multiplication matricielle](#c-2) ###

[Retour TOC](#toc)

Le produit de deux matrices $A \in \mathbb{R}^{m \times n}$ et $B \in \mathbb{R}^{n \times p}$ est la matrice




$$
C=A B \in \mathbb{R}^{m \times p},
$$




où




$$
C_{i j}=\sum_{k=1}^n A_{i k} B_{k j} .
$$




Notez que pour que le produit matriciel existe, le nombre de colonnes de $A$ doit être égal au nombre de lignes de $B$. Il existe de nombreuses façons de considérer la multiplication matricielle, et nous allons commencer par examiner quelques cas particuliers.

<a name="C-2-1"/>

#### [C.2.1 Produits vectoriels](#c-2-1) ####

[Retour TOC](#toc)

Étant donné deux vecteurs $x, y \in \mathbb{R}^n$, la quantité $x^T y$, parfois appelée produit interne ou produit scalaire des vecteurs, est un nombre réel donné par




$$
x^T y \in \mathbb{R}=\sum_{i=1}^n x_i y_i .
$$




Notez que c'est toujours le cas que $x^T y=y^T x$.

Étant donné les vecteurs $x \in \mathbb{R}^m, y \in \mathbb{R}^n$ (il n'est plus nécessaire qu'ils aient la même taille), $x y^T$ est appelé le produit externe des vecteurs. Il s'agit d'une matrice dont les entrées sont données par $\left(x y^T\right)_{i j}=x_i y_j$, c'est-à-dire,




$$
x y^T \in \mathbb{R}^{m \times n}=\left[\begin{array}{cccc}
x_1 y_1 & x_1 y_2 & \cdots & x_1 y_n \\
x_2 y_1 & x_2 y_2 & \cdots & x_2 y_n \\
\vdots & \vdots & \ddots & \vdots \\
x_m y_1 & x_m y_2 & \cdots & x_m y_n
\end{array}\right]
$$


<a name="C-2-2"/>

#### [C.2.2 Produits matrice-vecteur](#c-2-2) ####

[Retour TOC](#toc)

Étant donné une matrice $A \in \mathbb{R}^{m \times n}$ et un vecteur $x \in \mathbb{R}^n$, leur produit est un vecteur $y=A x \in \mathbb{R}^m$. Il existe deux façons de considérer la multiplication matrice-vecteur, et nous allons les examiner toutes les deux.

Si nous écrivons $A$ par lignes, nous pouvons alors exprimer $A x$ comme,




$$
y=\left[\begin{array}{ccc}
{-} & a_1^T & {-} \\
{-} & a_2^T & {-} \\
\vdots & \\
{-} & a_m^T & {-}
\end{array}\right] x=\left[\begin{array}{c}
a_1^T x \\
a_2^T x \\
\vdots \\
a_m^T x
\end{array}\right] .
$$




En d'autres termes, la $i^{ème}$ entrée de $y$ est égale au produit interne de la $i^{ème}$ ligne de $A$ et de $x$, $y_i=a_i^T x$.

On peut aussi écrire $A$ sous forme de colonnes. Dans ce cas, nous voyons que,




$$
y=\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
a_1 & a_2 & \cdots & a_n \\
\mid & \mid & & \mid
\end{array}\right]\left[\begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{array}\right]=\left[a_1\right] x_1+\left[\begin{array}{c}
a_2 \\
x_2+\ldots+\left[x_n\right.
\end{array}\right] x_n
$$


En d'autres termes, y est une combinaison linéaire des colonnes de $A$, où les coefficients de la combinaison linéaire sont donnés par les entrées de $x$.

Jusqu'à présent, nous avons multiplié à droite par un vecteur colonne, mais il est également possible de multiplier à gauche par un vecteur ligne. Cela s'écrit $y^T=x^T A$ pour $A \in \mathbb{R}^{m \times n}, x \in \mathbb{R}^m$, et $y \in \mathbb{R}^n$. Comme précédemment, nous pouvons exprimer $y^T$ de deux manières évidentes, selon que nous exprimons $A$ en termes sur ses lignes ou ses colonnes.

Dans le premier cas, nous exprimons $A$ en termes de ses colonnes, ce qui donne




$$
y^T=x^T\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
a_1 & a_2 & \cdots & a_n \\
\mid & \mid & & \mid
\end{array}\right]=\left[\begin{array}{llll}
x^T a_1 & x^T a_2 & \cdots & x^T a_n
\end{array}\right]
$$




ce qui démontre que la $i$ ième entrée de $y^T$ est égale au produit interne de $x$ et de la $i$ ième colonne de $A$.

Enfin, en exprimant $A$ en termes de lignes, nous obtenons la représentation finale du produit vecteur-matrice,




$$
\begin{aligned}
y^T & =\left[\begin{array}{llll}
x_1 & x_2 & \cdots & x_n
\end{array}\right]\left[\begin{array}{ccc}
{-} &a_1^T & {-} \\
{-} &a_2^T & {-} \\
\vdots & \\
{-}&a_m^T & {-}
\end{array}\right] \\
& =x_1\left[-a_1^T-\right]+x_2\left[\begin{array}{lll}
{-} & a_2^T & {-}
\end{array}\right]+\ldots+x_n\left[\begin{array}{lll}
{-} & a_n^T{-}
\end{array}\right]
\end{aligned}
$$




Nous voyons donc que $y^T$ est une combinaison linéaire des lignes de $A$, où les coefficients de la combinaison linéaire sont donnés par les entrées de $x$.

<a name="C-2-3"/>

#### [C.2.3 Produits matriciels](#c-2-3) ####

[Retour TOC](#toc)

Forts de ces connaissances, nous pouvons maintenant examiner quatre façons différentes (mais bien sûr équivalentes) de considérer la multiplication matrice-matrice $C=A B$ telle que définie au début de cette section. Tout d'abord, nous pouvons considérer la multiplication matrice-matrice comme un ensemble de produits vecteur-vecteur. Le point de vue le plus évident, qui découle immédiatement de la définition, est que l'entrée $i, j$ de $C$ est égale au produit interne de la $i$ème ligne de $A$ et de la $j$ème ligne de $B$. Symboliquement, cela ressemble à ce qui suit,




$$
C=A B=\left[\begin{array}{cc}
{-} & a_1^T \\
{-} & a_2^T \\
\vdots \\
{-} & a_m^T
\end{array}\right]\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
b_1 & b_2 & \cdots & b_p \\
\mid & \mid & & \mid
\end{array}\right]=\left[\begin{array}{cccc}
a_1^T b_1 & a_1^T b_2 & \cdots & a_1^T b_p \\
a_2^T b_1 & a_2^T b_2 & \cdots & a_2^T b_p \\
\vdots & \vdots & \ddots & \vdots \\
a_m^T b_1 & a_m^T b_2 & \cdots & a_m^T b_p
\end{array}\right]
$$




Rappelez-vous que puisque $A \in \mathbb{R}^{m \times n}$ et $B \in \mathbb{R}^{n \times p}, a_i \in \mathbb{R}^n$ et $b_j \in \mathbb{R}^n$, ces produits internes ont tous un sens. C'est la représentation la plus "naturelle" lorsque nous représentons $A$ par des lignes et $B$ par des colonnes. Alternativement, nous pouvons représenter $A$ par des colonnes et $B$ par des lignes, ce qui conduit à l'interprétation de $A B$ comme une somme de produits externes. Symboliquement,




$$
C=A B=\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
a_1 & a_2 & \cdots & a_n \\
\mid & \mid & & \mid
\end{array}\right]\left[\begin{array}{ccc}
{-} & b_1^T & {-} \\
{-} & b_2^T & {-} \\
\vdots \\
{-} & b_n^T & {-}
\end{array}\right]=\sum_{i=1}^n a_i b_i^T .
$$


Autrement dit, $A B$ est égal à la somme, sur tous les $i$, du produit externe de la $i^{ème}$ colonne de $A$ et de la $i$ ème ligne de $B$. Puisque, dans ce cas, $a_i \in \mathbb{R}^m$ et $b_i \in \mathbb{R}^p$, la dimension du produit extérieur $a_i b_i^T$ est $m \times p$, ce qui coïncide avec la dimension de $C$.

Deuxièmement, nous pouvons également considérer la multiplication matrice-matrice comme un ensemble de produits matrice-vecteur. Plus précisément, si nous représentons $B$ par des colonnes, nous pouvons considérer les colonnes de $C$ comme des produits matrice-vecteur entre $A$ et les colonnes de $B$. Symboliquement,




$$
C=A B=A\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
b_1 & b_2 & \cdots & b_p \\
\mid & \mid & & \mid
\end{array}\right]=\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
A b_1 & A b_2 & \cdots & A b_p \\
\mid & \mid & & \mid
\end{array}\right] .
$$




Ici, la colonne $i$ de $C$ est donnée par le produit matrice-vecteur avec le vecteur de droite, $c_i=A b_i$. Ces produits matrice-vecteur peuvent à leur tour être interprétés en utilisant les deux points de vue donnés dans la sous-section précédente. Enfin, nous avons le point de vue analogue, où nous représentons $A$ par des lignes, et voyons les lignes de $C$ comme le produit matrice-vecteur entre les lignes de $A$ et $C$. Symboliquement,




$$
C=A B=\left[\begin{array}{cc}
{-} & a_1^T \\
{-} & a_2^T \\
\vdots \\
{-} & a_m^T
\end{array}\right] B=\left[\begin{array}{ccc}
{-} & a_1^T B & {-} \\
{-} & a_2^T B & {-} \\
\vdots & \\
{-} & a_m^T B & {-}
\end{array}\right]
$$




Ici la $i^{ème}$ ligne de $C$ est donnée par le produit matrice-vecteur avec le vecteur de gauche, $c_i^T=a_i^T B$.

Il peut sembler exagéré de disséquer la multiplication matricielle à un tel degré, surtout lorsque tous ces points de vue découlent immédiatement de la définition initiale que nous avons donnée (en une ligne de mathématiques environ) au début de cette section. Cependant, pratiquement toute l'algèbre linéaire traite des multiplications matricielles d'une manière ou d'une autre, et il vaut la peine de passer un peu de temps à essayer de développer une compréhension intuitive des points de vue présentés ici.

En plus de cela, il est utile de connaître quelques propriétés de base de la multiplication matricielle à un niveau supérieur :

- La multiplication matricielle est associative : $(A B) C=A(B C)$.

- La multiplication matricielle est distributive : $A(B+C)=A B+A C$.

- La multiplication matricielle n'est, en général, pas commutative, c'est-à-dire qu'il peut arriver que $A B \neq B A$.

<a name="C-3"/>

### [C.3 Opérations et propriétés](#c-3) ###

[Retour TOC](#toc)

Dans cette section, nous présentons plusieurs opérations et propriétés des matrices et des vecteurs. Nous espérons qu'une grande partie de ces notions vous sera familière et que les notes serviront juste de référence pour ces sujets.

<a name="C-3-1"/>

#### [C.3.1 Matrice d'identité et matrices diagonales](#c-3-1) ####

[Retour TOC](#toc)

La matrice d'identité, notée $I \in \mathbb{R}^{n \times n}$, est une matrice carrée avec des uns sur la diagonale et des zéros partout ailleurs. C'est-à-dire ,




$$
I_{i j}= \begin{cases}1 & i=j \\ 0 & i \neq j\end{cases}
$$




Il a la propriété que pour tout $A \in \mathbb{R}^{m \times n}$,




$$
A I=A=I A
$$


où la taille de $I$ est déterminée par les dimensions de $A$ de sorte que la multiplication de la matrice est possible.

Une matrice diagonale est une matrice dont tous les éléments non diagonaux sont 0 . Elle est généralement notée $D={diag}\left(d_1, d_2, \ldots, d_n\right)$, avec




$$
D_{i j}= \begin{cases}d_i & i=j \\ 0 & i \neq j\end{cases}
$$


Clairement, $I={diag}(1,1, \ldots, 1)$.

<a name="C-3-2"/>

#### [C.3.2 La transposition](#c-3-2) ####

[Retour TOC](#toc)

La transposition d'une matrice résulte de la " permutation " des lignes et des colonnes. Étant donné une matrice $A \in \mathbb{R}^{m \times n}$, sa transposition, écrite $A^T$, est définie comme suit




$$
A^T \in \mathbb{R}^{n \times m},\left(A^T\right)_{i j}=A_{j i} .
$$




En fait, nous avons déjà utilisé la transposition pour décrire les vecteurs lignes, puisque la transposition d'un vecteur colonne est naturellement un vecteur ligne.

Les propriétés suivantes des transpositions sont facilement vérifiées :

- $\left(A^T\right)^T=A$

- $(A B)^T=B^T A^T$

- $(A+B)^T=A^T+B^T$

<a name="C-3-3"/>

#### [C.3.3 Matrices symétriques](#c-3-3) ####

[Retour TOC](#toc)

Une matrice carrée $A \in \mathbb{R}^{n \times n}$ est symétrique si $A=A^T$. Elle est antisymétrique si $A=-A^T$. Il est facile de montrer que pour toute matrice $A \in \mathbb{R}^{n \times n}$, la matrice $A+A^T$ est symétrique et la matrice $A-A^T$ est antisymétrique. Il s'ensuit que toute matrice carrée $A \in \mathbb{R}^{n \times n}$ peut être représentée comme une somme d'une matrice symétrique et d'une matrice antisymétrique, puisque




$$
A=\frac{1}{2}\left(A+A^T\right)+\frac{1}{2}\left(A-A^T\right)
$$




et la première matrice à droite est symétrique, tandis que la seconde est antisymétrique. Il s'avère que les matrices symétriques sont très fréquentes dans la pratique et qu'elles possèdent de nombreuses propriétés intéressantes que nous allons examiner sous peu. Il est courant de désigner l'ensemble de toutes les matrices symétriques de taille $n$ par $\mathbb{S}^n$, de sorte que $A \in \mathbb{S}^n$ signifie que $A$ est une matrice symétrique $n \times n$ ;

<a name="C-3-4"/>

#### [C.3.4 La trace](#c-3-4) ####

[Retour TOC](#toc)

La trace d'une matrice carrée $A \in \mathbb{R}^{n \times n}$, notée ${tr}(A)$ (ou juste ${tr} A$ si les parenthèses sont évidemment implicites), est la somme des éléments diagonaux de la matrice :




$$
{tr} A=\sum_{i=1}^n A_{i i} .
$$




Comme décrit dans les notes de cours de CS229, la trace a les propriétés suivantes (incluses ici par souci d'exhaustivité) :

- Pour $A \in \mathbb{R}^{n \times n}, {tr} A={tr} A^T$.

- Pour $A, B \in \mathbb{R}^{n \times n}, {tr}(A+B)={tr} A+{tr} B$.

- Pour $A \in \mathbb{R}^{n \times n}, t\ dans\ \mathbb{R}, {tr}(t A)=t {tr} A$.

- Pour $A, B$ tels que $A B$ est carré, ${tr} A B={tr} B A$.

\- Pour $A, B, C$ tels que $A B C$ est carré, ${tr} A B C={tr} B C A={tr} C A B$, et ainsi de suite pour le produit de plusieurs matrices.

<a name="C-3-5"/>

#### [C.3.5 Normes](#c-3-5) ####

[Retour TOC](#toc)

La norme d'un vecteur $\|x\|$ est une mesure informelle de la "longueur" du vecteur. Par exemple, nous disposons de la norme euclidienne ou $\ell_2$ communément utilisée,




$$
\|x\|_2=\sqrt{\sideset{}{^n_{i=1}}\sum x_i^2}
$$




Notez que $\|x\|_2^2=x^T x$.

Plus formellement, une norme est toute fonction $f : \mathbb{R}^n \rightarrow \mathbb{R}$ qui satisfait 4 propriétés :

1. Pour tout $x \in \mathbb{R}^n, f(x) \geq 0$ (non-négativité).

2. $f(x)=0$ si et seulement si $x=0$ (définitude).

3. Pour tout $x \in \mathbb{R}^n, t \in \mathbb{R}, f(t x)=|t| f(x)$ (homogénéité).

4. Pour tout $x, y \in \mathbb{R}^n, f(x+y) \leq f(x)+f(y)$ (inégalité triangulaire).

D'autres exemples de normes sont la norme $\ell_1$,




$$
\|x\|_1=\sum_{i=1}^n\left|x_i\right|
$$




et la norme $\ell_{\infty}$,




$$
\|x\|_{\infty}=\max _i\left|x_i\right| .
$$




En fait, les trois normes présentées jusqu'ici sont des exemples de la famille des normes $\ell_p$, qui sont paramétrées par un nombre réel $p \geq 1$, et définies comme suit




$$
\|x\|_p=\biggl(\sideset{}{^n_{i=1}}\sum |x_i|^p\biggl)^{1 / p}
$$




Des normes peuvent également être définies pour les matrices, comme la norme de Frobenius,




$$
\|A\|_F=\sqrt{\sideset{}{^m_{i=1}}\sum \sideset{}{^n_{j=1}}\sum A_{i j}^2}=\sqrt{{tr}(A^T A)}
$$




De nombreuses autres normes existent, mais elles dépassent le cadre de cet examen.

<a name="C-3-6"/>

#### [C.3.6 Indépendance linéaire et rang](#c-3-6) ####

[Retour TOC](#toc)

Un ensemble de vecteurs $\{x_1, x_2, \ldots x_n\}$ est dit (linéairement) indépendant si aucun vecteur ne peut être représenté comme une combinaison linéaire des vecteurs restants. Inversement, un vecteur qui peut être représenté comme une combinaison linéaire des vecteurs restants est dit (linéairement) dépendant. Par exemple, si




$$
x_n=\sideset{}{^{n-1}_{i=1}}\sum \alpha_i x_i
$$




pour un certain $\{\alpha_1, \ldots, \alpha_{n-1}\}$ alors $x_n$ est dépendant de $\{x_1, \ldots, x_{n-1}\}$ ; sinon, il est indépendant de $\{x_1, \ldots, x_{n-1}\}$.

La colonne ${rank}$ d'une matrice $A$ est le plus grand nombre de colonnes de $A$ qui constituent un ensemble linéairement indépendant. On l'appelle souvent simplement le nombre de colonnes linéairement indépendantes, mais cette terminologie est un peu négligée, car il est possible que tout vecteur d'un ensemble $\{x_1, \ldots x_n\}$ puisse être exprimé comme une combinaison linéaire des vecteurs restants, même si un sous-ensemble des vecteurs peut être indépendant. De la même manière, le rang est le plus grand nombre de rangs de $A$ qui constituent un ensemble linéairement indépendant.

C'est un fait de base de l'algèbre linéaire, que pour toute matrice $A, {columnrank}(A)={rowrank}(A)$, et donc cette quantité est simplement désignée comme le ${rank}$ de $A$, noté ${rank}(A)$. Voici quelques propriétés de base du rang :

- Pour $A \in \mathbb{R}^{m \times n}, {rank}(A) \leq \min (m, n)$. Si ${rank}(A)=\min (m, n)$, alors $A$ est dit de rang complet.

- Pour $A \in \mathbb{R}^{m \times n}, {rank}(A)={rank}\left(A^T\right)$.

- Pour $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}, {rank}(A B) \leq \min ({rank}(A), {rank}(B))$.

- Pour $A, B \in \mathbb{R}^{m \times n}, {rank}(A+B) \leq {rank}(A)+{rank}(B)$.

<a name="C-3-7"/>

#### [C.3.7 L'inverse](#c-3-7) ####

[Retour TOC](#toc)

L'inverse d'une matrice carrée $A \in \mathbb{R}^{n \times n}$ est noté $A^{-1}$, et est la seule matrice telle que




$$
A^{-1} A=I=A A^{-1} .
$$




Il s'avère que $A^{-1}$ peut ne pas exister pour certaines matrices $A$ ; on dit que $A$ est inversible ou non singulière si $A^{-1}$ existe et non inversible ou singulière sinon. Nous connaissons déjà une condition d'inversibilité : il est possible de montrer que $A^{-1}$ existe si et seulement si $A$ est de rang complet. Nous verrons bientôt qu'il existe de nombreuses autres conditions suffisantes et nécessaires, en plus du rang complet, pour l'inversibilité. Les propriétés suivantes sont des propriétés de l'inverse ; toutes supposent que $A, B\ dans \mathbb{R}^{n \times n}$ sont non-singuliers :

- $(A^{-1})^{-1}=A$

- Si $A x=b$, nous pouvons multiplier par $A^{-1}$ des deux côtés pour obtenir $x=A^{-1} b$. Ceci démontre l'inverse par rapport au système original d'égalités linéaires avec lequel nous avons commencé cette revue.
- $(A B)^{-1}=B^{-1} A^{-1}$

- $(A^{-1})^T=(A^T)^{-1}$. Pour cette raison, cette matrice est souvent notée $A^{-T}$.

<a name="C-3-8"/>

#### [C.3.8 Matrices orthogonales](#c-3-8) ####

[Retour TOC](#toc)

Deux vecteurs $x, y \in \mathbb{R}^n$ sont orthogonaux si $x^T y=0$. Un vecteur $x \in \mathbb{R}^n$ est normalisé si $\|x\|_2=1$. Une matrice carrée $U \in \mathbb{R}^{n \times n}$ est orthogonale (notez les différentes significations lorsque vous parlez de vecteurs par rapport aux matrices) si toutes ses colonnes sont orthogonales les unes aux autres et sont normalisées (les colonnes sont alors dites orthonormales).

Il découle immédiatement de la définition de l'orthogonalité et de la normalité que




$$
U^T U=I=U U^T .
$$




En d'autres termes, l'inverse d'une matrice orthogonale est sa transposée. Notez que si $U$ n'est pas carrée - c'est-à-dire $U \in \mathbb{R}^{m \times n}, n < m$ - mais que ses colonnes sont toujours orthonormales, alors $U^T U=I$, mais $U U^T \neq I$. Nous n'utilisons généralement le terme orthogonal que pour décrire le cas précédent, où $U$ est carré.

Une autre propriété intéressante des matrices orthogonales est que le fait d'opérer sur un vecteur avec une matrice orthogonale ne changera pas sa norme euclidienne, c'est-à-dire..,




$$
\|U x\|_2=\|x\|_2
$$




pour tout $x \in \mathbb{R}^n, U \in \mathbb{R}^{n \times n}$ orthogonal.

<a name="C-3-9"/>

#### [C.3.9 Plage et nulspace d'une matrice](#c-3-9) ####

[Retour TOC](#toc)

L'étendue d'un ensemble de vecteurs $\{x_1, x_2, \ldots x_n\}$ est l'ensemble de tous les vecteurs qui peuvent être exprimés comme une combinaison linéaire de $\{x_1, \ldots, x_n\}$. C'est-à-dire,




$$
{span}\left(\{x_1, \ldots x_n\}\right)=\{v: v=\sideset{}{^n_{i=1}}\sum \alpha_i x_i, \quad \alpha_i \in \mathbb{R}\} .
$$




On peut montrer que si $\{x_1, \ldots, x_n\}$ est un ensemble de $n$ vecteurs linéairement indépendants, où chaque $x_i \in \mathbb{R}^n$, alors l'étendue $(\{x_1, \ldots x_n\})=\mathbb{R}^n$. En d'autres termes, tout vecteur $v\ dans\ \mathbb{R}^n$ peut être écrit comme une combinaison linéaire de $x_1$ à $x_n$. La projection d'un vecteur $y\ dans\ \mathbb{R}^m$ sur l'étendue de $\{x_1, \ldots, x_n\}$ (nous supposons ici $\left. x_i \ dans \mathbb{R}^m\right)$ est le vecteur $v \in {span}(\{x_1, \ldots x_n\})$, tel que $v$ soit aussi proche que possible de $y$, mesuré par la norme euclidienne $\|v-y|_2$. Nous désignons la projection par ${Proj}(y ;\{x_1, \ldots, x_n\})$ et pouvons la définir formellement comme,




$$
{Proj}(y ;\{x_1, \ldots x_n\})={argmin}_{v \in {span}(\{x_1, \ldots, x_n\})}\|y-v\|_2 .
$$




L'étendue (parfois aussi appelée espace des colonnes) d'une matrice $A \in \mathbb{R}^{m \times n}$, notée $\mathcal{R}(A)$, est l'étendue des colonnes de $A$. En d'autres termes,




$$
\mathcal{R}(A)=\{v \in \mathbb{R}^m: v=A x, x \in \mathbb{R}^n\} .
$$




En faisant quelques hypothèses techniques (à savoir que $A$ est de rang complet et que $n < m$ ), la projection d'un vecteur $y \in \mathbb{R}^m$ sur l'intervalle de $A$ est donnée par,




$$
{Proj}(y ; A)={argmin}_{v \in \mathcal{R}(A)}\|v-y\|_2=A(A^T A)^{-1} A^T y .
$$




Cette dernière équation devrait vous sembler extrêmement familière, puisqu'il s'agit presque de la même formule que nous avons dérivée en classe (et que nous allons bientôt dériver à nouveau) pour l'estimation des paramètres par les moindres carrés. En regardant la définition de la projection, il ne devrait pas être trop difficile de vous convaincre qu'il s'agit en fait du même objectif que celui que nous avons minimisé dans notre problème des moindres carrés (à l'exception d'un quadrillage de la norme, qui n'affecte pas le point optimal) et que ces problèmes sont donc naturellement très liés. Lorsque $A$ ne contient qu'une seule colonne, $a \in \mathbb{R}^m$, cela donne le cas particulier de la projection d'un vecteur sur une droite :




$$
{Proj}(y ; a)=\frac{a a^T}{a^T a} y .
$$




L'espace nul d'une matrice $A \in \mathbb{R}^{m \times n}$, noté $\mathcal{N}(A)$ est l'ensemble de tous les vecteurs qui sont égaux à 0 lorsqu'ils sont multipliés par $A$, c'est à dire,




$$
\mathcal{N}(A)=\{x \in \mathbb{R}^n: A x=0\} .
$$




Notez que les vecteurs dans $\mathcal{R}(A)$ sont de taille $m$, alors que les vecteurs dans $\mathcal{N}(A)$ sont de taille $n$, donc les vecteurs dans $\mathcal{R}\left(A^T\right)$ et $\mathcal{N}(A)$ sont tous deux dans $\mathbb{R}^n$. En fait, nous pouvons dire beaucoup plus. Il s'avère que




$$
\{w: w=u+v, u \in \mathcal{R}(A^T), v \in \mathcal{N}(A)\}=\mathbb{R}^n \text { and } \mathcal{R}(A^T) \cap \mathcal{N}(A)=\emptyset \text {. }
$$




En d'autres termes, $\mathcal{R}\left(A^T\right)$ et $\mathcal{N}(A)$ sont des sous-ensembles disjoints qui couvrent ensemble l'espace entier de $\mathbb{R}^n$. Les ensembles de ce type sont appelés des compléments orthogonaux, et nous désignons par $\mathcal{R}\left(A^T\right)=$ $\mathcal{N}(A)^{\perp}$ les compléments orthogonaux.

<a name="C-3-10"/>

#### [C.3.10 Le déterminant](#c-3-10) ####

[Retour TOC](#toc)

Le déterminant d'une matrice carrée $A \in \mathbb{R}^{n \times n}$, est une fonction ${det}$ : $\mathbb{R}^{n \times n}$ . $\mathbb{R}$, et est notée $|A|$ ou ${det} A$ (comme pour l'opérateur de trace, nous omettons généralement les parenthèses). La formule complète du déterminant donne peu d'intuition sur sa signification, aussi nous donnons d'abord trois propriétés déterminantes du déterminant, dont tout le reste découle (y compris la formule générale) :

1. Le déterminant de l'identité est $1,|I|=1$.

2. Étant donné une matrice $A \in \mathbb{R}^{n \times n}$, si nous multiplions une seule ligne de $A$ par un scalaire $t \in \mathbb{R}$, alors le déterminant de la nouvelle matrice est $t|A|$,

3. Si nous échangeons deux lignes quelconques $a_i^T$ et $a_j^T$ de $A$, alors le déterminant de la nouvelle matrice est $-|A|$, par exemple




$$
\left|\left[\begin{array}{ccc}
{-} & a_2^T & {-} \\
{-} & a_1^T & {-} \\
\vdots & \\
{-} & a_m^T & {-}
\end{array}\right]\right|=-|A|
$$




Cependant, ces propriétés ne donnent également que très peu d'intuition sur la nature du déterminant. Nous allons donc maintenant énumérer plusieurs propriétés qui découlent des trois propriétés ci-dessus :

- Pour $A \in \mathbb{R}^{n \times n},|A|=|A^T|$.

- Pour $A, B \in \mathbb{R}^{n \times n},|A B|=|A||B|$.

- Pour $A \in \mathbb{R}^{n \times n},|A|=0$ si et seulement si $A$ est singulier (i.e., non-invertible).

- Pour $A \in \mathbb{R}^{n \times n}$ et $A$ non singulier, $|A|^{-1}=1 /|A|$.

Avant de donner la définition générale du déterminant, on définit, pour $A \in \mathbb{R}^{n \times n}, A_{\backslash i \backslash j} \in$ $\mathbb{R}^{(n-1) \times(n-1)}$ la matrice qui résulte de la suppression de la $i$ $i^{ème}$ ligne et de la $j^{ème}$ colonne de $A$. La formule générale (récursive) du déterminant est la suivante




$$
\begin{aligned}
|A| & =\sideset{}{^n_{i=1}}\sum(-1)^{i+j} a_{i j}\left|A_{\backslash i, \backslash j}\right| \quad \text { (for any } j \in 1, \ldots, n \text { ) } \\
& =\sideset{}{^n_{j=1}}\sum(-1)^{i+j} a_{i j}\left|A_{\backslash i, \backslash j}\right| \quad \text { (for any } i \in 1, \ldots, n \text { ) }
\end{aligned}
$$




avec le cas initial que $|A|=a_{11}$ pour $A \in \mathbb{R}^{1 \times 1}$. Si nous devions développer cette formule complètement pour $A \in \mathbb{R}^{n \times n}$, il y aurait un total de $n$ ! ( $n$ factoriel) termes différents. Pour cette raison, nous n'écrivons même pas explicitement l'équation complète du déterminant pour les matrices supérieures à $3 \times 3$. Cependant, les équations des déterminants des matrices jusqu'à la taille de $3 \times 3$ sont assez courantes, et il est bon de les connaître :




$$
\begin{aligned}
&\left|\left[a_{11}\right]\right|=a_{11} \\
&\left|\left[\begin{array}{ll}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{array}\right]\right|= a_{11} a_{22}-a_{12} a_{21} \\
&\left|\left[\begin{array}{lll}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{array}\right]\right|=\begin{array}{c}
a_{11} a_{22} a_{33}+a_{12} a_{23} a_{31}+a_{13} a_{21} a_{32} \\
-a_{11} a_{23} a_{32}-a_{12} a_{21} a_{33}-a_{13} a_{22} a_{31}
\end{array}
\end{aligned}
$$




L'adjoint classique (souvent juste appelé l'adjoint) d'une matrice $A \in \mathbb{R}^{n \times n}$, est noté ${adj}(A)$, et défini comme suit




$$
{adj}(A) \in \mathbb{R}^{n \times n}, \quad({adj}(A))_{i j}=(-1)^{i+j}\left|A_{\backslash j, \backslash i}\right|
$$




(notez l'inversion des indices $A \backslash, \backslash$ ). On peut montrer que pour tout élément non singulier $A \in \mathbb{R}^{n \times n}$,




$$
A^{-1}=\frac{1}{|A|} {adj}(A) .
$$




Bien qu'il s'agisse d'une belle formule "explicite" pour l'inverse d'une matrice, nous devons noter que, numériquement, il existe en fait des moyens beaucoup plus efficaces de calculer l'inverse.

<a name="C-3-11"/>

#### [C.3.11 Formes quadratiques et matrices semi-définies positives](#c-3-11) ####

[Retour TOC](#toc)

Étant donné une matrice carrée $A \in \mathbb{R}^{n \times n}$ et un vecteur $x \in \mathbb{R}$, la valeur scalaire $x^T A x$ est appelée une forme quadratique. En écrivant explicitement, on voit que




$$
x^T A x=\sideset{}{^n_{i=1}}\sum \sideset{}{^n_{j=1}}\sum A_{i j} x_i x_j .
$$


Notez que,




$$
x^T A x=(x^T A x)^T=x^T A^T x=x^T(\frac{1}{2} A+\frac{1}{2} A^T) x
$$




c'est-à-dire que seule la partie symétrique de $A$ contribue à la forme quadratique. Pour cette raison, nous supposons souvent implicitement que les matrices apparaissant dans une forme quadratique sont symétriques.

Nous donnons les définitions suivantes :

- Une matrice symétrique $A \in \mathbb{S}^n$ est définie positive (DP) si pour tous les vecteurs non nuls $x \in \mathbb{R}^n, x^T A x>0$. Ceci est généralement noté $A \succ 0$ (ou juste $A > 0$ ), et souvent l'ensemble de toutes les matrices définies positives est noté $\mathbb{S}_{++}^n$.
- Une matrice symétrique $A \in \mathbb{S}^n$ est semi-définie positive (PSD) si pour tous les vecteurs $x^T A x \geq$ 0 . Ceci s'écrit $A \succeq 0$ (ou juste $A \geq 0$ ), et l'ensemble de toutes les matrices semi-définies positives est souvent noté $\mathbb{S}_{+}^n$.
- De même, une matrice symétrique $A \in \mathbb{S}^n$ est définie négative (DN), notée $A \prec 0$ (ou juste $A<0$ ) si pour tout $x \in \mathbb{R}^n$ non nul, $x^T A x<0$.
- De même, une matrice symétrique $A \in \mathbb{S}^n$ est semi-définie négative (NSD), notée $A \succeq 0$ (ou juste $A \leq 0$ ) si pour tout $x \in \mathbb{R}^n, x^T A x \leq 0$.
- Enfin, une matrice symétrique $A \in \mathbb{S}^n$ est indéfinie, si elle n'est ni semi-définie positive ni semi-définie négative - c'est-à-dire s'il existe $x_1, x_2 \in \mathbb{R}^n$ tels que $x_1^T A x_1 > 0$ et $x_2^T A x_2 < 0$.

Il devrait être évident que si $A$ est définie positive, alors $-A$ est définie négative et vice versa. De même, si $A$ est semi-définie positive, alors $-A$ est semi-définie négative et vice versa. Si $A$ est indéfinie, alors $-A$ l'est aussi. On peut également montrer que les matrices définies positives et définies négatives sont toujours inversibles.

Enfin, il existe un type de matrice définie positive qui apparaît fréquemment et qui mérite donc une mention spéciale. Pour toute matrice $A \in \mathbb{R}^{m \times n}$ (pas nécessairement symétrique ou même carrée), la matrice $G=A^T A$ (parfois appelée matrice de Gram) est toujours semi-définie positive. De plus, si $m \geq n$ (et nous supposons par commodité que $A$ est de rang complet), alors $G=A^T A$ est définie positive.

<a name="C-3-12"/>

#### [C.3.12 Valeurs propres et vecteurs propres](#c-3-12) ####

[Retour TOC](#toc)

Étant donné une matrice carrée $A \in \mathbb{R}^{n \times n}$, on dit que $\lambda \in \mathbb{C}$ est une valeur propre de $A$ et que $x \in \mathbb{C}^n$ est le vecteur propre correspondant si




$$
A x=\lambda x, \quad x \neq 0 .
$$




(Notez que $\lambda$ et les entrées de $x$ sont en fait dans $\mathbb{C}$, l'ensemble des nombres complexes, et pas juste les réels ; nous verrons bientôt pourquoi cela est nécessaire. Ne vous inquiétez pas de cette technicité pour l'instant, vous pouvez penser aux vecteurs complexes de la même manière qu'aux vecteurs réels.)

Intuitivement, cette définition signifie que la multiplication de $A$ par le vecteur $x$ donne un nouveau vecteur qui pointe dans la même direction que $x$, mais mis à l'échelle par un facteur $\lambda$. Notez également que pour tout vecteur propre $x \in \mathbb{C}^n$, et tout scalaire $t \in \mathbb{C}, A(c x)=c A x=c \lambda x=\lambda(c x)$, donc $c x$ est également un vecteur propre. C'est pourquoi, lorsque nous parlons du "vecteur propre" associé à $\lambda$, nous supposons généralement que le vecteur propre est normalisé pour avoir une longueur de 1 (cela crée encore une certaine ambiguïté, puisque $x$ et $-x$ seront tous deux des vecteurs propres, mais nous devons nous en accommoder).

Nous pouvons réécrire l'équation ci-dessus pour dire que $(\lambda, x)$ est une paire valeur propre-vecteur propre de $A$ si,




$$
(\lambda I-A) x=0, \quad x \neq 0 .
$$




Mais $(\lambda I-A) x=0$ a une solution non nulle à $x$ si et seulement si $(\lambda I-A)$ a un espace nul non vide, ce qui n'est le cas que si $(\lambda I-A)$ est singulier, c'est-à-dire,




$$
|(\lambda I-A)|=0 \text {. }
$$




Nous pouvons maintenant utiliser la définition précédente du déterminant pour développer cette expression en un (très grand) polynôme en $\lambda$, où $\lambda$ aura le degré maximum $n$. Nous trouvons ensuite les $n$ racines (éventuellement complexes) de ce polynôme pour trouver les $n$ valeurs propres $\lambda_1, \ldots, \lambda_n$. Pour trouver le vecteur propre correspondant à la valeur propre $\lambda_i$, il suffit de résoudre l'équation linéaire $\left(\lambda_i I-A\right) x=0$. Il convient de noter que cette méthode n'est pas celle qui est réellement utilisée dans la pratique pour calculer numériquement les valeurs propres et les vecteurs propres (rappelez-vous que l'expansion complète du déterminant comporte $n$ ! termes) ; il s'agit plutôt d'un argument mathématique.

Voici les propriétés des valeurs propres et des vecteurs propres (dans tous les cas, on suppose que $A \in \mathbb{R}^{n \times n}$ a des valeurs propres $\lambda_i, \ldots, \lambda_n$ et les vecteurs propres associés $x_1, \ldots x_n$ ) :

- La trace d'un $A$ est égale à la somme de ses valeurs propres,




$$
{tr} A=\sideset{}{^n_{i=1}}\sum \lambda_i
$$




- Le déterminant de $A$ est égal au produit de ses valeurs propres,




$$
|A|=\sideset{}{^n_{i=1}}\prod \lambda_i .
$$




- Le rang de $A$ est égal au nombre de valeurs propres non nulles de $A$.

- Si $A$ est non singulier, alors $1 / \lambda_i$ est une valeur propre de $A^{-1}$ à laquelle est associé un vecteur propre $x_i$, c'est-à-dire que $A^{-1} x_i=(1 / \lambda_i) x_i$.

- Les valeurs propres d'une matrice diagonale $D={diag}(d_1, \ldots d_n)$ sont juste les entrées diagonales $d_1, \ldots d_n$.

Nous pouvons écrire simultanément toutes les équations des vecteurs propres comme suit




$$
A X=X \Lambda
$$




où les colonnes de $X \in \mathbb{R}^{n \times n}$ sont les vecteurs propres de $A$ et $\Lambda$ est une matrice diagonale dont les entrées sont les valeurs propres de $A$, à savoir,




$$
X \in \mathbb{R}^{n \times n}=\left[\begin{array}{cccc}
\mid & \mid & & \mid \\
x_1 & x_2 & \cdots & x_n \\
\mid & \mid & & \mid
\end{array}\right], \Lambda={diag}(\lambda_1, \ldots, \lambda_n)
$$




Si les vecteurs propres de $A$ sont linéairement indépendants, alors la matrice $X$ sera inversible, donc $A=X \Lambda X^{-1}$. Une matrice qui peut être écrite sous cette forme est dite diagonalisable.

<a name="C-3-13"/>

#### [C.3.13 Valeurs propres et vecteurs propres des matrices symétriques](#c-3-13) ####

[Retour TOC](#toc)

Deux propriétés remarquables apparaissent lorsque l'on examine les valeurs propres et les vecteurs propres d'une matrice symétrique $A \in \mathbb{S}^n$. Premièrement, on peut montrer que toutes les valeurs propres de $A$ sont réelles. Deuxièmement, les vecteurs propres de $A$ sont orthonormés, c'est-à-dire que la matrice $X$ définie ci-dessus est une matrice orthogonale (pour cette raison, nous désignons la matrice des vecteurs propres par $U$ dans ce cas). Nous pouvons donc représenter $A$ comme $A=U \Lambda U^T$, en nous rappelant que l'inverse d'une matrice orthogonale est juste sa transposée.

En utilisant ceci, nous pouvons montrer que le caractère définitif d'une matrice dépend entièrement du signe de ses valeurs propres. Supposons que $A \in \mathbb{S}^n=U \Lambda U^T$. Alors




$$
x^T A x=x^T U \Lambda U^T x=y^T \Lambda y=\sideset{}{^n_{i=1}}\sum \lambda_i y_i^2
$$




où $y=U^T x$ (et puisque $U$ est de rang complet, tout vecteur $y \in \mathbb{R}^n$ peut être représenté sous cette forme). Comme $y_i^2$ est toujours positif, le signe de cette expression dépend entièrement des $\lambda_i$. Si tous les $\lambda_i>0$, alors la matrice est définie positive ; si tous les $\lambda_i \geq 0$, elle est semi-définie positive. De même, si tous les $\lambda_i<0$ ou $\lambda_i \leq 0$, alors $A$ est définie négative ou semi-définie négative respectivement. Enfin, si $A$ a des valeurs propres à la fois positives et négatives, il est indéfini.

Une application où les valeurs propres et les vecteurs propres sont fréquemment utilisés est la maximisation d'une fonction d'une matrice. En particulier, pour une matrice $A \in \mathbb{S}^n$, on considère le problème de maximisation suivant,




$$
\max _{x \in \mathbb{R}^n} x^T A x \quad \text { subject to }\|x\|_2^2=1
$$




c'est-à-dire que nous voulons trouver le vecteur (de norme 1) qui maximise la forme quadratique. En supposant que les valeurs propres sont ordonnées comme suit : $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_n$, le $x$ optimal pour ce problème d'optimisation est $x_1$, le vecteur propre correspondant à $\lambda_1$. Dans ce cas, la valeur maximale de la forme quadratique est $\lambda_1$. De même, la solution optimale du problème de minimisation,




$$
\min _{x \in \mathbb{R}^n} x^T A x \quad \text { subject to }\|x\|_2^2=1
$$




est $x_n$, le vecteur propre correspondant à $\lambda_n$, et la valeur minimale est $\lambda_n$. Ceci peut être prouvé en faisant appel à la forme vecteur propre-valeur propre de $A$ et aux propriétés des matrices orthogonales. Cependant, dans la section suivante, nous verrons un moyen de le démontrer directement en utilisant le calcul matriciel.

<a name="C-4"/>

### [C.4 Calcul matriciel](#c-4) ###

[Retour TOC](#toc)

Alors que les sujets des sections précédentes sont généralement abordés dans un cours standard d'algèbre linéaire, un sujet qui ne semble pas être abordé très souvent (et que nous utiliserons abondamment) est l'extension du calcul aux vecteurs. Bien que le calcul que nous utilisons soit relativement trivial, la notation peut souvent faire paraître les choses beaucoup plus difficiles qu'elles ne le sont. Dans cette section, nous présentons quelques définitions de base du calcul matriciel et fournissons quelques exemples.

<a name="C-4-1"/>

#### [C.4.1 Le gradient](#c-4-1) ####

[Retour TOC](#toc)

Supposons que $f : \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$ est une fonction qui prend en entrée une matrice $A$ de taille $m \times n$ et retourne une valeur réelle. Alors le gradient de $f$ (par rapport à $A \in \mathbb{R}^{m \times n}$ ) est la matrice des dérivées partielles, définie comme suit :




$$
\nabla_A f(A) \in \mathbb{R}^{m \times n}=\left[\begin{array}{cccc}
\frac{\partial f(A)}{\partial A_1} & \frac{\partial f(A)}{\partial A_{13}} & \cdots & \frac{\partial f(A)}{\partial A_1} \\
\frac{\partial f(A)}{\partial A_{21}} & \frac{\partial f(A)}{\partial A_{22}} & \cdots & \frac{\partial f(A)}{\partial A_{2 n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f(A)}{\partial A_{m 1}} & \frac{\partial f(A)}{\partial A_{m 2}} & \cdots & \frac{\partial f(A)}{\partial A_{m n}}
\end{array}\right]
$$




c'est-à-dire, une matrice de $m \times n$ avec




$$
\left(\nabla_A f(A)\right)_{i j}=\frac{\partial f(A)}{\partial A_{i j}} .
$$




Notez que la taille de $\nabla_A f(A)$ est toujours la même que la taille de $A$. Donc, si, en particulier, $A$ est juste un vecteur $x \in \mathbb{R}^n$,




$$
\nabla_x f(x)=\left[\begin{array}{c}
\frac{\partial f(x)}{\partial x_1} \\
\frac{\partial f(x)}{\partial x_2} \\
\vdots \\
\frac{\partial f(x)}{\partial x_n}
\end{array}\right] .
$$




Il est très important de rappeler que le gradient d'une fonction n'est défini que si la fonction est à valeur réelle, c'est-à-dire si elle renvoie une valeur scalaire. On ne peut pas, par exemple, prendre le gradient de $A x, A \in \mathbb{R}^{n \times n}$ par rapport à $x$, puisque cette quantité est à valeur vectorielle.

Il découle directement des propriétés équivalentes des dérivées partielles que :

- $\nabla_x(f(x)+g(x))=\nabla_x f(x)+\nabla_x g(x)$.

- Pour $t \in \mathbb{R}, \nabla_x(t f(x))=t \nabla_x f(x)$.

Il est un peu plus difficile de déterminer l'expression correcte de $\nabla_x f(A x), A \in \mathbb{R}^{n \times n}$, mais c'est également faisable (en fait, vous devrez résoudre ce problème pour un devoir à la maison).

<a name="C-4-2"/>

#### [C.4.2 Le hessien](#c-4-2) ####

[Retour TOC](#toc)

Supposons que $f : \mathbb{R}^n \rightarrow \mathbb{R}$ est une fonction qui prend un vecteur dans $\mathbb{R}^n$ et retourne un nombre réel. Alors la matrice hessienne par rapport à $x$, écrite $\nabla_x^2 f(x)$ ou simplement $H$ est la matrice $n \times n$ des dérivées partielles,




$$
\nabla_x^2 f(x) \in \mathbb{R}^{n \times n}=\left[\begin{array}{cccc}
\frac{\partial^2 f(x)}{\partial x_1^1} & \frac{\partial^2 f(x)}{\partial \partial_1 \theta x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial_{x_n} x_n} \\
\frac{\partial^2 f(x)}{\partial x_x \partial \theta_1} & \frac{\partial^2 f(x)}{\partial x_2^2} & \cdots & \frac{\partial^2 f(x)}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f(x)}{\partial x_x \theta_1} & \frac{\partial^2 f(x)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_n^2}
\end{array}\right] .
$$


En d'autres termes, $\nabla_x^2 f(x) \in \mathbb{R}^{n \times n}$, avec




$$
\left(\nabla_x^2 f(x)\right)_{i j}=\frac{\partial^2 f(x)}{\partial x_i \partial x_j} .
$$




Notez que le hessien est toujours symétrique, puisque




$$
\frac{\partial^2 f(x)}{\partial x_i \partial x_j}=\frac{\partial^2 f(x)}{\partial x_j \partial x_i}
$$




Tout comme le gradient, le hessien n'est défini que lorsque $f(x)$ est à valeur réelle.

Il est naturel de considérer le gradient comme l'analogue de la dérivée première pour les fonctions de vecteurs, et le hessien comme l'analogue de la dérivée seconde (et les symboles que nous utilisons suggèrent également cette relation). Cette intuition est généralement correcte, mais il y a quelques mises en garde à garder à l'esprit.

Premièrement, pour les fonctions à valeur réelle d'une variable $f : \mathbb{R} \rightarrow \mathbb{R}$, il est une définition de base que la dérivée seconde est la dérivée de la dérivée première, c'est-à-dire,




$$
\frac{\partial^2 f(x)}{\partial x^2}=\frac{\partial}{\partial x} \frac{\partial}{\partial x} f(x) .
$$




Cependant, pour les fonctions d'un vecteur, le gradient de la fonction est un vecteur, et nous ne pouvons pas prendre le gradient d'un vecteur - c'est-à-dire,




$$
\nabla_x \nabla_x f(x)=\nabla_x\left[\begin{array}{c}
\frac{\partial f(x)}{\partial x_1} \\
\frac{\partial f(x)}{\partial x_2} \\
\vdots \\
\frac{\partial f(x)}{\partial x_1}
\end{array}\right]
$$


et cette expression n'est pas définie. Par conséquent, il n'est pas vrai que le hessien est le gradient du gradient. Cependant, c'est presque vrai, dans le sens suivant : si nous regardons la $i^{ème}$ entrée du gradient $\left(\nabla_x f(x)\right)_i=\partial f(x) / \partial x_i$, et prenons le gradient par rapport à $x$ nous obtenons




$$
\nabla_x \frac{\partial f(x)}{\partial x_i}=\left[\begin{array}{c}
\frac{\partial^2 f(x)}{\partial x_i \partial x_1} \\
\frac{\partial^2 f(x)}{\partial x_i \partial x_2} \\
\vdots \\
\frac{\partial f(x)}{\partial x_i \partial x_n}
\end{array}\right]
$$




qui est la $i^{ème}$ colonne (ou ligne) du Hessien. Par conséquent,




$$
\nabla_x^2 f(x)=\left[\begin{array}{llll}
\nabla_x\left(\nabla_x f(x)\right)_1 & \nabla_x\left(\nabla_x f(x)\right)_2 & \cdots & \nabla_x\left(\nabla_x f(x)\right)_n
\end{array}\right] .
$$




Si cela ne nous dérange pas d'être un peu négligents, nous pouvons dire que (essentiellement) $\nabla_x^2 f(x)=\nabla_x\left(\nabla_x f(x)\right)^T$, tant que nous comprenons que cela signifie réellement prendre le gradient de chaque entrée de $\left(\nabla_x f(x)\right)^T$, et non le gradient du vecteur entier.

Enfin, notez que bien que nous puissions prendre le gradient par rapport à une matrice $A \in \mathbb{R}^n$, pour les besoins de ce cours, nous n'envisagerons de prendre le hessien que par rapport à un vecteur $x \in \mathbb{R}^n$. Ceci est simplement une question de commodité (et le fait qu'aucun des calculs que nous faisons ne nécessite de trouver le hessien par rapport à une matrice), puisque le hessien par rapport à une matrice devrait représenter toutes les dérivées partielles $\partial^2 f(A) /\left(\partial A_{i j} \partial A_{k \ell}\right)$, et il est plutôt encombrant de représenter ceci sous forme de matrice.

<a name="C-4-3"/>

#### [C.4.3 Gradients et hessian des fonctions quadratiques et linéaires](#c-4-3) ####

[Retour TOC](#toc)

Essayons maintenant de déterminer les matrices du gradient et de la hessienne de quelques fonctions simples. Il convient de noter que tous les gradients donnés ici sont des cas particuliers des gradients donnés dans les notes de cours de CS229.

Pour $x \in \mathbb{R}^n$, soit $f(x)=b^T x$ pour un vecteur connu $b \in \mathbb{R}^n$. Alors




$$
f(x)=\sideset{}{^n_{i=1}}\sum b_i x_i
$$


donc




$$
\frac{\partial f(x)}{\partial x_k}=\frac{\partial}{\partial x_k} \sideset{}{^n_{i=1}}\sum b_i x_i=b_k .
$$




A partir de là, nous pouvons facilement voir que $\nabla_x b^T x=b$. Ceci doit être comparé à la situation analogue dans le calcul à une variable, où $\partial /(\partial x) a x=a$.

Considérons maintenant la fonction quadratique $f(x)=x^T A x$ pour $A \in \mathbb{S}^n$. Rappelez-vous que




$$
f(x)=\sideset{}{^n_{i=1}}\sum \sideset{}{^n_{j=1}}\sum A_{i j} x_i x_j
$$


donc




$$
\frac{\partial f(x)}{\partial x_k}=\frac{\partial}{\partial x_k} \sideset{}{^n_{i=1}}\sum \sideset{}{^n_{j=1}}\sum A_{i j} x_i x_j=\sideset{}{^n_{i=1}}\sum A_{i k} x_i+\sideset{}{^n_{j=1}}\sum A_{k j} x_j=2 \sideset{}{^n_{i=1}}\sum A_{k i} x_i
$$




où la dernière égalité suit puisque $A$ est symétrique (ce que nous pouvons supposer sans risque, puisqu'il apparaît sous une forme quadratique). Notez que la $k^{ème}$ entrée de $\nabla_x f(x)$ est juste le produit interne de la $k^{ème}$ ligne de $A$ et de $x$. Par conséquent, $\nabla_x x^T A x=2 A x$. Encore une fois, cela devrait vous rappeler le fait analogue dans le calcul à une variable, à savoir que $\partial /(\partial x) a x^2=2 a x$.

Enfin, examinons le hessien de la fonction quadratique $f(x)=x^T A x$ (il devrait être évident que le hessien d'une fonction linéaire $b^T x$ est nul). C'est encore plus facile que de déterminer le gradient de la fonction, puisque


$$
\frac{\partial^2 f(x)}{\partial x_k \partial x_{\ell}}=\frac{\partial^2}{\partial x_k \partial x_{\ell}} \sideset{}{^n_{i=1}}\sum \sideset{}{^n_{j=1}}\sum A_{i j} x_i x_j=A_{k \ell}+A_{\ell k}=2 A_{k \ell} .
$$


Par conséquent, il devrait être clair que $\nabla_x^2 x^T A x=2 A$, ce qui devrait être tout à fait attendu (et à nouveau analogue au fait à une seule variable que $\left.\partial^2 /\left(\partial x^2\right) a x^2=2 a\right)$.

Pour résumer,

- $\nabla_x b^T x=b$

- $\nabla_x x^T A x=2 A x$ (if $A$ symmetric)

- $\nabla_x^2 x^T A x=2 A$ (if $A$ symmetric)

<a name="C-4-4"/>

#### [C.4.4 Les moindres carrés](#c-4-4) ####

[Retour TOC](#toc)

Appliquons les équations que nous avons obtenues dans la section précédente pour dériver les équations des moindres carrés. Supposons que l'on nous donne des matrices $A \in \mathbb{R}^{m \times n}$ (pour simplifier, nous supposons que $A$ est de rang complet) et un vecteur $b \in \mathbb{R}^m$ tel que $b \notin \mathcal{R}(A)$. Dans cette situation, nous ne serons pas capables de trouver un vecteur $x \in \mathbb{R}^n$, tel que $A x = b$, donc à la place nous voulons trouver un vecteur $x$ tel que $A x$ soit aussi proche que possible de $b$, tel que mesuré par le carré de la norme euclidienne $\|A x-b\|_2^2$.

En utilisant le fait que $\|x\|_2^2=x^T x$, nous avons




$$
\begin{aligned}
\|A x-b\|_2^2 & =(A x-b)^T(A x-b) \\
& =x^T A^T A x-2 b^T A x+b^T b
\end{aligned}
$$




En prenant le gradient par rapport à $x$ nous avons, et en utilisant les propriétés que nous avons dérivées dans la section précédente




$$
\begin{aligned}
\nabla_x\left(x^T A^T A x-2 b^T A x+b^T b\right) & =\nabla_x x^T A^T A x-\nabla_x 2 b^T A x+\nabla_x b^T b \\
& =2 A^T A x-2 A^T b
\end{aligned}
$$


En mettant cette dernière expression égale à zéro et en résolvant pour $x$ on obtient les équations normales




$$
x=\left(A^T A\right)^{-1} A^T b
$$




ce qui est identique à ce que nous avons dérivé en classe.

<a name="C-4-5"/>

#### [C.4.5 Gradients du déterminant](#c-4-5) ####

[Retour TOC](#toc)

Considérons maintenant une situation où nous trouvons le gradient d'une fonction par rapport à une matrice, à savoir pour $A \in \mathbb{R}^{n \times n}$, nous voulons trouver $\nabla_A|A|$. Rappelons de notre discussion sur les déterminants que




$$
|A|=\sideset{}{^n_{i=1}}\sum(-1)^{i+j} A_{i j}|A_{\backslash i, \backslash j}| \quad \text { (for any } j \in 1, \ldots, n \text { ) }
$$




donc




$$
\frac{\partial}{\partial A_{k \ell}}|A|=\frac{\partial}{\partial A_{k \ell}} \sideset{}{^n_{i=1}}\sum(-1)^{i+j} A_{i j}|A_{\backslash i, \backslash j}|=(-1)^{k+\ell}|A_{\backslash k, \backslash \ell}|=({adj}(A))_{\ell k} .
$$




Il s'ensuit immédiatement des propriétés de l'adjoint que




$$
\nabla_A|A|=({adj}(A))^T=|A| A^{-T}
$$




Considérons maintenant la fonction $f : \mathbb{S}_{++}^n \rightarrow \mathbb{R}, f(A)=\log |A|$. Notez que nous devons restreindre le domaine de $f$ aux matrices définies positives, car cela garantit que $|A|>0$, de sorte que le $\log$ de $|A|$ est un nombre réel. Dans ce cas, nous pouvons utiliser la règle de la chaîne (rien d'extraordinaire, juste la règle de la chaîne ordinaire du calcul à une variable) pour voir que




$$
\frac{\partial \log |A|}{\partial A_{i j}}=\frac{\partial \log |A|}{\partial|A|} \frac{\partial|A|}{\partial A_{i j}}=\frac{1}{|A|} \frac{\partial|A|}{\partial A_{i j}} .
$$




Il est donc évident que




$$
\nabla_A \log |A|=\frac{1}{|A|} \nabla_A|A|=A^{-1},
$$




où nous pouvons laisser tomber la transposition dans la dernière expression car $A$ est symétrique. Notez la similitude avec le cas à valeur unique, où $\partial /(\partial x) \log x=1 / x$.

<a name="C-4-6"/>

#### [C.4.6 Valeurs propres en tant qu'optimisation](#c-4-6) ####

[Retour TOC](#toc)

Enfin, nous utilisons le calcul matriciel pour résoudre un problème d'optimisation d'une manière qui mène directement à l'analyse des valeurs propres/vecteurs propres. Considérons le problème d'optimisation suivant, soumis à des contraintes d'égalité :




$$
\max _{x \in \mathbb{R}^n} x^T A x \quad \text { subject to }\|x\|_2^2=1
$$




pour une matrice symétrique $A \in \mathbb{S}^n$. Une façon standard de résoudre les problèmes d'optimisation avec des contraintes d'égalité est de former le Lagrangien, une fonction objectif qui inclut les contraintes d'égalité. ${ }^2$ Le lagrangien dans ce cas peut être donné par




$$
\mathcal{L}(x, \lambda)=x^T A x-\lambda x^T x
$$




où $\lambda$ est appelé le multiplicateur de Lagrange associé à la contrainte d'égalité. On peut établir que pour que $x^*$ soit un point optimal du problème, le gradient du Lagrangien doit être nul à $x^*$ (ce n'est pas la seule condition, mais elle est requise). C'est-à-dire ,




$$
\nabla_x \mathcal{L}(x, \lambda)=\nabla_x\left(x^T A x-\lambda x^T x\right)=2 A^T x-2 \lambda x=0 .
$$




Remarquez que c'est juste l'équation linéaire $A x=\lambda x$. Cela montre que les seuls points qui peuvent éventuellement maximiser (ou minimiser) $x^T A x$ en supposant que $x^T x=1$ sont les vecteurs propres de $A$.
