# Make Your own neural network in Python

Sigmoids: $\frac{1}{1+e^{-x}}$

## Backward error propagation

$$ e_{output,1} = (Truth - output) $$

$$e_{hidden,1} = e_{output,1} \cdot \frac{w_{11}}{w_{11}+w_{21}} + e_{output,2} \cdot \frac{w_{12}}{w_{12}+w_{22}} $$

Donc on fait une recombinaison d'erreur en rétro propagation. Évidemment on utilise des matrices pour les calculs.

$$error_{hidden} = w_{\text{hidden output}}^T \cdot error_{\text{output}}$$

## Adjusting the link Weights
To  have the gradient of the error we do 
$$\frac{\partial E}{\partial w_{jk}} = \frac{\partial }{\partial w_{jk}} (t_k - o_k)$$
Où $w_{jk}$ représente les poids, $t_k$ représente la valeur correct et $o_k$ la valeur obtenu.

La version simplifiée est 
$$\frac{\partial E}{\partial w_{jk}} = -2 (t_k - o_k) \frac{\partial o_k }{\partial w_{jk}}$$ 
car on ne dépend que de $o_k$ qui est le seul à varier selon $w_{jk}$

On sait que $o_k$ est le résultat de précédent résultat de layer différent tel que 
$$o_k = sigmoid (\sum_j w_{jk} o_j)$$

$$\frac{\partial}{\partial x}sigmoid(x) = sigmoid(x)(1 - sigmoid(x))$$

Résultat final: 
$$\frac{\partial E}{\partial w_{jk}} = -2 (t_k - o_k) sigmoid (\sum_j w_{jk} o_j)(1 - sigmoid(\sum_j w_{jk} o_j))o_j$$
On peut enlever le 2 car c'est une constante sans importance

#### Changer le poids
$$\text{new } w_{jk} = \text{old } w_{jk} - \alpha \cdot \frac{\partial E}{\partial w_{jk}} $$

Où $\alpha$ est un facteur qui modère l'importance du gradient. L'expression avec des matrices est:

$$ \Delta W_{jk} = \alpha \ast E_k \ast sigmoid(O_k) \ast (1-sigmoid(O_k)) \cdot O_j^T   $$

#### Prepare data
Avec les sigmoids il est techniquement impossible d'arriver à $0$ ou $1$. Donc on peut limiter la portée entre $0.01$ et $0.99$.

Pour débuter, on prend souvent des poids de $\pm 1$ pour éviter cette saturation dès le départ. Pour initaliser le tout, on distribue de manière aléatoire dans une portée qui est l'inverse de la racine carrée du nombres de noeuds. (eg: si 3 noeuds $1/\sqrt{3}$) On utilise aussi $tanh$ pour une distribution moins aléatoire et plus proche d'une variation nulle.

Il ne faut jamais mettre les poids initiaux à une valeur constante pour tous et surtout pas à $0$.

## A gentle Start with Python

```python
plt.imshow(DATA, interpolation='nearest')
```
Utile pour plot des données (matrices, ...) et utilise une interpolation pour que tout soit plus simple à comprendre.