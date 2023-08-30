# Matplotlib for Python
- [Matplotlib for Python](#matplotlib-for-python)
  - [Basics of Matplotlib](#basics-of-matplotlib)
    - [How to draw Various plot elements](#how-to-draw-various-plot-elements)
      - [Tick](#tick)
      - [Spine](#spine)
      - [Legend](#legend)
      - [Annotate](#annotate)
      - [Text](#text)
      - [Grid](#grid)
      - [Title](#title)


![](resume.png)

## Basics of Matplotlib
- Figure: une sorte de toile groupant tous les graphes
- Axes: un graphe
- Axis: un axe sur un graphe
- Artist: est une collection d'objet qui font parti du graphe. (eg: *title, legend, axis, spine, grid, tick*)

### How to draw Various plot elements
3 façons de faire la même chose
```python
plt.plot(x,y)
```
```python
ax = plt.subplot()
ax.plot(x, y)
```
```python
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.plot(x, y)
```

#### Tick
Le tick est une marque sur l'axe des coordonnées.
- `which`: change x/y ou major/minor
- `color`: change la couleur
- `labelrotation`: rotation du label
- `width`: change la largeur en point
- `length`: change la longueur en point
- `direction`: choisis la direction du tick
  
utile pour changer le nombre de trait sur les axes
![](tick.png)

#### Spine
Les `spine` sont les lignes qui connectent les ticks aux un aux autres. Donc on a *top, bottom, left, right*

On peut déplacer la spine avec `set_position(position_type, amount)` Il y a différent `position_type`:
- `axes`: pour mettre le spine à des coordonnées 
- `data`: pour mettre le spine aux coordonnées des data
- `outward`: pour le mettre hors des data

#### Legend
Permet de donner une description pour un élément d'une figure. Il y a 2 façons d'ajouter des labels:

```python
fig, axe = plt.subplots(dpi=800)

axe.plot(points, y1)
axe.plot(points, y2)
axe.legend(["tanh", "sin"])
```
```python
fig, axe = plt.subplots(dpi=800)

axe.plot(points, y1, label="tanh")
axe.plot(points, y2, label="sin")
axe.legend()
```
On peut énormément customiser ces labels via:
- `loc`: On peut choisir où se trouve la légende (eg: `upper left`, `upper right`, ...)
- `fontsize`: Taille de la police
- ``ncol``: Nombre de colonne pour la légdende
- ``frameon``: La légende va être dessiné ou non
- ``shadow``: Pour ajouter de l'ombre
- ``title``: Donne un titre à la légende
- ``facecolor``: Change la couleur de fond
- ``edgecolor``: Change la couleur du contour

#### Annotate
Une annotation est un morceau de texte qui identifie une donnée spécifique. On met en premier argument le texte qu'on veut ajouter. Voici quelques annotations basiques:

- ``xy``: Le point $(x,y)$ qui doit être annoté (un tuple)
- ``xytext``: La position où le texte va apparaitre (si rien alors sera égale à `xy`)
- ``xycoords``: Le système de coordonnées pour `xy`
- ``textcoords``: Le système de coordonnées pour `xytext`
- ``arrowprops``: Propriétés des flèches dessinées entre `xy` et `xytext`
- `bbox`: Permet de changer le style de l'étiquette.

Pour plus [d'info](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html)

```python
fig, axe = plt.subplots(dpi=800)
axe.plot(points, y1)
axe.plot(points, y2)
axe.legend(["tanh", "sin"])
axe.annotate("1.464=tanh(2)+0.5", xy=(2, 1.464), xycoords="data",
             xytext=(0.4, -40), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"))
```
![](note.png)

les différents paramètres pour les coordonnées:
|      Type       |                          Description                           |
| :-------------: | :------------------------------------------------------------: |
|  figure points  |            Points depuis le bas gauche de la figure            |
|  figure pixels  |            Pixels depuis le bas gauche de la figure            |
| figure fraction |         Une fraction depuis le bas gauche de la figure         |
|   axes points   |           Points depuis le coin bas gauche de l'axes           |
|   axes pixels   |           Pixels depuis le coin bas gauche de l'axes           |
|  axes fraction  |        Une fraction depuis le coin bas gauche de l'axes        |
|      data       |      Utilise le système de coordonnées de l'objet annoté       |
|      polar      | $(\theta, r)$, sinon utilise les coordonnées natives de `data` | 

#### Text
Le texte est une version simplifiée de `annotate`. Très utile quand notre texte n'est relié à aucune donnée spécifique.
```python
axe.text(x, y, text, fontdict, bbox)
```
`fontdict` est un dictionnaire pour une police d'écriture de type:
```python
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
```
Le texte supporte également [$\LaTeX$](https://www.latex-project.org/).

#### Grid
Cela correspond au quadrillé sur un graphe. quelques paramètres:

- `color`: Couleur des quadrillés
- `linestyle`: Style des quadrillés (eg: `-`, `dashed`)
- `linewidth`: Épaisseur des quadrillés
- `which`: Pour choisir quel `tick` on relie

Le `which` permet de choisir quel `tick` on relie. Soit ceux avec des chiffres qu'on appelle des `major` ou ceux intermédiaires sans chiffres `minor`.

```python
axe.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
axe.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
```
![](grid.png)

#### Title