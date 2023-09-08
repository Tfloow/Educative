[↰](../note.md)

## Clustering with scikit-learn

- [Clustering with scikit-learn](#clustering-with-scikit-learn)
  - [Introduction](#introduction)
  - [Cosine Similarity](#cosine-similarity)
  - [Nearest Neighbors](#nearest-neighbors)
  - [K-Means Clustering](#k-means-clustering)
  - [Hierarchical Clustering](#hierarchical-clustering)
  - [Mean Shift Clustering](#mean-shift-clustering)
  - [DBSCAN](#dbscan)
  - [Evaluating Clusters](#evaluating-clusters)
  - [Feature Clustering](#feature-clustering)


### Introduction

Now we will see how we can use unsupervised learning for our models. So now, we have only data with **no labels**. We let the model find links and relationships between our data. It is what **clustering** is about.

### Cosine Similarity

It's the most common way of measuring similarities in data. This cosine similarity ranges between $[-1; 1]$ and uses the fact that data are just vectors. So $1$ means they have a great similarity, $-1$ shows a sort of divergence and $0$ shows no correlation between the data.

#### Calculating

when we have $2$ vectors $u$ and $v$, we do a *dot product* between the L2-normalization of the vectors:

$$cossim(u,v) = \frac{u}{||u||_2} \cdot \frac{v}{||v||_2}$$

We implement this in scikit-learn with the `cosine_similarity` function (part of `metrics.pairwise` module). Here is an example with pairs of 2-D dataset:

```python
from sklearn.metrics.pairwise import cosine_similarity
data = np.array([
  [ 1.1,  0.3],
  [ 2.1,  0.6],
  [-1.1, -0.4],
  [ 0. , -3.2]])
cos_sims = cosine_similarity(data)
print('{}\n'.format(repr(cos_sims)))
```

<details>
<summary>Output</summary>
<br>

```
array([[ 1.        ,  0.99992743, -0.99659724, -0.26311741],
       [ 0.99992743,  1.        , -0.99751792, -0.27472113],
       [-0.99659724, -0.99751792,  1.        ,  0.34174306],
       [-0.26311741, -0.27472113,  0.34174306,  1.        ]])
```
</details>

If we only pass 1 dataset (like in the example) it will simply to the cosine similarity between each bit of the data.

We can also pass 2 dataset with equal numbers of columns:

```python
from sklearn.metrics.pairwise import cosine_similarity
data = np.array([
  [ 1.1,  0.3],
  [ 2.1,  0.6],
  [-1.1, -0.4],
  [ 0. , -3.2]])
data2 = np.array([
  [ 1.7,  0.4],
  [ 4.2, 1.25],
  [-8.1,  1.2]])
cos_sims = cosine_similarity(data, data2)
print('{}\n'.format(repr(cos_sims)))
```

<details>
<summary>Output</summary>
<br>

```
array([[ 0.9993819 ,  0.99973508, -0.91578821],
       [ 0.99888586,  0.99993982, -0.9108828 ],
       [-0.99308366, -0.9982304 ,  0.87956492],
       [-0.22903933, -0.28525359, -0.14654866]])
```
</details>

### Nearest Neighbors


### K-Means Clustering


### Hierarchical Clustering


### Mean Shift Clustering


### DBSCAN


### Evaluating Clusters


### Feature Clustering




[→](Gradient_Boosting_with_XGBoost.md)