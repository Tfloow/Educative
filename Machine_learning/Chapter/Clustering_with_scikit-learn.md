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

We can also find similarities by looking to a given data and looking for their nearest neighbors. We use the `NearestNeighbors` object (part of the `neighbors` module). Here, we find the 5 nearest neighbors for a new data observation (`new_obs`) based on its fitted dataset (`data`):

```python
data = np.array([
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3. , 1.4, 0.2],
  [4.7, 3.2, 1.3, 0.2],
  [4.6, 3.1, 1.5, 0.2],
  [5. , 3.6, 1.4, 0.2],
  [5.4, 3.9, 1.7, 0.4],
  [4.6, 3.4, 1.4, 0.3],
  [5. , 3.4, 1.5, 0.2],
  [4.4, 2.9, 1.4, 0.2],
  [4.9, 3.1, 1.5, 0.1]])

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors()
nbrs.fit(data)
new_obs = np.array([[5. , 3.5, 1.6, 0.3]])
dists, knbrs = nbrs.kneighbors(new_obs)

# nearest neighbors indexes
print('{}\n'.format(repr(knbrs)))
# nearest neighbor distances
print('{}\n'.format(repr(dists)))

only_nbrs = nbrs.kneighbors(new_obs,
                            return_distance=False)
print('{}\n'.format(repr(only_nbrs)))
```

<details>
<summary>Output</summary>
<br>

```
array([[7, 4, 0, 6, 9]])

array([[0.17320508, 0.24494897, 0.24494897, 0.45825757, 0.46904158]])

array([[7, 4, 0, 6, 9]])
```
</details>

The `NearestNeighbors` is fitted with a dataset. We use the `kneighbors` function which takes new data and return the k nearest neighbors and their distances (to no have the distance we can set `return_distance` to `False`).

```python
data = np.array([
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3. , 1.4, 0.2],
  [4.7, 3.2, 1.3, 0.2],
  [4.6, 3.1, 1.5, 0.2],
  [5. , 3.6, 1.4, 0.2],
  [5.4, 3.9, 1.7, 0.4],
  [4.6, 3.4, 1.4, 0.3],
  [5. , 3.4, 1.5, 0.2],
  [4.4, 2.9, 1.4, 0.2],
  [4.9, 3.1, 1.5, 0.1]])

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2)
nbrs.fit(data)
new_obs = np.array([
  [5. , 3.5, 1.6, 0.3],
  [4.8, 3.2, 1.5, 0.1]])
dists, knbrs = nbrs.kneighbors(new_obs)

# nearest neighbors indexes
print('{}\n'.format(repr(knbrs)))
# nearest neighbor distances
print('{}\n'.format(repr(dists)))
```

<details>
<summary>Output</summary>
<br>

```
array([[7, 0],
       [9, 2]])

array([[0.17320508, 0.24494897],
       [0.14142136, 0.24494897]])
```
</details>

### K-Means Clustering

This is one of the most common clustering method where we want to have $K$ clusters. We center each clusters around a mean or *centroid* in cluster term.

```python
cluster = np.array([
  [ 1.2, 0.6],
  [ 2.4, 0.8],
  [-1.6, 1.4],
  [ 0. , 1.2]])
print('Cluster:\n{}\n'.format(repr(cluster)))

centroid = cluster.mean(axis=0)
print('Centroid:\n{}\n'.format(repr(centroid)))
```

<details>
<summary>Output</summary>
<br>

```
Cluster:
array([[ 1.2,  0.6],
       [ 2.4,  0.8],
       [-1.6,  1.4],
       [ 0. ,  1.2]])

Centroid:
array([0.5, 1. ])
```
</details>

The algorithm is iterative and keep updating the centroids of each $K$ clusters. It will stop when there is no new update of centroid for any observed data. We implement this using the `KMeans` object (from the `cluster` module):

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
# predefined data
kmeans.fit(data)

# cluster assignments
print('{}\n'.format(repr(kmeans.labels_)))

# centroids
print('{}\n'.format(repr(kmeans.cluster_centers_)))

new_obs = np.array([
  [5.1, 3.2, 1.7, 1.9],
  [6.9, 3.2, 5.3, 2.2]])
# predict clusters
print('{}\n'.format(repr(kmeans.predict(new_obs))))
```

<details>
<summary>Output</summary>
<br>

```
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
       0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0,
       0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2], dtype=int32)

array([[6.85      , 3.07368421, 5.74210526, 2.07105263],
       [5.006     , 3.428     , 1.462     , 0.246     ],
       [5.9016129 , 2.7483871 , 4.39354839, 1.43387097]])

array([1, 0], dtype=int32)
```
</details>

The `KMeans` object uses K-means++ centroids for initialization. To get the final cluster assignments for each data observation we use the `labels_` and for the final centroids `cluster_centers_`. To predict new data into one of the cluster we use the function `predict`.

#### Mini-batch clustering

The process of clustering can be really slow. So we sometimes want to use smaller data and apply this clustering. We call this *mini-batch K-means clustering*. It produces poorer results but the tradeoff is tiny. In scikit we use the `MiniBatchKMeans` object (from `cluster` module). The new arguments to add is `batch_size` to specify the size of those new mini-batch.

```python
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=10)
# predefined data
kmeans.fit(data)

# cluster assignments
print('{}\n'.format(repr(kmeans.labels_)))

# centroids
print('{}\n'.format(repr(kmeans.cluster_centers_)))

new_obs = np.array([
  [5.1, 3.2, 1.7, 1.9],
  [6.9, 3.2, 5.3, 2.2]])
# predict clusters
print('{}\n'.format(repr(kmeans.predict(new_obs))))
```

<details>
<summary>Output</summary>
<br>

```
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
       0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0,
       0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2], dtype=int32)

array([[6.86933333, 3.15333333, 5.70666667, 2.096     ],
       [4.96276596, 3.38617021, 1.43617021, 0.24893617],
       [5.88347107, 2.7677686 , 4.43305785, 1.44958678]])

array([1, 0], dtype=int32)
```
</details>

### Hierarchical Clustering

The K-means clustering makes the assumption that everything is a circle and so can inaccurately predicts data. To avoid this issue, we can use **hierarchical clustering**. It allows us to cluster any type of data. We have 2 approaches:

1. Bottom up (divisive): it first treats all the data as a single cluster then splits it to the desired number of cluster.
2. Top-down (agglomerative): it treats each data as its own cluster then merges the two most similar together until we reach the desired amount of cluster.

The most used and often the best is the *agglomerative* one. In scikit, we use the `AgglomerativeClustering` object (part of `cluster` module). We specify the amount of cluster through `n_clusters`:

```python
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3)
# predefined data
agg.fit(data)

# cluster assignments
print('{}\n'.format(repr(agg.labels_)))
```

<details>
<summary>Output</summary>
<br>

```
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2,
       2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2,
       2, 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0])
```
</details>

There is no `predict` or `cluster_centers_` for those trees.


### Mean Shift Clustering

There is also an algorithm that can tell how many clusters we need. The **mean shift** is one of those clustering algorithm. It looks for cluster like centroid into the data. We use this in scikit thanks to the `MeanShift` object (part of `cluster`).

```python
from sklearn.cluster import MeanShift
mean_shift = MeanShift()
# predefined data
mean_shift.fit(data)

# cluster assignments
print('{}\n'.format(repr(mean_shift.labels_)))

# centroids
print('{}\n'.format(repr(mean_shift.cluster_centers_)))

new_obs = np.array([
  [5.1, 3.2, 1.7, 1.9],
  [6.9, 3.2, 5.3, 2.2]])
# predict clusters
print('{}\n'.format(repr(mean_shift.predict(new_obs))))
```

<details>
<summary>Output</summary>
<br>

```
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

array([[6.21142857, 2.89285714, 4.85285714, 1.67285714],
       [5.01632653, 3.45102041, 1.46530612, 0.24489796]])

array([1, 0])
```
</details>


### DBSCAN

#### Clustering by density
The issue with the mean shift clustering is the assumption of centroid in the data and how bad it scales. **DBSCAN** finds high-density regions in the data.

The high-density is like a cluster of data while the lower density is like a transition space between cluster. We define the *core samples* which is data with many neighbors.

#### Neighbors and core samples
We choose what we consider *neighbors* for the data by telling the maximum distance between two data $\varepsilon$. To decide what is high-density region, we tell how many neighbors it should have.

To do this we use `DBSCAN` function and we initialize the distance with `epc` and the amount of neighbors with `min_samples`.

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=30)
# predefined data
dbscan.fit(data)

# cluster assignments
print('{}\n'.format(repr(dbscan.labels_)))

# core samples
print('{}\n'.format(repr(dbscan.core_sample_indices_)))
num_core_samples = len(dbscan.core_sample_indices_)
print('Num core samples: {}\n'.format(num_core_samples))
```

<details>
<summary>Output</summary>
<br>

```
array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1])

array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
        27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
        40,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
        54,  55,  56,  58,  59,  61,  62,  63,  64,  65,  66,  67,  68,
        69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
        82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  94,  95,
        96,  97,  99, 101, 102, 103, 104, 108, 110, 111, 112, 113, 114,
       115, 116, 119, 120, 121, 123, 124, 125, 126, 127, 128, 129, 132,
       133, 134, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
       147, 148, 149])

Num core samples: 133
```
</details>

### Evaluating Clusters

We can simply look at our cluster if they make sense. But the best is to have access to the real labels of each cluster. Then we can evaluate and adjust.

With the **adjusted Rand index**. It measures the similarity between the true clustering (so the real labels) and predicted one. This algorithm is *corrected-for-chance* so it will never give a good score. ARI ranges $[-1; 1]$:
- Negatives: bad labeling.
- 0: random labeling.
- Positive: good labeling.
In scikit we use the `adjusted_rand_score` and it needs as arguments the real labels and the predicted cluster.

```python
from sklearn.metrics import adjusted_rand_score
true_labels = np.array([0, 0, 0, 1, 1, 1])
pred_labels = np.array([0, 0, 1, 1, 2, 2])

ari = adjusted_rand_score(true_labels, pred_labels)
print('{}\n'.format(ari))

# symmetric
ari = adjusted_rand_score(pred_labels, true_labels)
print('{}\n'.format(ari))

# Perfect labeling
perf_labels = np.array([0, 0, 0, 1, 1, 1])
ari = adjusted_rand_score(true_labels, perf_labels)
print('{}\n'.format(ari))

# Perfect labeling, permuted
permuted_labels = np.array([1, 1, 1, 0, 0, 0])
ari = adjusted_rand_score(true_labels, permuted_labels)
print('{}\n'.format(ari))

renamed_labels = np.array([1, 1, 1, 3, 3, 3])
# Renamed labels to 1, 3
ari = adjusted_rand_score(true_labels, renamed_labels)
print('{}\n'.format(ari))

true_labels2 = np.array([0, 1, 2, 0, 3, 4, 5, 1])
# Bad labeling
pred_labels2 = np.array([1, 1, 0, 0, 2, 2, 2, 2])
ari = adjusted_rand_score(true_labels2, pred_labels2)
print('{}\n'.format(ari))
```
<details>
<summary>Output</summary>
<br>

```
0.24242424242424243

0.24242424242424243

1.0

1.0

1.0

-0.12903225806451613

```
</details>

Another clustering evaluation metric is the **adjusted mutual information**. We use the `adjusted_mutual_info_score` which works like the ARI.

```python
from sklearn.metrics import adjusted_mutual_info_score
true_labels = np.array([0, 0, 0, 1, 1, 1])
pred_labels = np.array([0, 0, 1, 1, 2, 2])

ami = adjusted_mutual_info_score(true_labels, pred_labels)
print('{}\n'.format(ami))

# symmetric
ami = adjusted_mutual_info_score(pred_labels, true_labels)
print('{}\n'.format(ami))

# Perfect labeling
perf_labels = np.array([0, 0, 0, 1, 1, 1])
ami = adjusted_mutual_info_score(true_labels, perf_labels)
print('{}\n'.format(ami))

# Perfect labeling, permuted
permuted_labels = np.array([1, 1, 1, 0, 0, 0])
ami = adjusted_mutual_info_score(true_labels, permuted_labels)
print('{}\n'.format(ami))

renamed_labels = np.array([1, 1, 1, 3, 3, 3])
# Renamed labels to 1, 3
ami = adjusted_mutual_info_score(true_labels, renamed_labels)
print('{}\n'.format(ami))

true_labels2 = np.array([0, 1, 2, 0, 3, 4, 5, 1])
# Bad labeling
pred_labels2 = np.array([1, 1, 0, 0, 2, 2, 2, 2])
ami = adjusted_mutual_info_score(true_labels2, pred_labels2)
print('{}\n'.format(ami))
```

<details>
<summary>Output</summary>
<br>

```
0.2987924581708901

0.2987924581708903

1.0

1.0

1.0

-0.16666666666666655
```
</details>

We usually use **ARI** when the true clusters are large and as big as the other set and the **AMI** in the other case (especially if clusters are unbalanced).

### Feature Clustering

We can reduce the amount of features to speed up processes while not losing important original information. This is what we call **Agglomerative feature clustering**.

In scikit we use `FeatureAgglomeration`. We specify the new amount of cluster with the keyword `n_clusters`.

```python
# predefined data
print('Original shape: {}\n'.format(data.shape))
print('First 10:\n{}\n'.format(repr(data[:10])))

from sklearn.cluster import FeatureAgglomeration
agg = FeatureAgglomeration(n_clusters=2)
new_data = agg.fit_transform(data)
print('New shape: {}\n'.format(new_data.shape))
print('First 10:\n{}\n'.format(repr(new_data[:10])))
```

<details>
<summary>Output</summary>
<br>

```
Original shape: (150, 4)

First 10:
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2],
       [5.4, 3.9, 1.7, 0.4],
       [4.6, 3.4, 1.4, 0.3],
       [5. , 3.4, 1.5, 0.2],
       [4.4, 2.9, 1.4, 0.2],
       [4.9, 3.1, 1.5, 0.1]])

New shape: (150, 2)

First 10:
array([[1.7       , 5.1       ],
       [1.53333333, 4.9       ],
       [1.56666667, 4.7       ],
       [1.6       , 4.6       ],
       [1.73333333, 5.        ],
       [2.        , 5.4       ],
       [1.7       , 4.6       ],
       [1.7       , 5.        ],
       [1.5       , 4.4       ],
       [1.56666667, 4.9       ]])
```
</details>

[→](Gradient_Boosting_with_XGBoost.md)