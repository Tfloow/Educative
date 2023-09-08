[↰](../note.md)

## Data Manipulation with NumPy
*This chapter mostly covers previously learned topics so I will just write down some useful info only*.

- Casting: we can cast an entire array using `arr.astype(type)`.
- Filler: we use `np.nan` as a placeholder for missing data.
- Linspace: we can use `np.linspace(start, finish)` and there is some useful argument:
  - `num`: indicates the number of element we want in (set to 50 by default)
  - `endpoint`: if we want the last number (like in `np.arange` if `False`)
  - `dtype`: the encoded type of number
- Flatten: the inverse of reshape, it turns an array back to a 1D array. So something of shape `(x,)`. (notice it's `(x,)` and not `(x,y)`).
- Transpose: it's like the `.T` but we can set how the data will transpose with `np.transpose(arr, axes=(1, 2, 0))`.
- Like: we can create zeros and ones matrix by looking at the shape of other with `np.ones_like(matrix)`.

### Random
- `np.random.randint(x)`:
  - `x`: represent the upper bound starting from 0.
  - `high`: represent the upper bound, so `x` becomes the lower bound.
  - `size`: help us choose the size of the returned array.
- `np.random.seed(x)`: to generate a pseudo-random function with a specific seed.
- `np.random.shuffle(x)`: shuffle an array (only its first dimension).
- `np.random.uniform(x)`: draw samples form probability distributions (takes the same parameters as `np.random.randint()`).
- `np.random.normal(x)`: same as `np.random.uniform(x)` but with a normal gaussian distribution.

We also have custom smapling! Useful to choose (pseudo-)randomly from a set like:
```python
colors = ['red', 'blue', 'green']
print(np.random.choice(colors))
print(repr(np.random.choice(colors, size=2)))
print(repr(np.random.choice(colors, size=(2, 2),
                            p=[0.8, 0.19, 0.01])))
```
`p` helps us choose the distribution of probability among the list of possibilities.

### Slicing
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(repr(arr[:, -1]))
# array([3, 6, 9])
print(repr(arr[:, 1:]))
# array([[2, 3],
#       [5, 6],
#       [8, 9]])
```
We can use `argmin()` to find the minimum in an array. We can specify the axis thanks to the argument `axis=`. Each number represent the dimension. (eg: 0 equals to the colums, 1 to the rows, ...)

### Filtering Data
If we have an array we can compare it like this `print(arr == 1)` and it will print us an array with boolean value. We also can use `~` for the negation. We can find `NaN` by printing `print(np.isnan(arr))` to show the `NaN` location.

We can filter data thanks to `np.where()`. We pass as argument a comparison to do between our array and a specific value:
```python
arr = np.array([1,0,1])
print(np.where(arr == 1))
# [0, 2]

arr = np.array([[0, 0, 3],
                [1, 0, 0],
                [-3, 0, 0]])
x_ind, y_ind = np.where(arr != 0)
print(x_ind)
print(ry_ind)
```
It really shines when we pass *3* arguments. We can have then our comparison, then an array in case it's `True` and another in case it's `False`:
```python
np_filter = np.array([[True, False], [False, True]])
positives = np.array([[1, 2], [3, 4]])
negatives = np.array([[-2, -5], [-1, -8]])
print(np.where(np_filter, positives, negatives))
# [[ 1, -5],
#  [-1,  4]]
```
We can also use a constant arguments for `positives` and `negatives`.

If we want to compare the matrix as a whole we can use `np.any()` to see if anything in the array match the condition or `np.all()` if they all do. We can also choose the axis we want to do the comparison. We can also combine all of this:
```python
arr = np.array([[-2, -1, -3],
                [4, 5, -6],
                [3, 9, 1]])
has_positive = np.any(arr > 0, axis=1)
print(has_positive)
# [False  True  True]
print(arr[np.where(has_positive)])
# array([[ 4,  5, -6],
#       [ 3,  9,  1]])
```

### Statistics
We can get the minimum value and the maximum thanks to `arr.min()` and `arr.max()`. If we specify the argument `axis` we will get a row or a column with the smallest or biggest subarray. 

We have also some useful other methods for statistics such as:
- `np.mean()`: To get the average.
- `np.var()`: To get the variance.
- `np.median()`: To get the median. (Not: if no axis is precised, it will simply **flattened** the array)

### Aggregation

To sum all the values of ana rray, we use `np.sum()` to sum. We can pass `axis` to specify if we need to sum along the row or columns.

If we want to use the cumulative sum we can use `np.cumsum()`:
```python
arr = np.array([[0, 72, 3],
                [1, 3, -60],
                [-3, -2, 4]])
print(np.cumsum(arr))
# [ 0, 72, 75, 76, 79, 19, 16, 14, 18]
```

How to concatenate ? We can do it thanks to `np.concatenate()`. We concatenate along one on the axis so we must specify it thanks to `axis=`.

### Saving Data
We can save an object in NumPy by using `np.save(*.npy, arr)`. We need to save it in `.npy` extension to be reused by NumPy. Using this extension will make the raw data harder to process by a human.

To load the data, we simply do `arr = np.load(*.npy)`
