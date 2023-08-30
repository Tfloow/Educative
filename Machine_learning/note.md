# Machine Learning with NumPy, pandas, scikit-learn, and More
- [Machine Learning with NumPy, pandas, scikit-learn, and More](#machine-learning-with-numpy-pandas-scikit-learn-and-more)
  - [Introduction](#introduction)
    - [What is Machine Learning](#what-is-machine-learning)
    - [ML vs. AI vs. data science](#ml-vs-ai-vs-data-science)
    - [7 steps of the machine learning process](#7-steps-of-the-machine-learning-process)
  - [Data Manipulation with NumPy](#data-manipulation-with-numpy)
    - [Random](#random)
  - [Data Analysis with pandas](#data-analysis-with-pandas)
  - [Data Preprocessing with scikit-learn](#data-preprocessing-with-scikit-learn)
  - [Data Modeling with scikit-learn](#data-modeling-with-scikit-learn)
  - [Clustering with scikit-learn](#clustering-with-scikit-learn)
  - [Gradient Boosting with XGBoost](#gradient-boosting-with-xgboost)
  - [Deep Learning with TensorFlow](#deep-learning-with-tensorflow)
  - [Deep Learning with Keras](#deep-learning-with-keras)

## Introduction

### What is Machine Learning
We have different type of Machine Learning or ML:
- **Supervised learning**: we label data to train a model. Example a set of handwritten number that already have the right answer tied with the data. (similar to this [summary](https://github.com/Tfloow/Educative/blob/main/Make_your_neural_network))
- **Unsupervised learning**: we let the model make relationships between data. Most data out here are unlabeled. So the goal is not give a precise answer but rather cluster each data together.

### ML vs. AI vs. data science
They do not mean the same. ML is part of AI and AI overlaps with data science. There is also other way to make AI (like alpha-beta pruning, rule-based systems, ...)

### 7 steps of the machine learning process
1. **Data Collection**: one of the most important step to gather good quality data.
2. **Data Processing and Preparation**: make sure it's in the right shape to be fed into our model. (handling missing data, dealing with outliers, errors, ...)
3. **Feature Engineering**: choose to remove some features of the data to optimize the runtime.
4. **Model Selection**: usually we don't start with all new models but rather reuse and tweak some.
5. **Model Training and Data Pipeline**: then we need to choose a data pipeline. Making sure we always have a batch of data ready to be fed.
6. **Model Validation**: we need to use never seen before data by the model so we can test and validate it.
7. **Model Persistence**: then we need to save the crucial part! The

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


## Data Analysis with pandas

## Data Preprocessing with scikit-learn

## Data Modeling with scikit-learn

## Clustering with scikit-learn

## Gradient Boosting with XGBoost

## Deep Learning with TensorFlow

## Deep Learning with Keras