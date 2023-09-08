# Machine Learning with NumPy, pandas, scikit-learn, and More
- [Machine Learning with NumPy, pandas, scikit-learn, and More](#machine-learning-with-numpy-pandas-scikit-learn-and-more)
  - [Introduction](#introduction)
    - [What is Machine Learning](#what-is-machine-learning)
    - [ML vs. AI vs. data science](#ml-vs-ai-vs-data-science)
    - [7 steps of the machine learning process](#7-steps-of-the-machine-learning-process)

Summary:

- Machine Learning with NumPy, pandas, scikit-learn, and More
  -  [Data Manipulation with NumPy](Chapter/Data_Manipulation.md)
  -  [Data Analysis with pandas](Chapter/Data_Analysis.md)
  -  [Data Preprocessing with scikit-learn](Chapter/Data_Preprocessing.md)
  -  [Data Modeling with scikit-learn](Chapter/Data_Modeling.md)
  -  [Clustering with scikit-learn](Chapter/Clustering_with_scikit-learn.md)
  -  [Gradient Boosting with XGBoost](Chapter/Gradient_Boosting_with_XGBoost.md)
  -  [Deep Learning with TensorFlow](Chapter/Deep_Learning_with_TensorFlow.md)
  -  [Deep Learning with Keras](Chapter/Deep_Learning_with_Keras.md)

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

[→](Chapter/Data_Manipulation.md)