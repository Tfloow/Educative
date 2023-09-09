[↰](../note.md)

## Deep Learning with Keras

- [Deep Learning with Keras](#deep-learning-with-keras)
  - [Introduction](#introduction)
  - [Sequential Model](#sequential-model)
  - [Model Output](#model-output)
  - [Model Configuration](#model-configuration)
  - [Model Execution](#model-execution)



### Introduction

It is a simple and compact API for neural network. It is less complex than TensorFlow. It is often run on top of TensorFlow.
- Keras: good for small project.
- TensorFlow: industry standard.

### Sequential Model

Every neural network in *Keras* is an instance of the `Sequential` object. We can then stack multiple layers inside this object.

Of course, the most used is the `Dense` layer. We can either add layer by layer with `add` or instantiate it with all the layers.

```python
model = Sequential()
layer1 = Dense(5, input_dim=4)
model.add(layer1)
layer2 = Dense(3, activation='relu')
model.add(layer2)
```

Or immediately starting with all the layers in a list.

```python
layer1 = Dense(5, input_dim=4)
layer2 = Dense(3, activation='relu')
model = Sequential([layer1, layer2])
```

The only required arguments is the number of perceptron in the layer. We can specify the way it activates with `activation` (no activation needed for the first layer since it's just input).

For the first layer with input we must specify the amount of input with `input_dim`.

### Model Output

In Keras, the cross-entropy loss functions only calculate cross-entropy without applying sigmoid/softmax.

```python
model = Sequential()
layer1 = Dense(5, activation='relu', input_dim=4)
model.add(layer1)
layer2 = Dense(1, activation='sigmoid')
model.add(layer2)
```

```python
model = Sequential()
layer1 = Dense(5, input_dim=4)
model.add(layer1)
layer2 = Dense(3, activation='softmax')
model.add(layer2)
```

### Model Configuration

it is really simple to setup the model for training. We just need to tell which optimize to use (eg: ADAM. [list](https://keras.io/api/optimizers/)) in lowercase.

The most important keywords are:
- `loss`: specify the loss function to use.
  - For **binary classification**: `binary_crossentropy`.
  - For **multiclass classification**: `categorical_crossentropy`.
- `metrics`: list of strings. It represents the tracked metric. For classification we just want to track the model loss.

```python
model = Sequential()
layer1 = Dense(5, activation='relu', input_dim=4)
model.add(layer1)
layer2 = Dense(1, activation='sigmoid')
model.add(layer2)
model.compile('adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### Model Execution

#### Training

After doing the setup, we just need to use `fit` to start the training. We can simply use NumPy arrays for the fit.

We can specify the batch size with `batch_size` and the epoch with `epoch`.

```python
model = Sequential()
layer1 = Dense(200, activation='relu', input_dim=4)
model.add(layer1)
layer2 = Dense(200, activation='relu')
model.add(layer2)
layer3 = Dense(3, activation='softmax')
model.add(layer3)
model.compile('adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# predefined multiclass dataset
train_output = model.fit(data, labels,
                         batch_size=20, epochs=5)

print(train_output.history)
```

<details>
<summary>Output</summary>
<br>

```
{'loss': [1.047894505659739, 0.6740142941474915, 0.496600067615509, 0.4373936434586843, 0.40250584880510965], 
'accuracy': [0.5, 0.79333335, 0.93333334, 0.70666665, 0.82666665]}
```
</details>

We see the results for each epoch.

#### Evaluation

To evaluate a model we simply use `evaluate` with NumPy arrays.

```python
# predefined eval dataset
print(model.evaluate(eval_data, eval_labels))
```

<details>
<summary>Output</summary>
<br>

```
[0.31819412112236023, 0.8700000047683716]
```

First element in the list is the loss and then the accuracy.
</details>

#### Predictions

To predict we use the function `predict`. It will return the predictions for each labels for each possible category.

```python
# 3 new data observations
print('{}'.format(repr(model.predict(new_data))))
```

<details>
<summary>Output</summary>
<br>

```
array([[0.97158176, 0.02550585, 0.00291234],
       [0.00181993, 0.22216259, 0.7760175 ],
       [0.04493526, 0.73288304, 0.22218166]], dtype=float32)
```

So it means for observation number:
1. the class is 0 (highest probability)
2. the class is 2
3. the class is 1

</details>
