import Neural
import numpy as np
import time

input_nodes = 784; hidden_nodes = 100; output_nodes = 10 
learning_rate = 0.12

n = Neural.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.loadWeight()

for i in range(10):
    draw = np.zeros(10) + 0.01
    draw[i] = 0.99

    n.backquery(draw)
