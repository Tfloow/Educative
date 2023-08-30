import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt

class neuralNetwork:
    """
    My neural Network
    """
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set the number of node
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # Set the learning rate
        self.lr = learningrate
        
        # Weights matrices
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        """
        Pour avoir la belle distribution des poids
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        """
        self.activation_function = lambda x: expit(x)
        self.inverse_activation_function = lambda x: logit(x)

    
    def train(self, inputs_list, targets_list):
        # Convert into a 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
             
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        mat = np.reshape(inputs, (28,28))
        plt.imshow(mat, interpolation='nearest', cmap=plt.cm.binary)
        plt.savefig(f"MNIST/Backtrack{np.argmax(targets_list)}")
        
        return inputs
    
    def writeWeight(self):
        np.savetxt("weightHidden.csv", self.wih, delimiter=",")
        
        np.savetxt("weightOutput.csv", self.who, delimiter=",")
        
    def loadWeight(self):
        self.wih = np.loadtxt("weightHidden.csv",delimiter=",", dtype=float)
        
        self.who = np.loadtxt("weightOutput.csv",delimiter=",", dtype=float)

    
    
if __name__ == "__main__":
    input_nodes = 3; hidden_nodes = 3; output_nodes = 3
    learning_rate = 0.3
    
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    output = n.query([1.0,0.5,-1.5])
    print(output)