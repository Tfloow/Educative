import numpy as np
from Neural import neuralNetwork

skipTraining = True
epoch = 5

input_nodes = 784; hidden_nodes = 100; output_nodes = 10 
learning_rate = 0.12

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

train_data = "C:\\Users\\thoma\\OneDrive - UCL\\Code\\Educative\\MNIST\\mnist_train.csv"
test_data = "C:\\Users\\thoma\\OneDrive - UCL\\Code\\Educative\\MNIST\\mnist_test.csv"

if not skipTraining:
    with open(train_data, "r") as file:
        data_list = file.readlines()
    
with open(test_data, "r") as file:
    test_list = file.readlines() 

if not skipTraining:
    for e in range(epoch):
        for record in data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
else:
    n.loadWeight()
    
good = 0; bad = 0
print("Starting test")
for test in test_list:
    all_values = test.split(",")
    scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    #Querying the trained network
    res = n.query(scaled_input)
    guess = np.argmax(res)
    guess = int(guess); all_values[0] = int(all_values[0])
        
    if(guess == all_values[0]):
        good+=1
    else:
        print(guess, all_values[0])
        bad+=1

print(good/(good + bad))
if not skipTraining:
    n.writeWeight()