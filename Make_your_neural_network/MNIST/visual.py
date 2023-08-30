import matplotlib.pyplot as plt
import numpy as np

train_data = "C:\\Users\\thoma\\OneDrive - UCL\\Code\\Educative\\MNIST\\mnist_train.csv"
test_data = "C:\\Users\\thoma\\OneDrive - UCL\\Code\\Educative\\MNIST\\mnist_test.csv"

with open(test_data, "r") as file:
    count = 0
    for i in file.readlines():
        first = i.strip("\n").split(",")
        first = [int(x) for x in first]
        correct = first[0]; image = np.array(first[1:])
        
        mat = image.reshape((int(np.power(len(image), 0.5)), int(np.power(len(image), 0.5))))

        plt.imshow(mat, interpolation='nearest', cmap=plt.cm.binary)
        plt.savefig(f"C:\\Users\\thoma\\OneDrive - UCL\\Code\\Educative\\MNIST\\writing{correct}.png")
        if count > 20:
            break
        count+=1