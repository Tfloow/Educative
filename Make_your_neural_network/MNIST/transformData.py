# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset

# helper to load data from PNG image files
#import imageio
import scipy.misc
# glob helps select multiple files using patterns
import glob

import numpy
# library for plotting arrays
import matplotlib.pyplot
import imageio

# our own image test data set
our_own_dataset = []

for image_file_name in glob.glob('MNIST/handwritten?(Personnalisé).png'):
    print ("loading ... ", image_file_name)
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    # load image data from png files into an array
    img_array = imageio.imread(image_file_name)
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    # append label and image data  to test data set
    record = numpy.append(label,img_data)
    print(record)
    our_own_dataset.append(record)
    pass


print(our_own_dataset[0])