import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# split the train and validataion dataset
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes=10)
print(x_train.shape, x_valid.shape, x_test.shape)

# show a smaple of train
plt.figure(figsize=(6, 6))
# plt.imshow(x_train[1])
# plt.title(y_train[1].argmax())
# plt.show()

plt.plot([1, 3, 5], [2, 4, 6])
plt.show()
