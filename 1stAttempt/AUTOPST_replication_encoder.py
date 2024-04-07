import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, GroupNormalization

tensor_transform = transforms.ToTensor()

class Encoder():
    def __init__(self):
        self.layers = []
        for i in range(5): # first 5 layers have 512 filters
          conv = Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation='relu')
          group = GroupNormalization(groups=512)
          self.layers.append(conv)
          self.layers.append(group)

        conv6 = Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu')
        group6 = GroupNormalization(groups=128)
        self.layers.append(conv6)
        self.layers.append(group6)

        conv7 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')
        group7 = GroupNormalization(groups=32)
        self.layers.append(conv7)
        self.layers.append(group7)

        conv8 = Conv1D(filters=4, kernel_size=5, strides=1, padding='same', activation='relu')
        group8 = GroupNormalization(groups=4)
        self.layers.append(conv8)
        self.layers.append(group8)

    def forward(self, x):
        for layer in self.layers:
          x = layer(x)
        return x

# input_shape = tf.random.normal((1, 13, 1)) # Assuming 1D input

# # Create the encoder model
# encoder_model = Encoder()
# output = encoder_model.forward(input_shape)
# print(output)
