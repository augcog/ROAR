# Initial Setup for Keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def resize(img):
    """
    Resizes the images in the supplied tensor to the original dimensions of the NVIDIA model (66x200)
    """
    tf.image.resize(img, (66, 200))


def simple_model(input_shape=(600, 800, 3)):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))

    model.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
    return model


def nvidia_model(input_shape=(600, 800, 3)):
    model = Sequential()
    # Cropping image
    # model.add(Lambda(lambda imgs: imgs[:, 80:, :, :], input_shape=input_shape))
    # Normalise the image - center the mean at 0
    # model.add(Lambda(lambda imgs: (imgs / 255.0) - 0.5))
    # model.add(Lambda(resize))

    # We have a series of 3 5x5 convolutional layers with a stride of 2x2
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(36,(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
    model.add(Conv2D(64, (3,3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    # Flattening the output of last convolutional layer before entering fully connected phase
    model.add(Flatten())

    # Fully connected layers
    # model.add(Dense(1164, activation='relu'))
    # model.add(BatchNormalization())

    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1))

    model.compile(loss="MSE", optimizer=Adam(lr=0.001))
    return model

