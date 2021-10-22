# # Initial Setup for Keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Lambda
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D
#
# def resize(img):
#     """
#     Resizes the images in the supplied tensor to the original dimensions of the NVIDIA model (66x200)
#     """
#     tf.image.resize(img, (66, 200))
#
#
# def simple_model(input_shape=(600, 800, 3)):
#     model = Sequential()
#     model.add(Flatten(input_shape=input_shape))
#     model.add(Dense(1))
#
#     model.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
#     return model
#
#
# def michael_model_2(input_shape=(600, 800, 3)):
#     base_model = ResNet50(weights='imagenet', include_top=False)
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
#     predictions = Dense(200, activation="softmax")(x)
#     model = Model(inputs=base_model.input, output=predictions)
#     for layer in base_model.layers:
#         layer.trainable = False
#     model.add(Flatten())
#     model.add(BatchNormalization())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#     model.add(Dense(2, activation='relu'))
#     return model
#
#
# def michael_model(input_shape=(600, 800, 3)):
#     model = Sequential()
#
#     resnet_model: Model = ResNet50(include_top=False, input_shape=input_shape)
#     for layer in model.layers:
#         layer.trainable = False
#
#     model.add(resnet_model)
#     model.add(Flatten())
#     model.add(BatchNormalization())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#     model.add(Dense(2, activation='relu'))
#     return model
#
#
# def nvidia_model(input_shape=(600, 800, 3)):
#     model = Sequential()
#
#     # We have a series of 3 5x5 convolutional layers with a stride of 2x2
#     model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'), input_shape=input_shape)
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
#     model.add(BatchNormalization())
#
#     # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
#     model.add(BatchNormalization())
#
#     # Flattening the output_oct_10 of last convolutional layer before entering fully connected phase
#     model.add(Flatten())
#
#     # Fully connected layers
#     model.add(Dense(512, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(200, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(50, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(10, activation='relu'))
#     model.add(BatchNormalization())
#
#     # Output layer
#     model.add(Dense(1))
#     # huber loss
#     model.compile(loss="MSE", optimizer=Adam(lr=0.001))
#     return model
#
#
# def nvidia_model_throttle_steering(input_shape=(600, 800, 3)):
#     model = Sequential()
#
#     # We have a series of 3 5x5 convolutional layers with a stride of 2x2
#     model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
#     model.add(BatchNormalization())
#
#     # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
#     model.add(BatchNormalization())
#
#     # Flattening the output_oct_10 of last convolutional layer before entering fully connected phase
#     model.add(Flatten())
#
#     # Fully connected layers
#     model.add(Dense(512, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(200, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(50, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(10, activation='relu'))
#     model.add(BatchNormalization())
#
#     # Output layer
#     model.add(Dense(2))
#
#     model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=0.001))
#     return model

import torch.nn as nn
import torch

class CarModel(nn.Module):
    def __init__(self, batch_size=1, image_width=800, image_height=600):
        super(CarModel, self).__init__()
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),


            nn.Conv2d(24, 48, 5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),


            nn.Conv2d(48, 96, 5),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(in_features=256, out_features=256 // 2),
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(in_features=256 // 2, out_features=256 // 4),
            nn.ELU(),
            nn.Linear(in_features=256 // 4, out_features=self.batch_size)
        )

    def forward(self, input):
        input = torch.reshape(input, (self.batch_size, 1, self.image_height, self.image_width))
        output = self.conv_layers(input)
        output = output.flatten()
        output = self.linear_layers(output)
        return output