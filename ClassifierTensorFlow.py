"""

This script implements a binary image classification model using the TensorFlow and Keras frameworks.

Implements a robust CNN architecture for binary image classification.
Automates data loading and preprocessing using Keras utilities.
Saves the trained model for reuse.
Provides a clear, step-by-step process for predicting the class of test images.

Functionality:

1. Importing Libraries
The script uses several essential libraries:

TensorFlow/Keras: For building and training the neural network.
Pillow (PIL): To preprocess image data.
NumPy: For numerical operations.
os: To handle file operations for training and testing.

2. Image Data Preparation
Training Images:
The script resizes all training images to 150x150 pixels (img_width, img_height).
Images are rescaled by dividing pixel values by 255 to normalize them between [0, 1] using ImageDataGenerator.
The flow_from_directory method loads images and labels them automatically based on the folder structure.

train_data = train_rescale.flow_from_directory(
    'train/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

3. Model Architecture
The model follows a Convolutional Neural Network (CNN) architecture with the following layers:

Convolutional Layers (Conv2D): Extract spatial features from images.
Activation Layers: Use the ReLU activation function to introduce non-linearity.
MaxPooling Layers: Reduce the spatial dimensions while retaining important features.
Flatten Layer: Flattens the feature maps into a 1D array.
Dense Layers: Fully connected layers for classification, with dropout added to prevent overfitting.
Output Layer: A single neuron with a sigmoid activation function for binary classification.

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
...
model.add(Dense(1))
model.add(Activation('sigmoid'))

4. Model Compilation
The model is compiled with the following configurations:

Loss Function: binary_crossentropy, suitable for binary classification.
Optimizer: rmsprop, a gradient-based optimizer.
Metrics: Tracks accuracy during training.

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

5. Model Training
The model is trained for 128 epochs using the training data (train_data).
Each epoch processes all training batches defined by steps_per_epoch.

model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=128
)

6. Saving the Model
The script saves the trained model in two formats:

Weights-only file: Saved in HDF5 format (model_weights.weights.h5).
Full Keras model: Saved in the native .keras format for reuse (model_keras.keras).

model.save_weights('model_weights.weights.h5')
model.save('model_keras.keras')

7. Test Image Prediction
The script loads images from a test/ directory, preprocesses each image (resizing, normalizing, and expanding dimensions), and uses the trained model to make predictions.

Preprocessing: Ensures that test images match the dimensions and scale used in training.
Prediction: If the model predicts a probability (result[0][0]) greater than or equal to 0.5, the image is classified as "pass"; otherwise, it's "fail."

for image in test_images:
    img = Image.open('test/' + image).convert('RGB')
    ...
    result = model.predict(img)
    prediction = 'pass' if result[0][0] >= 0.5 else 'fail'
    print(f"The image {image} is a: {prediction}")

8. Output
The script prints the prediction result for each test image in the format:
The image <image_name> is a: <pass/fail>

"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image

img_width, img_height = 150, 150

train_rescale = ImageDataGenerator(rescale=1.0 / 255)
validation_rescale = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_rescale.flow_from_directory(
    'train/',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

validation_data = validation_rescale.flow_from_directory(
    'validation/',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))  # 3 channels (RGB)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=128,
    validation_data=validation_data,
    validation_steps=len(validation_data)
)

model.save_weights('model_weights.weights.h5')
model.save('model_keras.keras')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

test_images = os.listdir('test/')

for image in test_images:

    img = Image.open('test/' + image).convert('RGB')
    img = img.resize((img_width, img_height))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)

    prediction = 'pass' if result[0][0] >= 0.5 else 'fail'

    print(f"The image {image} is a: {prediction}")
