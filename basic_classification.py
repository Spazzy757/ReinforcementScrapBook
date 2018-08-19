# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
# Split it into Training and Test Sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data
# Training Set
train_images.shape
len(train_labels)
train_labels

# Test Set
test_images.shape
len(test_labels)
test_labels

# Pre process the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

# We scale these values to a range of 0 to 1 before feeding to the neural
# network model. For this, cast the datatype of the image components from an
# integer to a float, and divide by 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# import matplotlib.pyplot as plt
# %matplotlib osx

# Display the first 25 images from the training set and display the class name
# below each image.
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid('off')
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])


# Build The Model
# Setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# TRain the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Make Predictions
predictions = model.predict(test_images)
# List of confidence levels
predictions[0]
# can see which label has the highest confidence value
np.argmax(predictions[0])
test_labels[0]


# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                  class_names[true_label]),
                                  color=color)

# Finally, use the trained model to make a prediction about a single image.
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

print(img.shape)

# Prediction
predictions = model.predict(img)

print(predictions)
prediction = predictions[0]

np.argmax(prediction)

