'''
Project 2 for CSCI 602
Group Members: Aditi Misra, Rickey Prewitt, Mousa Toure, Sai Tirumalasetty, Mikala Simons, Caleb Winfield, and Christopher Payne
Date: 30 October 2024

Project Description:
modify the provided image auto encoder to use a middle layer with shape 20x20, and input size of 32x32, the output should still be 32x32. use dataset cifar10
Import TensorFlow and other libraries
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.models import Model

# Load the dataset
(x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)

"""Basic autoencoder
Define an autoencoder with two Dense layers: an encoder, which compresses the images into a 400 dimensional latent vector,
 and a decoder, that reconstructs the original image from the latent space.
To define your model, use the Keras Model Subclassing API. (https://www.tensorflow.org/guide/keras/custom_layers_and_models)"""

class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


shape = x_test.shape[1:]
latent_dim = 400
print(f"shape: {shape}")
print(f"latent_dim: {latent_dim}")
autoencoder = Autoencoder(latent_dim, shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

# Train the model using x_train as both the input and the target. The encoder will learn to compress the dataset from 1024 (32x32) dimensions to the latent space,
#  and the decoder will learn to reconstruct the original images.
history = autoencoder.fit(x_train, x_train,
                epochs=3,
                shuffle=True,
                validation_data=(x_test, x_test))


# Plot training & validation loss values
epochs = range(1,4)

ax1 = plt.subplot(2,1,1,label="loss")
plt.plot(epochs,history.history['loss'])
plt.plot(epochs,history.history['val_loss'])

ax2 = plt.subplot(2,1,2,label="accuracy", sharex=ax1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model loss & accuracy vs. epoch')
plt.ylabel('Loss & Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', "Accuracy"], loc='upper right')
plt.show()


# Now that the model is trained, let's test it by encoding and decoding images from the test set.
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
print(f"encoded_imgs: {encoded_imgs.shape}")
print(f"decoded_imgs: {decoded_imgs.shape}")

# plot the first ten images of the original and reconstruction.

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()