import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow
import numpy as np
import imageio
from matplotlib import pyplot as plt

# Load model
model = keras.models.load_model('digitmodel')

# Get the image
im = imageio.imread("images/five.png")

# Turn the image into grayscale
im = np.dot(im[...,:3], [0.299, 0.587, 0.114])

# Display the image to the user
plt.imshow(im, cmap = plt.get_cmap('gray'))
plt.show()

# Reshape and normalizethe image so that the model can "read" it
im = im.reshape(1, 28, 28, 1)
im /= 255


# Make the prediction and print it
prediction = model.predict(im)
print("\n\t Your number is %d with %.2f%% confidence!\n" % (prediction.argmax(), prediction[0][prediction.argmax()] * 100))