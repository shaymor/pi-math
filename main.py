import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow
import numpy as np
import imageio
from matplotlib import pyplot as plt

# Load models
digit_model = keras.models.load_model("models/digit_model")
symbol_model = keras.models.load_model('models/operator_model')

# Function to get the images ready
def read_image(path):
    im = imageio.imread(path)
    # Turn the image into grayscale
    im = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    # Display the image to the user
    plt.imshow(im, cmap = plt.get_cmap('gray'))
    plt.show()
    # Reshape and normalize the image so that the model can "read" it
    im = im.reshape(1, 28, 28, 1)
    im /= 255
    return im

# Labels
symbols = ["+", "-", "/", "*", "="]
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# First digit
digit1 = read_image("test_images/seven.png")
dPrediction1 = digit_model.predict(digit1)
d1 = dPrediction1.argmax()
d1_confidence = dPrediction1[0][dPrediction1.argmax()]
print("\n\t Your first digit is %d with %.2f%% confidence!" % (d1, d1_confidence * 100))

# Symbol
symbol = read_image("test_images/divide.png")
sPrediction = symbol_model.predict(symbol)
s = symbols[sPrediction.argmax()]
s_confidence = sPrediction[0][sPrediction.argmax()]
print("\n\t Your symbol is %s with %.2f%% confidence!" % (s, s_confidence * 100))

# Second digit
digit2 = read_image("test_images/two.png")
dPrediction2 = digit_model.predict(digit2)
d2 = dPrediction2.argmax()
d2_confidence = dPrediction2[0][dPrediction2.argmax()]
print("\n\t Your second digit is %d with %.2f%% confidence!" % (d2, d2_confidence * 100))

# Get overall confidence
confidence = d1_confidence * s_confidence * d2_confidence

# Perform the operation
if s == "+":
    print("\n\t Your answer is %d + %d = %d with overall %.2f%% confidence!" % (d1, d2, d1 + d2, confidence * 100))
elif s == "-":
    print("\n\t Your answer is %d - %d = %d with overall %.2f%% confidence!" % (d1, d2, d1 - d2, confidence * 100))
elif s == "*":
    print("\n\t Your answer is %d * %d = %d with overall %.2f%% confidence!" % (d1, d2, d1 * d2, confidence * 100))
elif s == "/":
    print("\n\t Your answer is %d / %d = %.2f with overall %.2f%% confidence!" % (d1, d2, d1 / d2, confidence * 100))
elif s == "=":
    print("\n\t Your equation is %d = %d with overall %.2f%% confidence! Your equation is %b!" % (d1, d2, confidence * 100, d1 == d2))
print()
