import pandas
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

data_directory = pathlib.WindowsPath("./math_dataset/extracted_images")
CLASSES = np.array([item.name for item in data_directory.glob('*') if item.name != "LICENSE.txt"])

print(CLASSES)

list_dataset = tf.data.Dataset.list_files(str(data_directory/'*/*'))

class DataSetCreator(object):
    def __init__(self, batch_size, image_height, image_width, dataset):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.dataset = dataset

    def _get_class(self, path):
        pat_splited = tf.strings.split(path, os.path.sep)
        return pat_splited[-2]

    def _load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.image.resize(image, [self.image_height, self.image_width])

    def _load_labeled_data(self, path):
        label = self._get_class(path)
        image = self._load_image(path)
        return image, label

    def load_process(self, shuffle_size = 1000):
        self.loaded_dataset = self.dataset.map(self._load_labeled_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.loaded_dataset = self.loaded_dataset.cache()

        # Shuffle data and create batches
        self.loaded_dataset = self.loaded_dataset.shuffle(buffer_size=shuffle_size)
        self.loaded_dataset = self.loaded_dataset.repeat()
        self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def get_batch(self):
        return next(iter(self.loaded_dataset))



dataProcessor = DataSetCreator(32, 300, 500, list_dataset)
dataProcessor.load_process()

image_batch, label_batch = dataProcessor.get_batch()
print(image_batch)
print(label_batch)
