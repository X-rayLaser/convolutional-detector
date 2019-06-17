from random import shuffle
import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


class MNISTDataSet:
    NUM_CLASSES = 10

    def __init__(self):
        train_data, dev_data = mnist.load_data()
        self._train_data = train_data
        self._dev_data = dev_data

    @staticmethod
    def total_classes():
        return MNISTDataSet.NUM_CLASSES + 1

    @property
    def input_shape(self):
        x, _ = self.training_set
        return x[0].shape

    def rank3_shape(self):
        height, width = self.input_shape
        return height, width, 1

    def batch_shape(self, batch_size):
        height, width = self.input_shape
        return batch_size, height, width, 1

    @property
    def train_size(self):
        _, y = self.training_set
        return len(y)

    @property
    def validation_size(self):
        _, y = self.validation_set
        return len(y)

    @property
    def training_set(self):
        return self._train_data

    @property
    def validation_set(self):
        return self._dev_data

    @staticmethod
    def background_class():
        return MNISTDataSet.NUM_CLASSES

    @staticmethod
    def to_one_hot(y):
        return to_categorical(y, num_classes=MNISTDataSet.total_classes())

    @staticmethod
    def to_label(class_index):
        if class_index < 0 or class_index > MNISTDataSet.total_classes():
            return '?'

        if MNISTDataSet.is_background(class_index):
            return 'background'

        return str(class_index)

    @staticmethod
    def is_background(class_index):
        return class_index == MNISTDataSet.NUM_CLASSES


# todo: create dataset wrapper class that is agnostic about whether it works with training data or validation data
# todo: rewrite existing generator code in terms of that new class


class MNISTGenerator:

    SHIFT_TOLERANCE = 3

    def __init__(self, mnist_dataset, batch_size, p_background=None):
        self._mnist = mnist_dataset
        self._batch_size = batch_size

        if not p_background:
            self._p_background = 1.0 / (self._mnist.NUM_CLASSES + 1)
        else:
            self._p_background = p_background

    def flow_from_training_data(self):
        x, y = self._mnist.training_set
        return self._generate(x, y), self._steps_per_epoch(len(y))

    def flow_from_validation_data(self):
        x, y = self._mnist.validation_set
        return self._generate(x, y), self._steps_per_epoch(len(y))

    def _steps_per_epoch(self, m):
        if m < self._batch_size:
            return 1

        return int(m / self._batch_size)

    def _generate(self, x, y):
        m = len(y)

        while True:
            x, y = self._sorted(x, y)

            for i in range(0, m, self._batch_size):
                index_from = i
                index_to = i + self._batch_size
                x_batch = x[index_from:index_to]
                y_batch = y[index_from:index_to]
                x_batch, y_batch = self._shift((x_batch, y_batch))
                x_norm = self._normalize(x_batch)
                y_1hot = self._mnist.to_one_hot(y_batch)

                y_1hot = y_1hot.reshape(
                    (len(y_1hot), 1, 1, self._mnist.total_classes())
                )

                yield x_norm, y_1hot

    def _sorted(self, x, y):
        m = len(y)

        indices = list(range(m))
        shuffle(indices)

        x_out = []
        y_out = []
        for index in indices:
            x_out.append(x[index])
            y_out.append(y[index])

        x_out = np.array(x_out)
        y_out = np.array(y_out)
        return x_out, y_out

    def _shift(self, batch):
        x, y = batch

        x_out = np.zeros(self._mnist.batch_shape(len(x)))
        y_out = np.zeros(len(y))

        x = self._to_rank4(x)
        for i in range(len(y)):

            if np.random.random() < self._p_background:
                image_width = self._mnist.input_shape[0]
                x_shifted = self._transform_example(x[i:i+1], max_shift=image_width)
                y_out[i] = self._mnist.background_class()
            else:
                x_shifted = self._transform_example(
                    x[i:i+1], max_shift=self.SHIFT_TOLERANCE
                )
                y_out[i] = y[i]

            x_out[i] = x_shifted

        x_out = self._to_rank4(x_out)
        y_out = np.array(y_out)
        return x_out, y_out

    def _transform_example(self, x, max_shift):
        gen = ImageDataGenerator(width_shift_range=max_shift,
                                 height_shift_range=max_shift)
        for x_batch in gen.flow(x, batch_size=1):
            x_shifted = x_batch[0]
            return x_shifted

    def _to_rank4(self, x_batch):
        return x_batch.reshape(self._mnist.batch_shape(len(x_batch)))

    def _normalize(self, x):
        return x / 255.0
