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

    @property
    def input_shape(self):
        x, _ = self.training_set
        return x[0].shape

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
        return to_categorical(y, num_classes=MNISTDataSet.NUM_CLASSES + 1)

    @staticmethod
    def to_label(class_index):
        if class_index < 0 or class_index > MNISTDataSet.NUM_CLASSES + 1:
            return '?'

        if MNISTDataSet.is_background(class_index):
            return 'background'

        return str(class_index)

    @staticmethod
    def is_background(class_index):
        return class_index == MNISTDataSet.NUM_CLASSES


class MnistGenerator:

    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._mnist = MNISTDataSet()

    def flow_from_training_data(self):
        x, y = self._mnist.training_set
        return self._generate(x, y)

    def flow_from_validation_data(self):
        x, y = self._mnist.validation_set

        x_norm = self.normalize(x)
        y_1hot = self._mnist.to_one_hot(y)

        return self._datagen.flow(x_norm, y_1hot, batch_size=self._batch_size)

    def _generate(self, x, y):
        m = len(y)

        for i in range(0, m, self._batch_size):
            print(i)
            index_from = i
            index_to = i + self._batch_size
            x_batch = x[index_from:index_to]
            y_batch = y[index_from:index_to]
            x_batch, y_batch = self._shift((x_batch, y_batch))
            x_norm = self.normalize(x_batch)
            y_1hot = self._mnist.to_one_hot(y_batch)

            yield x_norm, y_1hot

    def _shift(self, batch):
        x, y = batch

        p_background = 1.0 / (self._mnist.NUM_CLASSES + 1)

        x_out = np.zeros(self.rank4_shape(x))
        y_out = np.zeros(len(y))

        x = self.to_rank4(x)
        for i in range(len(y)):

            if np.random.random() < p_background:
                x_shifted = self._transform_example(x[i:i+1], max_shift=28)
                y_out[i] = self._mnist.background_class()
            else:
                x_shifted = self._transform_example(x[i:i+1], max_shift=3)
                y_out[i] = y[i]

            x_out[i] = x_shifted

        x_out = self.to_rank4(x_out)
        y_out = np.array(y_out)
        return x_out, y_out

    def _transform_example(self, x, max_shift):
        gen = ImageDataGenerator(width_shift_range=max_shift,
                                 height_shift_range=max_shift)
        for x_batch in gen.flow(x, batch_size=1):
            x_shifted = x_batch[0]
            return x_shifted

    def rank4_shape(self, x_batch):
        batch_size = len(x_batch)
        height, width = self._mnist.input_shape
        return batch_size, height, width, 1

    def to_rank4(self, x_batch):
        return x_batch.reshape(self.rank4_shape(x_batch))

    def normalize(self, x):
        return x / 255


gen = MnistGenerator(batch_size=128)


for x_batch, y_batch in gen.flow_from_training_data():
    print(x_batch.shape)
    print(y_batch.shape)
    print(y_batch[0])
