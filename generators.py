import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from PIL.ImageDraw import ImageDraw
from PIL import Image


class BaseDataSet:
    def __init__(self, height, width, gray_scale=True):
        self._height = height
        self._width = width
        self._gray_scale = gray_scale

    def total_classes(self):
        return self.num_classes + 1

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        return self._height, self._width

    def rank3_shape(self):
        height, width = self.input_shape

        if self._gray_scale:
            channels = 1
        else:
            channels = 3
        return height, width, channels

    def batch_shape(self, batch_size):
        return (batch_size, ) + self.rank3_shape()

    @property
    def size(self):
        raise NotImplementedError

    def background_class(self):
        return self.num_classes

    def to_one_hot(self, y):
        return to_categorical(y, num_classes=self.total_classes())

    def to_label(self, class_index):
        if class_index < 0 or class_index > self.total_classes():
            return '?'

        if self.is_background(class_index):
            return 'background'

        return str(class_index)

    def mapping_table(self):
        raise NotImplementedError

    def to_index(self, label):
        raise NotImplementedError

    def is_background(self, class_index):
        return class_index == self.num_classes

    def mini_batches(self, batch_size):
        raise NotImplementedError


class MNISTDataSet(BaseDataSet):
    def __init__(self, x, y, gray_scale=True):
        super().__init__(height=28, width=28, gray_scale=gray_scale)
        self._data = (x, y)

    @property
    def num_classes(self):
        return 10

    @property
    def size(self):
        _, y = self._data
        return len(y)

    def to_index(self, label):
        return int(label)

    def mapping_table(self):
        return dict((i, str(i)) for i in range(10))

    def mini_batches(self, batch_size):
        x, y = self._data
        x = x.reshape(self.batch_shape(len(y)))
        gen = ImageDataGenerator()
        return gen.flow(x, y, batch_size=batch_size)


class DirectoryDataSet(BaseDataSet):
    def __init__(self, path, height, width, gray_scale=True):
        super().__init__(height=height, width=width, gray_scale=gray_scale)
        self._path = path
        self._target_size = (height, width)
        self._gen = ImageDataGenerator()
        self._class_to_index = None

    @property
    def num_classes(self):
        return len(os.listdir(self._path))

    @property
    def size(self):
        count = 0
        for label_dir in os.listdir(self._path):
            label_path = os.path.join(self._path, label_dir)
            for fname in os.listdir(label_path):
                file_path = os.path.join(label_path, fname)
                if os.path.isfile(file_path):
                    count += 1

        return count

    def mapping_table(self):
        class_to_index = self._class_to_index
        return dict((index, label) for label, index in class_to_index)

    def mini_batches(self, batch_size):
        if self._gray_scale:
            color_mode = "grayscale"
        else:
            color_mode = "rgb"
        gen = self._gen.flow_from_directory(directory=self._path,
                                            target_size=self._target_size,
                                            color_mode=color_mode,
                                            class_mode='sparse',
                                            batch_size=batch_size)

        self._class_to_index = gen.class_indices.items()

        for x_batch, y_batch in gen:
            x_batch = 255 - x_batch
            yield x_batch, y_batch


class DataSetGenerator:

    SHIFT_TOLERANCE = 0

    def __init__(self, mnist_dataset, batch_size, p_background=None):
        self._mnist = mnist_dataset
        self._batch_size = batch_size

        if not p_background:
            self._p_background = 1.0 / (self._mnist.total_classes())
        else:
            self._p_background = p_background

    def flow(self):
        m = self._mnist.size
        batch_generator = self._mnist.mini_batches(self._batch_size)
        return self._generate(batch_generator), self._steps_per_epoch(m)

    def _steps_per_epoch(self, m):
        if m < self._batch_size:
            return 1

        return int(m / self._batch_size)

    def _generate(self, batch_generator):
        for x_batch, y_batch in batch_generator:
            x_batch, y_batch = self._shift((x_batch, y_batch))
            x_norm = self._normalize(x_batch)
            y_1hot = self._mnist.to_one_hot(y_batch)

            y_1hot = y_1hot.reshape(
                (len(y_1hot), 1, 1, self._mnist.total_classes())
            )

            yield x_norm, y_1hot

    def _shift(self, batch):
        x, y = batch

        x_out = np.zeros(self._mnist.batch_shape(len(x)))
        y_out = np.zeros(len(y))

        x = self._to_rank4(x)
        for i in range(len(y)):
            image_width = self._mnist.input_shape[0]

            if np.random.random() < self._p_background:
                x_shifted = self._transform_example(x[i:i+1],
                                                    max_shift=image_width)
                y_out[i] = self._mnist.background_class()
            else:
                x_shifted = self._transform_example(
                    x[i:i+1], max_shift=self.SHIFT_TOLERANCE
                )
                y_out[i] = y[i]

            if np.random.random() < 0.5:
                for _ in range(2):
                    x_shifted = self._add_noise(x_shifted, x)

            x_out[i] = x_shifted

        x_out = self._to_rank4(x_out)
        y_out = np.array(y_out)
        return x_out, y_out

    def _add_noise(self, x_shifted, x_batch):
        index = np.random.randint(0, len(x_batch) - 1)
        image_width = self._mnist.input_shape[0]

        noise = self._transform_example(x_batch[index:index + 1],
                                        max_shift=image_width)

        return np.minimum(255, x_shifted + noise)

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


class CharacterSource:
    def __init__(self):
        self._groups = None

    def load_data(self):
        raise NotImplementedError

    def array_to_image(self, a):
        h, w = a.shape
        a = np.array(a, dtype=np.uint8)
        return Image.frombytes('L', (w, h), a.tobytes())

    def _group_examples(self, x, y):
        groups = {}
        for i in range(len(y)):
            label = y[i]
            if label not in groups:
                groups[label] = []

            groups[label].append((i, x[i]))

        return groups

    def category_text_to_class(self, category_text):
        return int(category_text)

    def get_digit_image(self, category_name):
        class_name = self.category_text_to_class(category_name)
        tuples = self._groups[class_name]

        index = np.random.randint(0, len(tuples) - 1)
        index, x = tuples[index]

        return self.array_to_image(x)


class MNISTSource(CharacterSource):
    def load_data(self):
        from keras.datasets import mnist
        _, (x_test, y_test) = mnist.load_data()
        self._groups = self._group_examples(x_test, y_test)

    @property
    def num_classes(self):
        return 10


class DirectorySource(CharacterSource):
    def __init__(self, path, height, width):
        super().__init__()
        self._path = path
        self._height = height
        self._width = width
        self._data_set = DirectoryDataSet(self._path, height=self._height,
                                          width=self._width)

    @property
    def num_classes(self):
        return self._data_set.num_classes

    def load_data(self):
        data_set = self._data_set
        for x_batch, y_batch in data_set.mini_batches(data_set.size):
            x = x_batch.reshape(data_set.size, self._height, self._width)
            y = y_batch
            self._groups = self._group_examples(x, y)
            break


class RandomCanvasGenerator:
    def __init__(self, source, width=200, height=200, character_size=28):
        self._factory = source
        self._width = width
        self._height = height
        self._char_size = character_size

        self._factory.load_data()
        self._num_classes = self._factory.num_classes

    def generate_image(self, num_digits=30, noise=None):
        if noise:
            a = np.array(np.random.random_integers(0, 64, (self._height, self._width)), dtype=np.uint8)
        else:
            a = np.zeros((self._height, self._width), dtype=np.uint8)

        im = Image.frombytes('L', (self._width, self._height), a.tobytes())
        canvas = ImageDraw(im)

        for i in range(num_digits):
            import random

            x = random.randint(0, self._width - self._char_size)
            y = random.randint(0, self._height - self._char_size)

            class_index = random.randint(0, self._num_classes - 1)
            ch = str(class_index)

            character_bitmap = self._factory.get_digit_image(ch)
            character_bitmap = character_bitmap.resize(
                (self._char_size, self._char_size)
            )

            canvas.bitmap((x, y), character_bitmap, fill=255)

        return im
