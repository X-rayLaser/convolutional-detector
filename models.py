import json
from keras.layers import Input, InputLayer, Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.models import Sequential
from keras import Model


class ModelBuilder:
    def __init__(self, input_shape, initial_num_filters=6):
        self._filters = initial_num_filters
        self._model = Sequential()
        self._model.add(InputLayer(input_shape=input_shape))
        h, w, d = input_shape
        self._input_width = w
        self._input_height = h

    def add_conv_layer(self, last_one=False, kernel_size=(3, 3)):
        kwargs = dict(filters=self._filters, kernel_size=kernel_size,
                      kernel_initializer='he_normal', activation='relu')
        if last_one:
            kwargs['name'] = 'last_feature_extraction_layer'

        layer = Conv2D(**kwargs)

        self._model.add(layer)
        self._filters *= 2
        self._input_height -= (kernel_size[0] - 1)
        self._input_width -= (kernel_size[1] - 1)

        return self

    def add_pooling_layer(self):
        self._model.add(MaxPool2D(pool_size=(2, 2)))
        self._input_width = self._input_width // 2
        self._input_height = self._input_height // 2

        return self

    def add_batch_norm_layer(self):
        self._model.add(BatchNormalization())
        return self

    def add_dropout_layer(self, drop_prob=0.5):
        self._model.add(Dropout(drop_prob))
        return self

    def add_fully_connected_layer(self):
        self._model.add(
            Conv2D(filters=self._filters,
                   kernel_size=(self._input_height, self._input_width),
                   kernel_initializer='he_normal', activation='relu')
        )

        self._input_height = 1
        self._input_width = 1
        return self

    def add_output_layer(self, num_classes):
        self._model.add(Conv2D(filters=num_classes, kernel_size=(1, 1),
                               kernel_initializer='he_normal',
                               activation='softmax')
                        )
        return self

    def add_binary_classification_layer(self):
        self._model.add(Conv2D(filters=1, kernel_size=(1, 1),
                               kernel_initializer='he_normal',
                               activation='sigmoid')
                        )
        return self

    def load_weights(self, path):
        self._model.load_weights(path)

    def get_complete_model(self, input_shape):
        inp = Input(shape=input_shape)
        x = inp
        for layer in self._model.layers:
            x = layer(x)

        return Model(input=inp, output=x)

    def index_of_last_extraction_layer(self):
        for i in range(len(self._model.layers)):
            layer = self._model.layers[i]
            if layer.name == 'last_feature_extraction_layer':
                return i

    def get_feature_extractor(self, input_shape):
        inp = Input(shape=input_shape)
        x = inp

        for i in range(self.index_of_last_extraction_layer() + 1):
            layer = self._model.layers[i]
            x = layer(x)

        out = x
        return Model(input=inp, output=out)

    def get_classifier(self, input_shape):
        inp = Input(shape=input_shape)
        x = inp

        pooling_index = self.index_of_last_extraction_layer() + 1

        for i in range(pooling_index, len(self._model.layers)):
            layer = self._model.layers[i]
            x = layer(x)

        out = x
        return Model(input=inp, output=out)


def build_model(input_shape, num_classes):
    builder = ModelBuilder(input_shape)
    builder.add_conv_layer().add_batch_norm_layer()
    builder.add_conv_layer().add_batch_norm_layer()
    builder.add_conv_layer(last_one=True).add_batch_norm_layer()
    builder.add_fully_connected_layer().add_batch_norm_layer().add_dropout_layer(drop_prob=0.2)
    builder.add_fully_connected_layer().add_batch_norm_layer().add_dropout_layer()
    builder.add_output_layer(num_classes=num_classes)

    return builder


def model_from_config(path):
    with open(path, 'r') as f:
        s = f.read()

    d = json.loads(s)

    return model_from_dict(d)


def model_from_dict(config_dictionary):
    model_config = config_dictionary['model_config']
    input_shape = model_config['input_shape']
    initial_num_filters = model_config['initial_num_filters']
    kernel_size = model_config['kernel_size']
    convolutional_layers = model_config['convolutional_layers']
    drop_prob = model_config['drop_prob']

    num_classes = config_dictionary['num_classes']

    num_background_classes = 1

    total_classes = num_classes + num_background_classes

    builder = ModelBuilder(input_shape, initial_num_filters=initial_num_filters)

    for _ in range(convolutional_layers - 1):
        builder.add_conv_layer(kernel_size=kernel_size).add_batch_norm_layer()

    builder.add_conv_layer(last_one=True, kernel_size=kernel_size)
    builder.add_batch_norm_layer()

    builder.add_fully_connected_layer().add_dropout_layer(drop_prob=drop_prob)
    builder.add_fully_connected_layer()
    builder.add_output_layer(num_classes=total_classes)

    return builder
