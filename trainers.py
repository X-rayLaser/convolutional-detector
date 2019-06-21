from keras.datasets import mnist
from generators import MNISTDataSet, DataSetGenerator, DirectoryDataSet
from models import model_from_config


def train_on_directory(config_path, training_dir, validation_dir, height,
                       width, gray_scale, batch_size, epochs=2):
    params = dict(path=training_dir, height=height,
                  width=width, gray_scale=gray_scale)

    training_set = DirectoryDataSet(**params)

    training_generator = DataSetGenerator(mnist_dataset=training_set,
                                          batch_size=batch_size)

    train_gen, training_steps = training_generator.flow()

    if validation_dir:
        params['path'] = validation_dir
        validation_set = DirectoryDataSet(**params)
        validation_generator = DataSetGenerator(mnist_dataset=validation_set,
                                                batch_size=batch_size)

        validation_gen, validation_steps = validation_generator.flow()
    else:
        validation_gen, validation_steps = None, None

    builder = model_from_config(config_path)

    shape = training_set.rank3_shape()

    model = builder.get_complete_model(input_shape=shape)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=training_steps,
                                  epochs=epochs,
                                  validation_data=validation_gen,
                                  validation_steps=validation_steps)

    return model, training_set.mapping_table()


def train_on_mnist(config_path, batch_size=32, epochs=2):
    (x_train, y_train), (x_val, y_val) = mnist.load_data()

    training_dataset = MNISTDataSet(x_train, y_train)
    gen = DataSetGenerator(training_dataset, batch_size=batch_size)
    train_gen, training_steps = gen.flow()

    validation_dataset = MNISTDataSet(x_val, y_val)
    gen = DataSetGenerator(validation_dataset, batch_size=batch_size)
    validation_gen, validation_steps = gen.flow()

    builder = model_from_config(config_path)

    shape = training_dataset.rank3_shape()

    model = builder.get_complete_model(input_shape=shape)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=training_steps,
                                  epochs=epochs,
                                  validation_data=validation_gen,
                                  validation_steps=validation_steps)

    return model, training_dataset.mapping_table()
