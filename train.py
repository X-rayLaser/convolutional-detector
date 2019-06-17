if __name__ == '__main__':
    import argparse

    architecture_conf_path = ''
    model_save_path = ''

    from generators import MNISTDataSet, MNISTGenerator
    from models import build_model

    batch_size = 128

    mnist_dataset = MNISTDataSet()
    gen = MNISTGenerator(mnist_dataset=mnist_dataset, batch_size=batch_size)

    shape = mnist_dataset.rank3_shape()

    num_classes = mnist_dataset.total_classes()

    builder = build_model(input_shape=shape, num_classes=num_classes)

    model = builder.get_complete_model(input_shape=shape)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_gen, training_steps = gen.flow_from_training_data()
    validation_gen, validation_steps = gen.flow_from_validation_data()

    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=training_steps,
                                  epochs=2,
                                  validation_data=validation_gen,
                                  validation_steps=validation_steps)
