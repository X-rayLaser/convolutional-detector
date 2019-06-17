if __name__ == '__main__':
    import argparse

    architecture_conf_path = ''
    model_save_path = ''

    from generators import MnistGenerator
    from models import build_model

    batch_size = 128
    gen = MnistGenerator(batch_size=batch_size)

    shape = gen._mnist.rank3_shape()
    print(shape)
    num_classes = gen._mnist.total_classes()

    m = gen._mnist.train_size
    m_val = gen._mnist.validation_size

    builder = build_model(input_shape=shape, num_classes=num_classes)

    model = builder.get_complete_model(input_shape=shape)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=gen.flow_from_training_data(),
                                  steps_per_epoch=int(m / batch_size),
                                  epochs=2,
                                  validation_data=gen.flow_from_validation_data(),
                                  validation_steps=int(m_val / batch_size))
