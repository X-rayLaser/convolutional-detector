import argparse
from generators import MNISTDataSet, MNISTGenerator
from models import build_model


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, default='')
    parser.add_argument('--validation_dir', type=str, default='')
    parser.add_argument('--save_path', type=str, default='MNIST_classifier.h5')
    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_cmd_arguments()

    mnist_dataset = MNISTDataSet()
    gen = MNISTGenerator(mnist_dataset=mnist_dataset, batch_size=args.batch_size)
    train_gen, training_steps = gen.flow_from_training_data()
    validation_gen, validation_steps = gen.flow_from_validation_data()

    shape = mnist_dataset.rank3_shape()

    num_classes = mnist_dataset.total_classes()

    builder = build_model(input_shape=shape, num_classes=num_classes)

    model = builder.get_complete_model(input_shape=shape)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epochs = args.epochs

    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=training_steps,
                                  epochs=epochs,
                                  validation_data=validation_gen,
                                  validation_steps=validation_steps)

    model.save(args.save_path)
