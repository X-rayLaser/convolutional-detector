import argparse
import os
import json
from generators import MNISTDataSet, MNISTGenerator, DirectoryDataSet
from models import build_model, model_from_config
from trainers import train_on_mnist


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, default='')
    parser.add_argument('--validation_dir', type=str, default='')
    parser.add_argument('--save_path', type=str, default='MNIST_classifier2.h5')
    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--grayscale', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_config', type=str, default='model_config.json')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_cmd_arguments()

    if args.training_dir:
        from trainers import train_on_directory
        model, mapping = train_on_directory(config_path=args.model_config,
                                            training_dir=args.training_dir,
                                            validation_dir=args.validation_dir,
                                            height=args.image_height,
                                            width=args.image_width,
                                            num_classes=args.num_classes,
                                            gray_scale=args.grayscale,
                                            batch_size=args.batch_size)
    else:
        model, mapping = train_on_mnist(config_path=args.model_config,
                                        batch_size=args.batch_size,
                                        epochs=args.epochs)

    base_path, _ = os.path.splitext(args.save_path)
    meta_path = base_path + '.meta.json'

    with open(args.model_config, 'r') as f:
        s = f.read()

    model_meta = {
        'model_path': args.save_path,
        'index_to_class': mapping,
        'model_config': json.loads(s)
    }

    model.save(args.save_path)

    with open(meta_path, 'w') as f:
        f.write(json.dumps(model_meta))
