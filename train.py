import argparse
import os
import json
from trainers import train_on_mnist


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, default='')
    parser.add_argument('--validation_dir', type=str, default='')
    parser.add_argument('--save_path', type=str, default='MNIST_classifier2.h5')
    parser.add_argument('--model_config', type=str, default='model_config.json')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_cmd_arguments()

    with open(args.model_config) as f:
        s = f.read()

    session_config = json.loads(s)
    model_config = session_config['model_config']

    height, width, channels = model_config['input_shape']
    batch_size = session_config['batch_size']
    epochs = session_config['epochs']
    gray_scale = (channels == 1)

    if args.training_dir:
        from trainers import train_on_directory
        model, mapping = train_on_directory(config_path=args.model_config,
                                            training_dir=args.training_dir,
                                            validation_dir=args.validation_dir,
                                            height=height,
                                            width=width,
                                            gray_scale=gray_scale,
                                            batch_size=batch_size,
                                            epochs=epochs)
    else:
        model, mapping = train_on_mnist(config_path=args.model_config,
                                        batch_size=batch_size,
                                        epochs=epochs)

    base_path, _ = os.path.splitext(args.save_path)
    meta_path = base_path + '.meta.json'

    model_meta = {
        'model_path': args.save_path,
        'index_to_class': mapping,
        'training_config': session_config
    }

    model.save(args.save_path)

    with open(meta_path, 'w') as f:
        f.write(json.dumps(model_meta))
