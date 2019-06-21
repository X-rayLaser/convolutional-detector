import argparse
import traceback
import os
import json
from keras.preprocessing.image import img_to_array
from detection_pipeline import detect_locations
from draw_bounding_box import visualize_detection
from generators import RandomCanvasGenerator
from models import model_from_dict


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--canvas_width', type=int, default=200)
    parser.add_argument('--canvas_height', type=int, default=200)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--classifier', type=str, default='MNIST_classifier.h5')
    return parser.parse_args()


def start_interactive_loop(canvas_generator, object_size, index_to_class, model):
    while True:
        try:
            num_digits = int(input('Enter a number of digits:\n'))
            image = canvas_generator.generate_image(num_digits=num_digits)

            bounding_boxes, labels = detect_locations(
                img_to_array(image), model, object_size=object_size,
                index_to_class=index_to_class
            )

            visualize_detection(img_to_array(image), bounding_boxes, labels)
        except Exception:
            traceback.print_exc()
            print('Number of digits must be an integer')


def model_meta_data(classifier_path):
    base_path, _ = os.path.splitext(classifier_path)
    meta_path = base_path + '.meta.json'

    with open(meta_path, 'r') as f:
        s = f.read()

    return json.loads(s)


if __name__ == '__main__':
    args = get_cmd_arguments()

    meta_data = model_meta_data(args.classifier)

    index_to_class = meta_data['index_to_class']
    index_to_class = dict((int(k), v) for k, v in index_to_class.items())

    input_shape = meta_data['training_config']['model_config']['input_shape']
    input_height, input_width, _ = input_shape

    builder = model_from_dict(meta_data['training_config'])

    builder.load_weights(args.classifier)

    model = builder.get_complete_model(input_shape=(args.canvas_height,
                                                    args.canvas_width, 1))

    from generators import MNISTSource, DirectorySource

    if args.dataset_path:
        source = DirectorySource(path=args.dataset_path, height=input_height,
                                 width=input_width)
    else:
        source = MNISTSource()
    gen = RandomCanvasGenerator(source, width=args.canvas_width,
                                height=args.canvas_height)

    start_interactive_loop(gen, (input_height, input_width), index_to_class, model)
