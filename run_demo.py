import argparse
from keras.preprocessing.image import img_to_array
from detection_pipeline import detect_locations
from draw_bounding_box import visualize_detection
from models import build_model
from generators import RandomCanvasGenerator


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--canvas_width', type=int, default=200)
    parser.add_argument('--canvas_height', type=int, default=200)
    parser.add_argument('--input_width', type=int, default=28)
    parser.add_argument('--input_height', type=int, default=28)
    parser.add_argument('--classifier', type=str, default='MNIST_classifier.h5')
    parser.add_argument('--num_classes', type=int, default=10)
    return parser.parse_args()


def start_interactive_loop(canvas_generator, object_size):
    while True:
        try:
            num_digits = int(input('Enter a number of digits:\n'))
            image = canvas_generator.generate_image(num_digits=num_digits)

            bounding_boxes, labels = detect_locations(
                img_to_array(image), model, object_size=object_size
            )

            visualize_detection(img_to_array(image), bounding_boxes, labels)
        except Exception:
            print('Number of digits must be an integer')


if __name__ == '__main__':
    args = get_cmd_arguments()

    total_num_classes = args.num_classes + 1

    builder = build_model(input_shape=(args.input_height, args.input_width, 1),
                          num_classes=total_num_classes)
    builder.load_weights(args.classifier)

    model = builder.get_complete_model(input_shape=(args.canvas_height,
                                                    args.canvas_width, 1))

    gen = RandomCanvasGenerator(width=args.canvas_width,
                                height=args.canvas_height)

    start_interactive_loop(gen, (args.input_height, args.input_width))
