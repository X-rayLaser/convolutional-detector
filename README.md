# Introduction

# Installation

Clone this repository into a local directory, go inside that directory, install all dependencies
```
pip install -r requirements.txt
```

# Quick Start

## Working via CLI

Train the classifier on MNIST data set using configuration file
'training_config.json' and save the model in 'MNIST_classifier.h5'

```
python train.py --save_path 'MNIST_classifier.h5' --config 'training_config.json'
```

Train the classifier using a directory of images

```
python train.py --training_dir 'path/to/training_images' --validation_dir 'path/to/validation_images'
```

Test model interactively, by showing image and detection results one at at time
```
python run_demo.py --canvas_width 200 --canvas_height 300 --classifier 'MNIST_classifier.h5' 
```

Run demo using different image generator where image of each character is
drawn from a directory (see below the format of the directory)
```
python run_demo.py --dataset_path '<path/to/images_directory' 
```

## Usage in Python project

```
from detection_pipeline import detect_locations
from keras.preprocessing.image import img_to_array
...
a = img_to_array(image)
index_to_class = {0: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
object_size = (28, 28)
bounding_boxes, labels = detect_locations(
    a, model, object_size=object_size,
    index_to_class=index_to_class
)
box = bounding_boxes[0]
label = labels[0]
x0, y0, w, h = box.geometry
...
```