# Introduction

This repository contains a simple implementation of the convolutional 
sliding window for objection detection built with Keras.

Concretely, it contains a pre-trained convolutional net trained on 
MNIST dataset, a script for training a custom convolutional net model on 
your own images and the demo script that shows the detection system in action.

## Limitations

What is not supported:
- RGB/RGBA images
- relatively large images (800x800 pixels and above)
- detection of more than 1 object occupying the same region on the image
- training a system to detect relatively large objects (200x200 pixels and 
above)

That said, one can use the system or get an inspiration from it to write a 
component of an OCR system or something similar. 

## Pipeline architecture

The detection system is based around the convolutional neural net 
which consists of interleaving convolutional and batch normalization 
layers followed by fully-connected layers and a Softmax layer. Crucially, 
the architecture does not contain pooling layers. That is essential for 
doing precise object localization.

The input can be a gray-scale image of any size which may contain any number of 
objects. Given the image, the network is run densely on the whole image 
to produce a 3D volume of predictions. This is followed by a 
post-processing phase which mostly consists of extracting bounding 
boxes and corresponding labels and running Non-Max Suppression algorithm.

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

Train the classifier using a directory of images (see below the required 
layout of the directory)

```
python train.py --training_dir 'path/to/training_images' --validation_dir 'path/to/validation_images'
```

Test model interactively, by showing image and detection results one at at time
```
python run_demo.py --canvas_width 200 --canvas_height 300 --classifier 'MNIST_classifier.h5' 
```

Run demo using different image generator where image of each character is
drawn from a directory (see below the required layout of the directory)
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

# Training a convolutional net on your own data images

In order to train with your own images of characters (or any other primitives) 
stored in some folder, that folder must have a specific structure. 
Specifically, for **C** distinct classes there need to be **C** subfolders named 
after corresponding class name. Each of those folders needs to contain images 
belonging to the class corresponding to the folder. For more details, look at 
the directory layout requirements of the flow_from_directory method in Keras 
documentation: https://keras.io/preprocessing/image/

Below is an example of directory layout for hypothetical jpeg pictures of 
cats and dogs:
```
cats_and_dogs/
  cat/
     cat1.jpg
     cat2.jpg
  dog/
     dog1.jpg
     dog2.jpg

 ```