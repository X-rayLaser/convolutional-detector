import numpy as np
from PIL.ImageDraw import Image, ImageDraw
from PIL import ImageFont


def array_to_image(a):
    h, w, d = a.shape
    a = np.array(a, dtype=np.uint8).reshape(h, w)

    return Image.frombytes('L', (w, h), a.tobytes())


def clip(v, max_val):
    return min(max_val, max(0, v))


def visualize_detection(a, boxes, labels):
    image = array_to_image(a)
    canvas = ImageDraw(image)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 15)

    for i in range(len(boxes)):
        x0, y0, w, h = boxes[i].geometry
        label = labels[i]

        x0 = clip(x0, image.width - 1)
        y0 = clip(y0, image.height - 1)

        x = clip(x0 + w, image.width - 1)
        y = clip(y0 + h, image.height - 1)

        xy = [(x0, y0), (x, y)]
        canvas.rectangle(xy, width=2, outline=128)
        canvas.text((x0 + 2, y0 + 2), font=fnt, text=label, fill=255)

    image.show()
