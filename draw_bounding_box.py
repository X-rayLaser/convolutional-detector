import numpy as np
from PIL.ImageDraw import Image, ImageDraw
from PIL import ImageFont


def array_to_image(a):
    h, w, d = a.shape
    a = np.array(a, dtype=np.uint8).reshape(h, w)

    return Image.frombytes('L', (w, h), a.tobytes())


def visualize_detection(a, boxes, labels):
    image = array_to_image(a)
    canvas = ImageDraw(image)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 15)

    for i in range(len(boxes)):
        xc, yc, w, h = boxes[i]
        label = labels[i]
        x = int(round(xc - w / 2))
        y = int(round(yc - h / 2))

        xy = [(x, y), (x + w, y + h)]
        canvas.rectangle(xy, width=2, outline=128)
        canvas.text((x + 2, y + 2), font=fnt, text=label, fill=255)

    image.show()
