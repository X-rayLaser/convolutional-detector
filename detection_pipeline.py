from shapely.geometry import box
from keras.preprocessing.image import img_to_array
from models import build_model
from generators import RandomCanvasGenerator
from draw_bounding_box import visualize_detection


class BoundingBox:
    def __init__(self, xy, width, height):
        x0, y0 = xy

        self._x = x0
        self._y = y0
        self._width = width
        self._height = height

    @property
    def geometry(self):
        return self._x, self._y, self._width, self._height

    def to_shapely_box(self):
        x, y, w, h = self.geometry
        return box(x, y, x + w, y + h)

    def IoU(self, other_box):
        b1 = self.to_shapely_box()
        b2 = other_box.to_shapely_box()

        intersection = b1.intersection(b2).area

        union = b1.union(b2).area

        if union == 0:
            return 1

        return intersection / union


def non_max_suppression(label_detections, iou_threshold=0.1):
    boxes = [result.bounding_box for result in label_detections]
    probs = [result.score for result in label_detections]

    pairs = list(zip(boxes, probs))
    pairs.sort(key=lambda t: t[1])

    rems = list(pairs)
    survived_indices = []
    while rems:
        bounding_box, prob = rems.pop()
        index = probs.index(prob)
        survived_indices.append(index)

        def small_iou(t):
            b, p = t
            return bounding_box.IoU(b) < iou_threshold

        rems = list(filter(small_iou, rems))

    return survived_indices


def thresholding(y_hat, p_threshold=0.2):
    h, w, d = y_hat.shape

    for i in range(h):
        for j in range(w):
            p_object = 1 - y_hat[i, j, -1]
            if p_object < p_threshold:
                y_hat[:d - 1, :d - 1] = 0

    return y_hat[:, :, :d - 1]


class DetectionResult:
    def __init__(self, bounding_box, confidence_score, predicted_class):
        self.bounding_box = bounding_box
        self.score = confidence_score
        self.predicted_class = predicted_class


class HeatMap:
    def __init__(self, feature_map, map_index, object_size):
        self._a = feature_map
        self._object_size = object_size
        self._map_index = map_index

    def detect_boxes(self, p_threshold=0.9):
        object_height, object_width = self._object_size

        prediction_grid = self._a

        mask = prediction_grid > p_threshold

        prediction_grid = prediction_grid * mask

        rows, cols = prediction_grid.shape

        results = []
        for row in range(rows):
            for col in range(cols):
                score = prediction_grid[row, col]

                if score > p_threshold:
                    bounding_box = BoundingBox((col, row), object_width,
                                               object_height)

                    res = DetectionResult(bounding_box, score,
                                          str(self._map_index))
                    results.append(res)

        return results

    def get_bounding_boxes(self):
        return self.detect_boxes()

    def non_max_suppression(self):
        detection_results = self.get_bounding_boxes()

        indices = non_max_suppression(detection_results, iou_threshold=0.2)
        return [detection_results[i] for i in indices]


def detect_locations(image, model, object_size):
    image_height, image_width, _ = image.shape

    y_pred = model.predict(image.reshape(1, image_height,
                                         image_width, 1) / 255.0)[0]

    y_pred = thresholding(y_pred)

    results = []

    for k in range(10):
        a = y_pred[:, :, k]
        heat_map = HeatMap(feature_map=a, map_index=k, object_size=object_size)

        results.extend(heat_map.non_max_suppression())

    indices = non_max_suppression(results, iou_threshold=0.2)

    cleaned_boxes = [results[i].bounding_box for i in indices]
    cleaned_labels = [results[i].predicted_class for i in indices]
    return cleaned_boxes, cleaned_labels


if __name__ == '__main__':
    img_width = 200
    img_height = 200

    object_height = 28
    object_width = object_height

    builder = build_model(input_shape=(object_height, object_width, 1), num_classes=11)
    builder.load_weights('MNIST_classifier.h5')

    model = builder.get_complete_model(input_shape=(200, 200, 1))

    gen = RandomCanvasGenerator(width=img_width, height=img_height)
    image = gen.generate_image(num_digits=10)

    bounding_boxes, labels = detect_locations(
        img_to_array(image), model, object_size=(object_height, object_width)
    )

    visualize_detection(img_to_array(image), bounding_boxes, labels)
