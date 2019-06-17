from shapely.geometry import box

def IoU(box1, box2):
    xc1, yc1, w1, h1 = box1
    xc2, yc2, w2, h2 = box2

    x1 = xc1 - 45 // 2
    y1 = yc1 - 45 // 2

    x2 = xc2 - 45 // 2
    y2 = yc2 - 45 // 2
    b1 = box(x1, y1, x1 + w1, y1 + h1)
    b2 = box(x2, y2, x2 + w2, y2 + h2)

    intersection = b1.intersection(b2).area

    union = b1.union(b2).area

    if union == 0:
        return 1

    return intersection / union


def non_max_suppression(boxes, probs, iou_threshold=0.1):
    pairs = list(zip(boxes, probs))
    pairs.sort(key=lambda t: t[1])

    rems = list(pairs)
    survived_indices = []
    while rems:
        box, prob = rems.pop()
        index = probs.index(prob)
        survived_indices.append(index)

        def small_iou(t):
            b, p = t
            return IoU(box, b) < iou_threshold

        rems = list(filter(small_iou, rems))

    return survived_indices


def detect_boxes(prediction_grid, width, height, p_threshold=0.9):
    mask = prediction_grid > p_threshold

    prediction_grid = prediction_grid * mask

    boxes = []
    scores = []

    rows, cols = prediction_grid.shape

    for row in range(rows):
        for col in range(cols):
            if prediction_grid[row, col] > p_threshold:
                x = col + 45 // 2
                y = row + 45 // 2

                boxes.append((x, y, 45, 45))
                scores.append(prediction_grid[row, col])

    return boxes, scores


def extract_patch(image, box, delta_x, delta_y):
    height, width = image.shape
    xc, yc, w, h = box

    x = xc - 45 // 2
    y = yc - 45 // 2

    col = int(round(x + delta_x))
    row = int(round(y + delta_y))

    col = min(width - w - 1, max(0, col))
    row = min(height - h - 1, max(0, row))

    return image[row:row + h, col:col + w]


def cropped_areas(image, box):
    pixel_shift = 1

    for i in range(-2, 2, 1):
        for j in range(-2, 2, 1):
            delta_x = i * pixel_shift
            delta_y = j * pixel_shift
            yield extract_patch(image, box, delta_x, delta_y)


def fit_classifier_input_size(height, width, classifier):
    from keras.layers import Input
    from keras import Model

    image_patch_shape = (height, width, 1)
    inp = Input(shape=image_patch_shape)
    x = inp
    for layer in classifier.layers:
        x = layer(x)

    return Model(input=inp, output=x)


def recognize_object(image, classifier):
    h, w = image.shape

    image = image.reshape((1, h, w, 1)) / 255.0
    y_hat = classifier.predict(image)[0]

    v = y_hat.max(axis=(0, 1))
    pred = index_to_class[v.argmax()]
    score = v.max()
    return pred, score


def group_indices(labels):
    groups = {}
    for i in range(len(labels)):
        label = labels[i]
        if label not in groups:
            groups[label] = []

        groups[label].append(i)

    return groups


def suppress_class_wise(boxes, scores, labels):
    groups = group_indices(labels)
    cleaned_groups = dict(groups)
    for label, indices in groups.items():
        label_boxes = [boxes[i] for i in indices]
        label_scores = [scores[i] for i in indices]
        remaining_indices = non_max_suppression(label_boxes, label_scores, iou_threshold=0.02)
        cleaned_groups[label] = [indices[i] for i in remaining_indices]

    rem_boxes = []
    rem_labels = []
    rem_scores = []
    for label, indices in cleaned_groups.items():
        rem_boxes.extend([boxes[i] for i in indices])
        rem_labels.extend([label] * len(indices))
        rem_scores.extend([scores[i] for i in indices])

    return rem_boxes, rem_scores, rem_labels


def get_candidate_boxes(image, dmodel):
    img_height, img_width = image.shape
    output_shape = dmodel.output_shape[1:]

    y_pred = dmodel.predict(image.reshape(1, img_height, img_width, 1) / 255.0)
    y_pred = y_pred.reshape(output_shape)[:, :, 0]

    boxes, scores = detect_boxes(y_pred, img_width, img_height)
    remaining_indices = non_max_suppression(boxes, scores, iou_threshold=0.6)

    return [boxes[i] for i in remaining_indices]


def detect_locations(image, model):
    y_pred = model.predict(image.reshape(1, img_height, img_width, 1) / 255.0)[0]

    h, w, d = y_pred.shape
    for i in range(h):
        for j in range(w):
            p_object = 1 - y_pred[i, j, 10]
            if p_object < 0.2:
                y_pred[:10, :10] = 0

    y_pred = y_pred[:, :, :10]
    all_boxes = []
    all_scores = []
    all_labels = []
    for k in range(10):
        boxes, scores = detect_boxes(y_pred[:, :, k], img_width, img_height)
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_labels.extend([str(k)] * len(boxes))

    boxes, scores, labels = suppress_class_wise(all_boxes, all_scores, all_labels)

    indices = non_max_suppression(boxes, scores, iou_threshold=0.2)

    cleaned_boxes = [boxes[i] for i in indices]
    cleaned_labels = [labels[i] for i in indices]
    return cleaned_boxes, cleaned_labels


if __name__ == '__main__':
    img_width = 200
    img_height = 200

    from models import build_model

    from generators import RandomCanvasGenerator

    builder = build_model(input_shape=(28, 28, 1), num_classes=11)
    builder.load_weights('MNIST_classifier.h5')

    model = builder.get_complete_model(input_shape=(200, 200, 1))

    gen = RandomCanvasGenerator(width=img_width, height=img_height)
    image = gen.generate_image(num_digits=4)

    from keras.preprocessing.image import img_to_array

    bounding_boxes, labels = detect_locations(img_to_array(image), model)

    from draw_bounding_box import visualize_detection
    visualize_detection(img_to_array(image), bounding_boxes, labels)
