import cv2
import numpy as np

MODEL = "source/model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "source/model/MobileNetSSD_deploy.prototxt.txt"


def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


def annotate_image(
    image, detections, confidence_threshold=0.5
):
    # loop over the detections
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), 70, 2)
    return image


def edit_distance(source, target):
    cache = [[float("inf") for _ in range(len(target)+1)]
             for _ in range(len(source)+1)]
    for i in range(len(target)+1):
        cache[0][i] = i
    for i in range(len(source)+1):
        cache[i][0] = i

    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if source[i - 1] == target[j - 1]:
                cache[i][j] = cache[i - 1][j - 1]
            else:
                cache[i][j] = min(
                    cache[i][j - 1], min(cache[i - 1][j], cache[i - 1][j - 1])) + 1
    return cache[len(source)][len(target)]
