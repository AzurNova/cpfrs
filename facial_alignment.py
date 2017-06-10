import numpy as np
import argparse, os
import dlib, cv2, scipy.optimize
import imutils
from PIL import Image
from random import randint


def detection_to_bounds(detection):
    x, y = detection.left(), detection.top()
    w, h = detection.right() - x, detection.bottom() - y
    return x, y, w, h


def bounds_to_detection(bounds):
    x, y, w, h = bounds
    return dlib.rectangle(int(x), int(y), int(x+w), int(y+h))


def prediction_to_points(prediction, dtype="int"):
    points = [[prediction.part(i).x, prediction.part(i).y] for i in range(68)]
    return np.array(points, dtype=dtype)


def bounds_of(points):
    min_x = min(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_x = max(points, key=lambda p: p[0])[0]
    max_y = max(points, key=lambda p: p[1])[1]
    full = max(max_x - min_x, max_y - min_y)
    half = full / 2
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    return (mid_x - half, mid_y - half, full, full)


def buffer_bounds(array, theta):
    Y, X, _ = array.shape
    Yp = int(Y * abs(np.cos(theta)) + X * abs(np.sin(theta)))
    Xp = int(Y * abs(np.sin(theta)) + X * abs(np.cos(theta)))
    return Yp, Xp


def array_rotation_buffer(array, theta):
    Y, X, _ = array.shape
    theta = np.radians(theta)
    Yp, Xp = buffer_bounds(array, theta)
    newarray = np.zeros((Yp, Xp, 3), dtype='uint8')
    for y in range(Y):
        for x in range(X):
            newarray[y + (Yp - Y) / 2, x + (Xp - X) / 2] = array[y, x]
    return newarray


def draw_facial_bounds(image, bounds, face_num=-1):
    x, y, w, h = bounds
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "face %d" % (face_num + 1), (x, y - 10), 0, 0.5, (0, 255, 0), 2)
    return image


def draw_facial_features(image, points, select=None, colored=False):
    if select != None:
        new_points = []
        for i in range(len(select)):
            left, right = select[i]
            new_points.append(points[left])
            new_points.append(points[right])
        points = new_points
    color = (0, 0, 255)
    for i in range(len(points)):
        x, y = points[i]
        if colored and i % 2 == 0:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            while np.linalg.norm(color) < 128:
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(image, (x, y), 1, color, -1)


def standalone(image, show_bounds=True, show_features=True, show=False, title='Output'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('predictor.dat')

    H, W, _ = np.array(image).shape
    image = imutils.resize(image, 1024 * W / max([H, W]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections = detector(gray, 1)
    for (i, detection) in enumerate(detections):
        prediction = predictor(gray, detection)
        bounds = detection_to_bounds(detection)
        if show_bounds:
            draw_facial_bounds(image, bounds, i)
        if show_features:
            points = prediction_to_points(prediction)
            draw_facial_features(image, points)
    if show:
        cv2.imshow(image_file, image)
        cv2.waitKey(0)
    return prediction


def aligned(image, bounds=None, show_bounds=True, show_features=True, show=False, title='Original', colored=False):
    predictor = dlib.shape_predictor('predictor.dat')

    H, W, _ = np.array(image).shape
    original = imutils.resize(image, 1024 * W / max([H, W]))
    image = imutils.resize(image, 1024 * W / max([H, W]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if bounds != None:
        detection = bounds_to_detection(bounds)
    else:
        detector = dlib.get_frontal_face_detector()
        detections = detector(gray, 1)
        if len(detections) == 0:
            return None
        else:
            detection = detections[0]
        bounds = detection_to_bounds(detection)
    prediction = predictor(gray, detection)
    points = prediction_to_points(prediction)
    pairs, angle = optimal_angle(points)

    if show_bounds:
        points = prediction_to_points(prediction)
        bounds = bounds_of(points)
        draw_facial_bounds(image, bounds)
    if show_features:
        draw_facial_features(image, points, pairs, colored)
    if show:
        cv2.imshow(title, image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return original, points, angle


def optimal_angle(points):
    eyes = [[36, 45],
            [37, 44],
            [38, 43],
            [39, 42],
            [40, 47],
            [41, 46]]
    lips = [[48, 54],
            [49, 53],
            [50, 52],
            [59, 55],
            [58, 56]]
    brow = [[17, 26],
            [18, 25],
            [19, 24],
            [20, 23],
            [21, 22]]
    pairs = []
    pairs.extend(eyes)
    pairs.extend(lips)
    pairs.extend(brow)

    def error(theta):
        err = []
        for pair in pairs:
            x1, y1 = points[pair[0]]
            x2, y2 = points[pair[1]]
            dx, dy = float(x2 - x1), float(y2 - y1)
            err.append(np.linalg.norm([dx, dy]) * np.sin(np.arctan(dy / dx) - theta))
        return np.linalg.norm(err)

    theta = scipy.optimize.minimize(error, 0.).x
    return pairs, np.degrees(theta)


def rotate(points, image, angle):
    angle = np.radians(angle)
    Yp, Xp = buffer_bounds(image, angle)
    Y, X, _ = image.shape
    new_points = []
    for x, y in points:
        x, y = x - X / 2, y - Y / 2
        xp = x * np.cos(-angle) - y * np.sin(-angle)
        yp = x * np.sin(-angle) + y * np.cos(-angle)
        new_points.append([xp + Xp / 2, yp + Yp / 2])
    return new_points


def prepared(image, points, angle, show=False):
    buf = array_rotation_buffer(image, angle)
    rotated = np.array(Image.fromarray(buf).rotate(angle))
    points = rotate(points, image, angle)
    bounds = bounds_of(points)
    if show:
        copy = np.copy(rotated)
        draw_facial_bounds(copy, bounds)
        draw_facial_features(copy, points)
        cv2.imshow('Aligned', copy)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return rotated, bounds


def corrected(image, box, debug=False):
    #cv2.imshow('test', image)
    out = aligned(image, box, show_bounds=debug, show_features=debug, show=debug, colored=True)
    if out is None:
        return None
    return prepared(*out, show=debug)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='image path')
    path = vars(ap.parse_args())['path']
    for image_file in os.listdir(path):
            image = cv2.imread(os.path.join(path, image_file))
            # standalone(image, show_bounds=True, show_features=True, show=True, title=image_file)
            box = [179, 271, 354, 354]
            # out = aligned(image, None, show_bounds=True, show_features=True, show=True, title=image_file, colored=True)
            # prepared(*out, show=True)
            corrected(image, None, debug=True)
