#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

# Unconfuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    pass

import sys
import numpy as np
import glob
import os

# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image

MAX_IMAGE_DIM = 1024


def remove_subsuming_boxes(arr):
    if len(arr) == 0:
        return arr
    pick = set(range(arr.shape[0]))
    for i in range(len(arr)):
        a = arr[i]
        for j in range(len(arr)):
            b = arr[j]
            if a[0] < b[0] and a[1] < b[1] and a[0] + a[2] > b[0] + b[2] and a[1] + a[3] > b[1] + b[3]:
                pick.discard(j)
    return arr[list(pick)]


def find_face(im, fc, name=None):
    # Resize if too large
    if im.shape[0] > MAX_IMAGE_DIM or im.shape[1] > MAX_IMAGE_DIM:
        if im.shape[1] > im.shape[0]:
            scale = MAX_IMAGE_DIM / (im.shape[1] * 1.0)
            new_dim = (MAX_IMAGE_DIM, int(im.shape[0] * scale))
        else:
            scale = MAX_IMAGE_DIM / (im.shape[0] * 1.0)
            new_dim = (int(im.shape[1] * scale), MAX_IMAGE_DIM)
        im = cv2.resize(im, new_dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    faces = remove_subsuming_boxes(faces)
    #print "{0}: Found {1} faces!".format(name, len(faces))
    #print faces
    # Draw a rectangle around the faces
    copy = np.copy(im)
    for (x, y, w, h) in faces:
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if name is not None:
        cv2.imshow(name, copy)
    return faces


def main():
    if len(sys.argv) < 2:
        print("USAGE: face_detection.py </path/to/images> [</path/to/cascade>]")
        sys.exit()

    if len(sys.argv) < 3:
        cascade_path = './haarcascade_frontalface_default.xml'
    else:
        cascade_path = sys.argv[2]
    image_folder_path = sys.argv[1]
    for filename in os.listdir(image_folder_path):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            image = cv2.imread(os.path.join(image_folder_path, filename))
            cascade = cv2.CascadeClassifier(cascade_path)
            find_face(image, cascade, filename)
    cv2.waitKey()

if __name__ == "__main__":
    main()
