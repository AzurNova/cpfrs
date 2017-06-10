#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import sys

sys.path.append("..")

# Import Matplotlib:
import matplotlib
matplotlib.use('Agg')

# import facerec modules
from facerec.preprocessing import LBPPreprocessing
from facerec.operators import ChainOperator
from facerec.feature import PCA, SpatialHistogram, Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.serialization import save_model, load_model
from facerec.visual import subplot
from facerec.util import minmax_normalize
import matplotlib.cm as cm

from validation_with_logging import KFoldCrossValidation

# import from other file
import face_detection as fd
import facial_alignment as fa

# import numpy, matplotlib and logging
import numpy as np

# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import logging

CASCADE_PATH = './haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(CASCADE_PATH)
MAX_IMAGE_DIM = 2048
MAX_FACE_DIM = 256
MIN_NUMBER_OF_PICS = 5
ALIGNMENT = True
DEBUG = False


def process_image(im, show_box=False, show_final=False):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = fd.find_face(im, cascade)

    if show_box:
        copy = np.copy(im)
        for (x, y, w, h) in faces:
           cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('1', copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if ALIGNMENT:
        aligned = fa.corrected(im, None, debug=DEBUG)
        if aligned is None:
            return None
        aligned_image, box = aligned
        gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        xp, yp, wp, hp = box
        x, y, w, h = int(xp[0]), int(yp[0]), int(wp[0]), int(hp[0])
    else:
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
    if x < 0 or y < 0 or x + h > gray.shape[0] or y + w > gray.shape[1]:
        return None
    im = gray[y:y + h, x:x + w]

    ## Resize if too large
    #if im.shape[0] > MAX_FACE_DIM or im.shape[1] > MAX_FACE_DIM:
    #    if im.shape[1] > im.shape[0]:
    #        scale = MAX_FACE_DIM / (im.shape[1] * 1.0)
    #        new_dim = (MAX_FACE_DIM, int(im.shape[0] * scale))
    #    else:
    #        scale = MAX_FACE_DIM / (im.shape[0] * 1.0)
    #        new_dim = (int(im.shape[1] * scale), MAX_FACE_DIM)
    #    im = cv2.resize(im, new_dim, interpolation=cv2.INTER_AREA)
    im = cv2.resize(im, (MAX_FACE_DIM, MAX_FACE_DIM), interpolation=cv2.INTER_AREA)

    if show_final:
        cv2.imshow('1', im)
        cv2.waitKey()

    return im


def read_images(path, sz=None):
    c = 0
    X,Y = [], []
    names = []
    total_faces = 0
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            files = os.listdir(subject_path)
            if len(files) < MIN_NUMBER_OF_PICS:
                continue
            print "Reading {0}".format(subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename))
                    # Resize if too large
                    if im.shape[0] > MAX_IMAGE_DIM or im.shape[1] > MAX_IMAGE_DIM:
                        if im.shape[1] > im.shape[0]:
                            scale = MAX_IMAGE_DIM / (im.shape[1] * 1.0)
                            new_dim = (MAX_IMAGE_DIM, int(im.shape[0] * scale))
                        else:
                            scale = MAX_IMAGE_DIM / (im.shape[0] * 1.0)
                            new_dim = (int(im.shape[1] * scale), MAX_IMAGE_DIM)
                        im = cv2.resize(im, new_dim, interpolation=cv2.INTER_AREA)
                    #cv2.imshow('read_images', im)
                    #cv2.waitKey()
                    #cv2.destroyAllWindows()
                    #im = process_image(im)
                    im = process_image(im, show_box=DEBUG, show_final=DEBUG)
                    if im is None:
                        continue
                    X.append(np.asarray(im, dtype=np.uint8))
                    Y.append(c)
                    total_faces += 1
                except IOError as e:
                    print("I/O error: {0}".format(e))
                    raise e
                except:
                    print("Unexpected error: {0}".format(sys.exc_info()[0]))
                    raise
            c = c+1
    print "{0} people. Averaging {1} faces.".format(str(len(names)), total_faces / (len(names) * 1.0))
    return [X,Y], names


def main():
    if len(sys.argv) < 2:
        print("USAGE: facerec_demo.py </path/to/images>")
        sys.exit()

    [X, y], names = read_images(sys.argv[1])
    print "done reading images"

    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler to facerec modules
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Feature Extraction
    feature = Fisherfaces()
    #feature = ChainOperator(LBPPreprocessing(), SpatialHistogram())
    print "extracted features"

    # Classifier
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)

    my_model = PredictableModel(feature=feature, classifier=classifier)
    my_model.compute(X, y)
    print "created model"

    model_name = 'model.pkl'
    save_model(model_name, my_model)
    model = load_model(model_name)

    if isinstance(feature, Fisherfaces):
        # Sample (at most 16) eigenvectors to plot
        E = []
        for i in xrange(min(model.feature.eigenvectors.shape[1], 16)):
            e = model.feature.eigenvectors[:, i].reshape(X[0].shape)
            E.append(minmax_normalize(e, 0, 255, dtype=np.uint8))
        subplot(title="Fisherfaces", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet,
                filename="fisherfaces.png")

    # Perform cross validation
    kfcv = KFoldCrossValidation(model, names=names)
    kfcv.validate(X, y)
    kfcv.print_results()

    # Manual prediction and tracing
    print "==Prediction of image in the model=="
    print names[my_model.predict(process_image(cv2.imread('../facebook/facebook_hard/Helen_Jiang/1.jpg')))[0]]

    print "\n==Prediction of image outside the model=="
    print names[my_model.predict(process_image(cv2.imread('../facebook/facebook_hard_other/dontwannacry.png'), show_box=True, show_final=True))[0]]


if __name__ == '__main__':
    main()
