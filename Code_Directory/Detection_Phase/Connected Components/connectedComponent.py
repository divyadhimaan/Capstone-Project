#!/usr/bin/python

from PIL import Image

import operator
import os
import random
import numpy as np
import cv2
from itertools import product
from unionFindArray import *


def run(img):
    data = img.load()
    width, height = img.size

    # Union find data structure
    uf = UFarray()

    # Dictionary of point:label pairs
    labels = {}

    for y, x in product(range(height), range(width)):

        #
        # Pixel names were chosen as shown:
        #
        #       x -->
        #     -------------
        #  y  | a | b | c |
        #     -------------
        #     | d | e |   |
        #     -------------
        #     |   |   |   |
        #     -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #

        # If the current pixel is white, it's obviously not a component...
        if data[x, y] == 255:
            pass

        # If pixel b is in the image and black:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and data[x, y - 1] == 0:
            labels[x, y] = labels[(x, y - 1)]

        # If pixel c is in the image and black:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x + 1 < width and y > 0 and data[x + 1, y - 1] == 0:

            c = labels[(x + 1, y - 1)]
            labels[x, y] = c

            # If pixel a is in the image and black:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and data[x - 1, y - 1] == 0:
                a = labels[(x - 1, y - 1)]
                uf.union(c, a)

            # If pixel d is in the image and black:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and data[x - 1, y] == 0:
                d = labels[(x - 1, y)]
                uf.union(c, d)

        # If pixel a is in the image and black:
        #    We already know b and c are white
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
            labels[x, y] = labels[(x - 1, y - 1)]

        # If pixel d is in the image and black
        #    We already know a, b, and c are white
        #    so simpy assign d's label to e
        elif x > 0 and data[x - 1, y] == 0:
            labels[x, y] = labels[(x - 1, y)]

        # All the neighboring pixels are white,
        # Therefore the current pixel is a new component
        else:
            labels[x, y] = uf.makeLabel()

    uf.flatten()

    colors = {}

    # Image to display the components in a nice, colorful way
    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()

    for (x, y) in labels:

        # Name of the component the current point belongs to
        component = uf.find(labels[(x, y)])

        # Update the labels with correct information
        labels[(x, y)] = component

        # Associate a random color with this component
        if component not in colors:
            colors[component] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        # Colorize the image
        outdata[x, y] = colors[component]

    return (labels, output_img)


def main():
    images_dir = "../OCR/OCR_Results"
    input_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), images_dir
    )
    total_files = 0
    processed_files = 0

    for filename in os.listdir(input_path):
        total_files = total_files + 1

        fileSize = os.stat(os.path.join(input_path,filename)).st_size

        print("Processing" + filename)
        if fileSize != 0:

            processed_files = processed_files + 1
            img = Image.open(os.path.join(input_path,filename))
            temp = np.array(img)

            grayscale = img.convert("L")
            xtra, thresh = cv2.threshold(
                np.array(grayscale), 127, 255, cv2.THRESH_BINARY_INV
            )

            # Threshold the image, this implementation is designed to process b+w
            # Project_Images only
            img = img.point(lambda p: p > 128 and 255)
            img = img.convert("1")

            # labels is a dictionary of the connected component data in the form:
            #     (x_coordinate, y_coordinate) : component_id
            #
            # if you plan on processing the component data, this is probably what you
            # will want to use
            #
            # output_image is just a frivolous way to visualize the components.
            (labels, output_img) = run(img)

            s = sorted(set(val for val in labels.values()))
            points = {}
            for x in s:
                points[x] = []

            for k, v in labels.items():
                points[v].append(k)

            cropByConnectedComponent(points, temp, filename)

    print(str(processed_files) + "/" + str(total_files) + " files processed successfully")
    print("Processing Complete.")
    print("You may check the Result folder in the same directory to check the cropped Project_Images.")


def cropByConnectedComponent(points, temp, filename):

    sig = {}
    for data in points.values():
        data = np.array(data)
        x, y, w, h = cv2.boundingRect(data)
        tup = (x, y, w, h)
        sig[tup] = w * h

    # sort by values
    sorted_x = sorted(sig.items(), key=operator.itemgetter(1))
    (t, v) = sorted_x[-1]
    (x, y, w, h) = t

    temp_np = temp[y: y + h, x: x + w]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ConnectedComponents_Results")

    if not os.path.exists(path):
        os.makedirs(path)

    s1 = "CC_Result_" + filename
    cv2.imwrite(os.path.join(path, s1), temp_np)


if __name__ == "__main__":
    main()
