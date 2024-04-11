"""
_summary_
"""

import sys
import os
import shutil
import time

import cv2 as cv

import numpy as np

from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

from pydantic import BaseModel

class BoundingLines(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

def main():

    if len(sys.argv) < 2:
        default = "default_name.pdf"
        print(f"No file specified, using default file: {default}")
        filename = default
    else:
        filename = sys.argv[1]

    images = pdf_to_image(filename)

    cv_images = []
    for img in images:
        cv_images.append(np.array(img))

    for i, image in enumerate(images):

        image = cv_images[i]
        coloured_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # writable_box(gray_image, coloured_image, i)
        bounding_box(gray_image, coloured_image, i)


def writable_box(gray_image, coloured_image, i):

    temp = coloured_image.copy()

    blurred = cv.GaussianBlur(gray_image, (7, 7), 0)

    threshold, binary_image = cv.threshold(
        blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    # Edge detection and PHLT for different aperture sizes
    detect_and_draw_lines(binary_image, 3, temp, i)
    detect_and_draw_lines(binary_image, 5, temp, i)
    detect_and_draw_lines(binary_image, 7, temp, i)

    output_image(temp, f"writable_box_{i}.png")


def detect_and_draw_lines(image, apertureSize, coloured_image, i):
    # Apply edge detection
    edges = cv.Canny(image, 30, 90, apertureSize=apertureSize)

    # Dilate the edges
    dilated_edges = cv.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    # Erode the dilated edges
    closed_edges = cv.erode(dilated_edges, np.ones((5, 5), np.uint8), iterations=2)

    smaller_edges = cv.erode(closed_edges, np.ones((3, 3), np.uint8), iterations=1)

    output_image(smaller_edges, f"edges_{apertureSize}_page{i}.png")

    # Detect lines using Probabilistic Hough Line Transform
    lines1 = cv.HoughLinesP(
        smaller_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=0
    )

    lines2 = cv.HoughLinesP(
        smaller_edges, 1, np.pi / 180, 50, minLineLength=200, maxLineGap=0
    )

    lines3 = cv.HoughLinesP(
        smaller_edges, 1, np.pi / 180, 50, minLineLength=150, maxLineGap=20
    )

    # Draw lines on the image
    if lines1 is not None:
        for line in lines1:
            x1, y1, x2, y2 = line[0]
            cv.line(coloured_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv.line(coloured_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    if lines3 is not None:
        for line in lines3:
            x1, y1, x2, y2 = line[0]
            cv.line(coloured_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    output_image(coloured_image, f"writable_aperture_{apertureSize}_page{i}.png")

def bounding_box(image, coloured_image, i):

    temp = coloured_image.copy()

    blurred = cv.GaussianBlur(image, (5, 5), 0)

    output_image(blurred, f"blurred_{i}.png")

    _, threshold = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    
    boundstart = time.process_time()
    
    bounding_rects = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # Draw a filled rectangle on the mask
        bounding_rects.append((x,y, x+w, y+h))

    boundstop = time.process_time()
    print(f"Bounding box time: {boundstop - boundstart} seconds")


    mergestart = time.process_time()
    merged_rects = []
    while len(bounding_rects) > 0:
        x1, y1, x2, y2 = bounding_rects.pop(0)
        merged = False
        for j, (mx1, my1, mx2, my2) in enumerate(merged_rects):
            if (max(x1, mx1) <= min(x2, mx2) and max(y1, my1) <= min(y2, my2)):
                merged_rects[j] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                merged = True

                #remove after coding
                print(f"merged: {merged_rects[j]}")

                break
        if not merged:
            merged_rects.append((x1, y1, x2, y2))
    mergestop = time.process_time()
    print(f"Merge time: {mergestop - mergestart} seconds")
    # shade in rectangles
    for x1, y1, x2, y2 in merged_rects:
        print(f"rectangle: {x1, y1, x2, y2}")
        cv.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), -1)
    
    alpha = 0.5
    output = cv.addWeighted(coloured_image, 1 - alpha, temp, alpha, 0)
    
    output_image(output, f"bounding_box_{i}.png")

def pdf_to_image(filename):

    start_time = time.process_time()
    images = convert_from_path(filename, dpi=1200)
    end_time = time.process_time()

    print(f"Time taken to convert: {end_time - start_time} seconds")

    return images


def show_image(image):
    screen_res = 1920, 1080  # Change this to your screen resolution
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)

    # Resized dimensions
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    dim = (width, height)

    # resize image
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    cv.imshow("Resized Image", resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


def wipe_output():
    if os.path.exists("output_images"):
        shutil.rmtree("output_images")


def output_image(image, filename):

    output_dir = "output_images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv.imwrite(os.path.join(output_dir, filename), image)


if __name__ == "__main__":
    wipe_output()
    starttime = time.process_time()
    main()
    endtime = time.process_time()
    print(f"Total time taken: {endtime - starttime} seconds")

# end of file(EOF)
