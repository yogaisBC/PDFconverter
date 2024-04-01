"""
_summary_
"""
import sys
import os
import shutil
import time

# pylint: disable=no-member,import-error
import cv2 as cv
# pylint: enable=no-member,import-error

import numpy as np

from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)


def main():
    if len(sys.argv) < 2:
        default = "default_name.pdf"
        print(f"Using default file: {default}")
        filename = default
    else:
        filename = sys.argv[1]
    
    start_time = time.process_time()
    images = convert_from_path(filename, dpi = 1200)
    end_time = time.process_time()

    print(f"Time taken to convert: {end_time - start_time} seconds")

    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv_images = []
    for img in images:
        cv_images.append(np.array(img)) 

    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f"image_{i}.png"), "PNG")

        image = cv_images[i]
        coloured_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        blurred = cv.GaussianBlur(gray_image, (5, 5), 0)

        threshold, binary_image = cv.threshold(
            gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )

        # Apply edge detection
        edges = cv.Canny(binary_image, 30, 90, apertureSize=3)

        # Detect lines using Probabilistic Hough Line Transform
        lines = cv.HoughLinesP(
            edges, 1, np.pi / 180, 50, minLineLength=200, maxLineGap=10
        )

        # Draw lines on the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(coloured_image, (x1, y1), (x2, y2), (0, 255, 0), 15)

        ### ? show_image(coloured_image)
        cv.imwrite(os.path.join(output_dir, f"image_highlighted_{i}.png"), coloured_image)


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

if __name__ == "__main__":
    wipe_output()
    starttime = time.process_time()
    main()
    endtime = time.process_time()
    print(f"Total time taken: {endtime - starttime} seconds")

# end of file(EOF)
