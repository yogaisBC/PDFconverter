import os
from main import pdf_to_image
from PIL import Image

def test_pdf_to_image():
    # Call the function with a test PDF file
    images = pdf_to_image("tests/testfile.pdf")
    # Check that the function returns a list
    assert isinstance(images, list)

    # Check that each item in the list is an instance of PIL.Image.Image (which represents an image)
    assert all(isinstance(image, Image.Image) for image in images)

    # Check that the function correctly converts the PDF file to images
    assert len(images) == 1