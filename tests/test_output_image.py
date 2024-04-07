import os

import numpy as np

from main import output_image

def test_output_image():
    # Create a test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Call the function with the test image
    output_image(image, "test_output_image.png")

    # Check that the output directory was created
    assert os.path.exists("output_images")
    
    # Check that the output image file was created
    assert os.path.exists("output_images/test_output_image.png")