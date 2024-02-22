import cv2
import numpy as np
from PIL import Image
def semantic_mask_to_rgb(mask_path, output_path):
    # Define a color for each of the 11 possible class values (0 through 10)
    colors = [
        (0, 0, 0),       # 0: Black
        (0, 0, 255),     # 1: Blue
        (0, 255, 0),     # 2: Green
        (255, 0, 0),     # 3: Red
        (0, 255, 255),   # 4: Yellow
        (255, 0, 255),   # 5: Magenta
        (255, 255, 0),   # 6: Cyan
        (128, 0, 0),     # 7: Dark Red
        (0, 128, 0),     # 8: Dark Green
        (0, 0, 128),     # 9: Dark Blue
        (128, 128, 128)  # 10: Gray
    ]

    # Read the semantic mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert the grayscale mask to an RGB image
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(11):  # we have 11 classes
        rgb_mask[mask == i] = colors[i]
    im = Image.fromarray(rgb_mask)
    im.save("/home/zfchen/working/diff_aug/test_test_test/sample_capsule/convert_color/test.png")
    # Save the RGB image
    #cv2.imwrite(output_path, rgb_mask)


# Example usage
mask_path = '/home/zfchen/working/diff_aug/test_test_test/sample_capsule/masks/your_file.png'  # Replace with your input image path
output_path = '/home/zfchen/working/diff_aug/test_test_test/sample_capsule/convert_color'       # Replace with desired output path
semantic_mask_to_rgb(mask_path, output_path)