import json
import os
import numpy as np
import cv2

# Specify the path to your COCO JSON file and the directory to save the mask images
json_file_path = '/home/zfchen/working/VISION_dataset/screw/train/_annotations.coco.json'
#current_image_dir = '/home/zfchen/working/VISION_dataset/screw/train'
mask_save_dir = '/home/zfchen/working/VISION_dataset/screw/train_mask'
converted_mask_save_dir = '/home/zfchen/working/VISION_dataset/screw/converted_ground_truth'


# Create the directory to save the mask images if it does not exist
#if not os.path.exists(mask_save_dir):
os.makedirs(mask_save_dir, exist_ok=True)
os.makedirs(converted_mask_save_dir, exist_ok=True)

# Read the JSON file
with open(json_file_path, 'r') as file:
    coco_data = json.load(file)

# Create a dictionary to store image dimensions
image_dimensions = {}
for image_info in coco_data['images']:
    image_dimensions[image_info['id']] = (image_info['width'], image_info['height'])

# Iterate over the annotations in the COCO dataset
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    segmentation = annotation['segmentation']

    # Get the dimensions for the current image
    image_width, image_height = image_dimensions.get(image_id, (None, None))

    # If dimensions are not found, skip this annotation
    if image_width is None or image_height is None:
        print(f"Dimensions not found for image ID: {image_id}")
        continue

    # Create an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    converted_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for segment in segmentation:
        # Reshape segment data to a 2D array
        polygon = np.array(segment).reshape((-1, 2))
        # Fill the polygon on the mask
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        cv2.fillPoly(converted_mask, [polygon.astype(np.int32)], category_id + 1)

    # Save the mask to the specified directory
    mask_file_path = os.path.join(mask_save_dir, f'{image_id}_mask.png')
    converted_mask_file_path = os.path.join(converted_mask_save_dir, f'{image_id}_mask.png')
    cv2.imwrite(mask_file_path, mask)
    cv2.imwrite(converted_mask_file_path, converted_mask)

print("Mask creation completed!")

