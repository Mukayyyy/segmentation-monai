import os
import glob
import numpy as np
from monai.transforms import Compose, RandRotate, RandFlip, RandZoom

data_dir = "dataset"  # Path to the MSD dataset
output_dir = "augmented_data"  # Path to save augmented data

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all the image files in the dataset directory
image_files = glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz"))

# Define the data augmentation transforms
transforms = Compose([
    RandRotate(range_x=15, prob=0.5),
    RandFlip(prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
])

# Loop through each image file
for image_file in image_files:
    # Load the image data
    image = np.load(image_file)

    # Apply data augmentation
    augmented_image = transforms(image)

    # Save the augmented image
    file_name = os.path.basename(image_file)
    output_file = os.path.join(output_dir, file_name)
    np.save(output_file, augmented_image)

    print(f"Augmented image saved: {output_file}")
