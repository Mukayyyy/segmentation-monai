import os
import numpy as np
import torch
from monai.transforms import Compose, LoadImage, AddChannel, ScaleIntensity, Resize, ToTensor


def preprocess_data(image_files, mask_files, output_dir, target_size):
        transform = Compose([
            LoadImage(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            Resize(target_size),
            ToTensor()
        ])

        for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
            image = transform(image_file)
            mask = transform(mask_file)

            # Save preprocessed data
            torch.save(image, os.path.join(output_dir, f"preprocessed_image_{i}.pt"))
            torch.save(mask, os.path.join(output_dir, f"preprocessed_mask_{i}.pt"))
