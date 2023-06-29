import monai.transforms as transforms

def preprocess_data(images, masks):
    # Define the data preprocessing transforms
    transform = transforms.Compose([
        transforms.RescaleIntensity(),
        transforms.Resample(target_spacing=(1.0, 1.0, 1.0)),
        transforms.ToTensor()
    ])

    # Apply the transforms to the images and masks
    preprocessed_images = transform(images)
    preprocessed_masks = transform(masks)

    return preprocessed_images, preprocessed_masks
