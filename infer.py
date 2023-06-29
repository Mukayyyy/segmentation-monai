import os
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import Activations, AsDiscrete
from monai.handlers import SegmentationSaver
from monai.networks.nets import UNet

def inference(data_dir, model_file, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = UNet(dimensions=3, in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)

    # Create a DataLoader for the test data
    test_data = Dataset(data_dir, data_dict={"image": "test_image*.nii.gz"})
    dataloader = DataLoader(test_data, batch_size=1, num_workers=0)

    # Perform inference on the test data and save the segmentation results
    model.eval()
    post_transforms = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    segmentation_saver = SegmentationSaver(output_dir=output_dir, batch_transform=post_transforms)
    
    for batch in dataloader:
        inputs = batch["image"].to(device)
        outputs = model(inputs)
        segmentation_saver.save_batch(outputs)

# Usage example
data_dir = "test_data"
model_file = "path/to/trained_model.pth"
output_dir = "output_data"
inference(data_dir, model_file, output_dir)
