import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.networks import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from monai.handlers import CheckpointSaver, SegmentationSaver

def train_model(data_dir, output_dir, batch_size, num_epochs):
    # Load preprocessed data
    dataset = Dataset(data_dir, data_dict={"image": "preprocessed_image*.pt", "mask": "preprocessed_mask*.pt"})
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create UNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(dimensions=2, in_channels=1, out_channels=1).to(device)

    # Define loss function and optimizer
    loss_fn = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            inputs, targets = batch["image"].to(device), batch["mask"].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # Save the trained model
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_saver = CheckpointSaver(save_dir=checkpoint_dir, save_dict={"trained_model": model})
    checkpoint_saver.save_checkpoint()

    # Perform inference on test data and save segmentation results
    model.eval()
    post_transforms = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    segmentation_saver = SegmentationSaver(output_dir=output_dir, batch_transform=post_transforms)
    test_data = Dataset(data_dir, data_dict={"image": "preprocessed_image*.pt"})
    dataloader = DataLoader(test_data, batch_size=1)
    for batch in dataloader:
        inputs = batch["image"].to(device)
        outputs = model(inputs)
        segmentation_saver.save_batch(outputs)

# Usage example
data_dir = "preprocessed_data"
output_dir = "saved_models"
batch_size = 4
num_epochs = 10
train_model(data_dir, output_dir, batch_size, num_epochs)
