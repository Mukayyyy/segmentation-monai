import os
import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

def evaluate(ground_truth_dir, predicted_dir):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_transforms = AsDiscrete(threshold_values=True)

    dice_scores = []
    for filename in os.listdir(ground_truth_dir):
        ground_truth = np.load(os.path.join(ground_truth_dir, filename))
        predicted = np.load(os.path.join(predicted_dir, filename))

        ground_truth = post_transforms(ground_truth)
        predicted = post_transforms(predicted)

        dice_score = dice_metric(predicted, ground_truth)
        dice_scores.append(dice_score.item())

    mean_dice = np.mean(dice_scores)
    return mean_dice

# Usage example
ground_truth_dir = "ground_truth_labels"
predicted_dir = "predicted_segmentations"
mean_dice = evaluate(ground_truth_dir, predicted_dir)
print("Mean Dice Score:", mean_dice)
