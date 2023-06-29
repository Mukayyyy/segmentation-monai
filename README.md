## Usage

1. Prepare the data:
- Download the MSD dataset and extract the DICOM images and corresponding labels.
- Preprocess the data using the provided `data_preprocessing.py` script to normalize intensities, resample images, and perform data augmentation.

2. Train the model:
- Run the `train.py` script to train the segmentation model on the preprocessed data.
- Adjust the hyperparameters and network architecture in the script to suit your needs.

3. Evaluate the model:
- Use the `evaluation.py` script to evaluate the trained model's performance on a separate validation dataset.
- Adjust any necessary evaluation metrics or post-processing transforms in the script.

4. Inference:
- Perform inference on new medical images using the trained model with the `inference.py` script.
- Adjust the input paths and other parameters in the script accordingly.

## Contributing

Contributions to this project are welcome. Feel free to open issues or submit pull requests to improve the code or add new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
