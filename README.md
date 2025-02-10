# Butterfly Classification CNN

This project implements a Convolutional Neural Network (CNN) for classifying butterfly species from images. It utilizes PyTorch for model building, training, and evaluation. The model is trained on a subset of the [Butterfly Image Classification dataset](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/data) from Kaggle.

**Note:** This project only utilizes the `training` folder from the dataset because the `test_labels.csv` file does not contain labels for the images.

## Model Architecture

The CNN architecture used in this project consists of the following layers:

*   **Convolutional Layers:** Two convolutional layers with ReLU activation functions and max-pooling layers for feature extraction.
*   **Fully Connected Layers:** Two fully connected layers for classification.

The model is trained using the CrossEntropyLoss function and the Adam optimizer

## Prerequisites

Before running this project, ensure you have the following prerequisites installed:

*   **Python 3.6 or higher**
*   **PyTorch**
*   **Torchvision**
*   **Pandas**
*   **Scikit-learn**
*   **Matplotlib**
*   **Shutil**
*   **Numpy**

You can install all the required packages using pip:

``` pip install -r requirements.txt```

et voila , bon codage (:
