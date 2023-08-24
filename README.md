# Mask R-CNN for Sugarcane Disease Classification
Sugarcane Disease Classification

This project implements the Mask R-CNN (Region-based Convolutional Neural Network) for the task of sugarcane disease classification. It uses a dataset of sugarcane images labeled with various disease types and leverages the power of deep learning to accurately classify and localize diseases within the sugarcane images.

# Prerequisites
Before running the code, ensure you have the following dependencies installed:

TensorFlow 1.15
Keras 2.2.5
h5py 2.10
You can install these dependencies using pip:
pip install tensorflow==1.15
pip install keras==2.2.5
pip install h5py==2.10


# Getting Started
Clone this repository:
git clone (https://github.com/Sauravpandey11/Mask_RCNN_Implementation_On_Sugarcane_diseases/)
Download the sugarcane disease dataset and place it in the dataset/ directory.

Open the Custom.pyk using any compatible notebook environment.

Follow the instructions in the notebook to train the Mask R-CNN model on the sugarcane disease dataset.

# Training
In the custom.py, you'll find code for training the Mask R-CNN model. Adjust the hyperparameters and training settings as needed to achieve the desired performance.

# Testing
Prediction.ipynb contains code that imports the model created by custom.py and uses it to predict the class and mark the segmentation of image that you provided.
