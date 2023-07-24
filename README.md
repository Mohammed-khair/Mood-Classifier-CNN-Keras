# Mood Identifier using CNN and TensorFlow Keras API

This project is a binary classifier that recognizes the mood of a person from a picture of their face. It was completed as an exercise in the course "Convolutional Neural Network" offered on Coursera.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E9-tLzEpsUB35jhlOuCQmVfp0L49P5HN?usp=sharing)



## Overview

In this project, we have built a mood identifier using the TensorFlow Keras API with a Convolutional Neural Network (CNN) architecture. The goal is to classify whether a person in an input image is "happy" or "not happy".

## Model Architecture

The model architecture follows the sequence: ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE.

1. **ZEROPAD2D**: Zero-padding is used to ensure that the spatial dimensions of the feature maps remain the same after applying convolution.

2. **CONV2D**: The convolutional layer applies a set of learnable filters to the input image to extract features.

3. **BATCHNORM**: Batch normalization is used to normalize the activations of the previous layer, which helps in faster convergence during training.

4. **RELU**: The Rectified Linear Unit (ReLU) activation function introduces non-linearity to the model.

5. **MAXPOOL**: Max pooling is applied to downsample the feature maps and retain the most important information.

6. **FLATTEN**: The flattened layer converts the 2D feature maps into a 1D vector for the dense layer.

7. **DENSE**: The dense (fully connected) layer performs the final classification.

## Data Preprocessing

Before feeding the data into the model, the following steps are applied:

1. **Load the Data**: The dataset containing images of happy and not happy faces is loaded.

2. **Split the Data**: The data is split into training and testing sets to evaluate the model's performance.

3. **Normalization**: The input image vectors are normalized to ensure that all pixel values are within the range [0, 1].

4. **Reshape**: The output vectors are reshaped to suit the model's requirements.

## Counting Parameters

The number of parameters in each layer are calculated as follows:

1. **CONV2D Layer**: The number of parameters is determined by the formula:
   ```
   (rows of filter × columns of filter × number of channels of input) × number of filters + number of filters
   ```
   For our model:
   ```
   (7 × 7 × 3) × 32 + 32 = 4,736
   ```

2. **BATCHNORM Layer**: The number of parameters is the sum of gamma, beta, moving mean, and moving variance parameters.
   ```
   Number of gamma parameters + Number of beta parameters + Number of moving mean parameters + Number of moving variance parameters
   ```
   For our model:
   ```
   32 + 32 + 32 + 32 = 128
   ```

3. **DENSE Layer**: The number of parameters is calculated as follows:
   ```
   length of input × number of units + number of units
   ```
   For our model:
   ```
   (32 × 32 × 32) × 1 + 1 = 32,769
   ```

## Training and Evaluation

The model is trained using the Adam optimizer and binary cross-entropy loss. After training, the model is evaluated on the test set.

## Results

The model achieved an impressive accuracy of 96.67% on the test set.

### Conclusion

In this project, we successfully built a mood identifier using a Convolutional Neural Network and the TensorFlow Keras API. The model demonstrated high accuracy in classifying whether a person is "happy" or "not happy" based on their facial expressions. This project serves as an excellent exercise to understand and implement CNNs for image classification tasks. 
