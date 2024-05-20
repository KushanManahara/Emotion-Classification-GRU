# Emotion Classification with GRU

## Overview

This project aims to classify emotions in textual data using a deep learning model based on GRU (Gated Recurrent Unit) neural networks. The model is trained on a dataset containing text samples labeled with dominant emotions such as joy, sadness, anger, etc. The trained model can then be used to predict the dominant emotion expressed in new text samples.

## Dataset

The dataset used in this project consists of text samples labeled with one of several emotion categories. It is split into training, testing, and validation sets for model development and evaluation.

## Model Architecture

The GRU model architecture consists of:
- Input layer
- GRU layer with 128 units
- Dropout layer with a dropout rate of 0.2
- Dense layer with 64 units and ReLU activation
- Output layer with softmax activation

## Evaluation

The model is evaluated on the test dataset using metrics such as loss and accuracy. Additionally, a confusion matrix is generated to visualize the model's performance in classifying different emotions.

## Usage

To train the model and evaluate its performance, follow these steps:
1. Load and preprocess the dataset.
2. Build the GRU model using TensorFlow/Keras.
3. Compile the model with appropriate loss and optimization functions.
4. Train the model on the training data.
5. Evaluate the model on the test dataset.
6. Visualize the model's performance using metrics and confusion matrix.

## Dependencies

- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
