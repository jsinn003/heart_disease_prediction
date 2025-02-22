# Heart Disease Prediction Model

This repository contains a predictive model that utilizes a neural network to identify the presence of heart disease based on various physiological and medical attributes. The model is built using TensorFlow and trained on the Heart Disease dataset from the UCI Machine Learning Repository.

# Features

- Data preprocessing including scaling and handling missing values.
- Neural network implementation with TensorFlow for binary classification.
- Evaluation of model performance using various metrics such as RMSE, accuracy, precision, recall, F1 score, and MCC.
- Visualization of the confusion matrix.

# Dataset

The dataset used is the Heart Disease dataset from the UCI Machine Learning Repository, which includes attributes like chest pain type, maximum heart rate achieved, and more. The data is preprocessed to handle missing values and normalize features before feeding them into the model.

# Prerequisites

Before you run this program, ensure you have the following installed:
- Python 3.7 or higher
- TensorFlow 2.x
- Scikit-Learn
- Pandas
- Numpy
- Matplotlib
- Seaborn

You can install the necessary libraries using the following command:

```
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

## Usage

To run this program, follow these steps:
1. Clone this repository to your local machine.
2. Ensure that all required libraries are installed.
3. Run the script from the command line by navigating to the directory containing the script and typing:

```
python heart_disease_prediction.py
```

# Model Architecture

The model consists of the following layers:
- Dense layer with 256 neurons and ReLU activation.
- Dropout layer with a rate of 0.3.
- Dense layer with 128 neurons and ReLU activation.
- Another dropout layer with a rate of 0.3.
- Output layer with a single neuron with sigmoid activation for binary classification.

# Evaluation Metrics

After training, the model evaluates the test data using the following metrics:
- Root Mean Squared Error (RMSE)
- Accuracy
- Precision
- Sensitivity (Recall)
- F1 Score
- Specificity
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

The results are printed on the console, and the confusion matrix is visualized using Matplotlib and Seaborn.