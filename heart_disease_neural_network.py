# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Fetch the Heart Disease dataset from the UCI Machine Learning Repository
heart_disease = fetch_ucirepo(id=45)

# Extract features and target variable from the dataset
X = heart_disease.data.features
y = heart_disease.data.targets

# Data Preprocessing: Combine features and target into a single DataFrame for easy manipulation
data = pd.concat([X, y], axis=1)
print(data.isnull().sum())  # Check for missing values in the data

# Drop rows with missing values in 'ca' and 'thal' columns
data.dropna(subset=['ca', 'thal'], inplace=True)

# Preparing feature matrix X and target vector y after dropping 'num' which is the target column
X = data.drop('num', axis=1)
y = data['num']

# Binarize the target variable
y[y >= 2] = 1

# Selecting specific features for the model
X = X[['cp', 'thalach', 'oldpeak', 'ca', 'thal']]
print(X)
print(y)

# Scale features to have zero mean and unit variance
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
X = scaled_features

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a neural network model for binary classification
model = tf.keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),  # First dense layer
    layers.Dropout(0.3),  # Dropout to reduce overfitting
    layers.Dense(128, activation="relu"),  # Second dense layer
    layers.Dropout(0.3),  # Another dropout layer
    layers.Dense(1, activation="sigmoid")  # Output layer with sigmoid activation for binary classification
])

# Compile the model with Adam optimizer and mean squared error as loss function
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model for 20 epochs with a batch size of 32
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions on the test set
predictions = model.predict(X_test)
y_pred = (predictions > 0.5).astype(int)

# Evaluate the model using RMSE (Root Mean Squared Error)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate and display various performance metrics
accuracy = round(accuracy_score(y_test, y_pred), 2)
print(f'Accuracy: {accuracy}')
precision = round(precision_score(y_test, y_pred), 2)
print("Precision:", precision)
sensitivity = round(recall_score(y_test, y_pred), 2)
print("Sensitivity:", sensitivity)
f1 = round(f1_score(y_test, y_pred), 2)
print("F1 Score:", f1)
tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.shape == (2,2) else (conf_matrix[0,0], conf_matrix[0,1:].sum(), conf_matrix[1:,0].sum(), conf_matrix[1:,1:].sum())
specificity = round(tn / (tn + fp), 2)
print("Specificity:", specificity)
classification_error = round((1 - accuracy), 2)
print("Classification Error:", classification_error)
mcc = round(matthews_corrcoef(y_test, y_pred), 2)
print("MCC:", mcc)

# Print a detailed classification report
print("Classification Report:\n", classification_report(y_test, y_pred))