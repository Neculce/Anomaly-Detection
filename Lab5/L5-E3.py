import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel() # Flatten to 1D array

# Split data (50% for testing as requested)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, shuffle=True
)

# Normalize data to [0, 1]
# We fit the scaler on training data and transform both train and test

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training shape: {X_train_scaled.shape}")
print(f"Testing shape: {X_test_scaled.shape}")
class AnomalyAutoencoder(Model):
    def __init__(self):
        super(AnomalyAutoencoder, self).__init__()
        
        # Encoder: Compresses input [9] -> [8] -> [5] -> [3]
        self.encoder = keras.Sequential([
            layers.Dense(8, activation='relu'),
            layers.Dense(5, activation='relu'),
            layers.Dense(3, activation='relu')
        ])
        
        # Decoder: Reconstructs [3] -> [5] -> [8] -> [9]
        # Last layer uses sigmoid to match the [0, 1] input range
        self.decoder = keras.Sequential([
            layers.Dense(5, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(9, activation='sigmoid') 
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate model
autoencoder = AnomalyAutoencoder()

autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
# We use X_train_scaled as both input and target (x=X, y=X)
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test_scaled, X_test_scaled),
    shuffle=True,
    verbose=1
)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training History")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()



# Function to calculate reconstruction error (MSE per sample)
def get_reconstruction_error(model, data):
    reconstructions = model.predict(data)
    # Mean Squared Error for each sample across features
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    return mse

# Get errors for training and testing
train_scores = get_reconstruction_error(autoencoder, X_train_scaled)
test_scores = get_reconstruction_error(autoencoder, X_test_scaled)

# Determine Threshold
# We assume the contamination rate in the training set is representative of the dataset.
# In ODDS, 'y=1' usually denotes the outlier.
contamination_rate = np.sum(y) / len(y)
print(f"Dataset Contamination Rate: {contamination_rate:.4f}")

# Calculate threshold using quantile on Training scores
# We want the top (contamination_rate)% of errors to be flagged
threshold = np.quantile(train_scores, 1 - contamination_rate)
print(f"Anomaly Threshold: {threshold:.6f}")

# Generate Predictions (1 if error > threshold, else 0)
y_pred_train = (train_scores > threshold).astype(int)
y_pred_test = (test_scores > threshold).astype(int)

# Calculate Balanced Accuracy
b_acc_train = balanced_accuracy_score(y_train, y_pred_train)
b_acc_test = balanced_accuracy_score(y_test, y_pred_test)

print("-" * 30)
print(f"Balanced Accuracy (Train): {b_acc_train:.4f}")
print(f"Balanced Accuracy (Test):  {b_acc_test:.4f}")
print("-" * 30)

# Optional: Visualize distribution of scores
plt.figure(figsize=(10, 6))
plt.hist(train_scores, bins=50, alpha=0.6, label='Train Reconstruction Errors')
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Frequency")
plt.title("Distribution of Reconstruction Errors")
plt.legend()
plt.show()
