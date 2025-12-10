import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# Load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape to (N, 28, 28, 1) for Conv2D layers
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Function to add noise
def add_noise(images, noise_factor=0.35):
    noisy_images = images + noise_factor * tf.random.normal(shape=images.shape)
    # Clip values to stay within [0, 1]
    noisy_images = tf.clip_by_value(noisy_images, clip_value_min=0., clip_value_max=1.)
    return noisy_images.numpy()

# Create noisy versions
x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

# --- 2. Model Architecture ---

class ConvAutoencoder(Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            # Layer 1: 8 filters, stride 2 (downsamples from 28x28 -> 14x14)
            layers.Conv2D(8, (3, 3), strides=2, padding='same', activation='relu'),
            # Layer 2: 4 filters, stride 2 (downsamples from 14x14 -> 7x7)
            layers.Conv2D(4, (3, 3), strides=2, padding='same', activation='relu')
        ])
        
        # Decoder
        self.decoder = keras.Sequential([
            # Layer 1: Upsamples 7x7 -> 14x14
            layers.Conv2DTranspose(4, (3, 3), strides=2, padding='same', activation='relu'),
            # Layer 2: Upsamples 14x14 -> 28x28
            layers.Conv2DTranspose(8, (3, 3), strides=2, padding='same', activation='relu'),
            # Output Layer: Reconstructs to 1 channel, sigmoid for [0, 1] range
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate model
cae = ConvAutoencoder()
cae.compile(optimizer='adam', loss='mse')

# --- 3. Training (Standard AE) ---
print("--- Training Standard Autoencoder ---")
# Train on clean data, Target is clean data
history = cae.fit(
    x_train, x_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, x_test),
    shuffle=True,
    verbose=1
)

# --- 4. Anomaly Detection Logic ---

# Compute reconstruction errors on training data
train_recon = cae.predict(x_train)
# MSE per image: mean over height, width, channels
train_errors = np.mean(np.square(x_train - train_recon), axis=(1, 2, 3))

# Define Threshold: Mean + Std
threshold = np.mean(train_errors) + np.std(train_errors)
print(f"Anomaly Threshold: {threshold:.5f}")

# Evaluate on Clean Test Data (Expected: Normal / Low Error)
test_clean_recon = cae.predict(x_test)
test_clean_errors = np.mean(np.square(x_test - test_clean_recon), axis=(1, 2, 3))
# Accuracy: Percentage of clean images labeled as Normal (Error <= Threshold)
acc_clean = np.mean(test_clean_errors <= threshold)

# Evaluate on Noisy Test Data (Expected: Anomaly / High Error)
test_noisy_recon = cae.predict(x_test_noisy)
test_noisy_errors = np.mean(np.square(x_test_noisy - test_noisy_recon), axis=(1, 2, 3))
# Accuracy: Percentage of noisy images labeled as Anomaly (Error > Threshold)
acc_noisy = np.mean(test_noisy_errors > threshold)

print(f"Accuracy on Clean Data (Normal): {acc_clean:.4f}")
print(f"Accuracy on Noisy Data (Anomaly): {acc_noisy:.4f}")

# --- 5. Visualization Function ---

def plot_results(model, clean_images, noisy_images, title="Autoencoder Results"):
    n = 5
    plt.figure(figsize=(10, 8))
    
    # Get reconstructions
    recon_from_clean = model.predict(clean_images[:n])
    recon_from_noisy = model.predict(noisy_images[:n])
    
    for i in range(n):
        # 1. Original
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(clean_images[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis("off")
        
        # 2. Noisy
        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(noisy_images[i].reshape(28, 28), cmap='gray')
        plt.title("Noisy")
        plt.axis("off")
        
        # 3. Recon from Clean
        ax = plt.subplot(4, n, i + 1 + 2*n)
        plt.imshow(recon_from_clean[i].reshape(28, 28), cmap='gray')
        plt.title("Recon(Clean)")
        plt.axis("off")
        
        # 4. Recon from Noisy
        ax = plt.subplot(4, n, i + 1 + 3*n)
        plt.imshow(recon_from_noisy[i].reshape(28, 28), cmap='gray')
        plt.title("Recon(Noisy)")
        plt.axis("off")
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Plot for Standard AE
plot_results(cae, x_test, x_test_noisy, title="Standard AE Results")

# --- 6. Denoising Autoencoder (DAE) ---
print("\n--- Training Denoising Autoencoder ---")

# Re-instantiate model to start fresh
dae = ConvAutoencoder()
dae.compile(optimizer='adam', loss='mse')

# Train on NOISY data, Target is CLEAN data
dae.fit(
    x_train_noisy, x_train, # Input: Noisy, Target: Clean
    epochs=10,
    batch_size=64,
    validation_data=(x_test_noisy, x_test),
    shuffle=True,
    verbose=1
)

# Plot for Denoising AE
plot_results(dae, x_test, x_test_noisy, title="Denoising AE Results")
