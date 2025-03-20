import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import sionna as sn

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision training for performance (optional, requires GPU)
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# 1. Generate synthetic CSI data using Sionna (3GPP TR 38.901)
def generate_csi_data(num_samples=1000, num_antennas=64, snr_db=20):
    """
    Generate realistic CSI data using Sionna's 3GPP TR 38.901 channel model.
    
    Args:
        num_samples (int): Number of CSI samples to generate.
        num_antennas (int): Number of antennas (for MIMO).
        snr_db (float): Signal-to-Noise Ratio in dB.
    
    Returns:
        np.ndarray: Flattened CSI data (real and imaginary parts concatenated).
    """
    # Configure Urban Micro (UMi) scenario for 5G mmWave
    model = sn.channel.tr38901.UMi(
        carrier_frequency=28e9,  # 28 GHz (mmWave)
        o2i_model="low",        # Outdoor-to-indoor penetration model
        ut_array=sn.channel.AntennaArray(num_rows=1, num_cols=1),  # User equipment
        bs_array=sn.channel.AntennaArray(num_rows=8, num_cols=8),  # Base station (64 antennas)
        direction="uplink",
        topology="random",
    )
    
    # Generate channel impulse responses
    cirs = model(num_samples, num_antennas)
    
    # Convert to frequency domain (CSI) with 64 subcarriers
    csi = sn.channel.cir_to_ofdm_channel(cirs, num_subcarriers=64)
    
    # Add AWGN noise based on SNR
    noise_power = 10 ** (-snr_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power / 2), csi.shape) + \
            1j * np.random.normal(0, np.sqrt(noise_power / 2), csi.shape)
    csi_noisy = csi + noise
    
    # Flatten and concatenate real and imaginary parts
    csi_flat = np.concatenate([csi_noisy.real, csi_noisy.imag], axis=-1)
    return csi_flat.astype(np.float32)

# 2. Build Convolutional Autoencoder (CAE)
def build_cae(input_shape=(64, 2), encoding_dim=32):
    """
    Build a convolutional autoencoder for CSI compression.
    
    Args:
        input_shape (tuple): Shape of CSI data (subcarriers, real/imag channels).
        encoding_dim (int): Size of the compressed representation.
    
    Returns:
        tf.keras.Model: Compiled autoencoder model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder: Extract features with convolutions
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder: Reconstruct using transposed convolutions
    x = layers.Dense(16 * 64, activation='relu')(encoded)
    x = layers.Reshape((16, 64))(x)
    x = layers.Conv1DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1DTranspose(2, 3, activation='linear', padding='same')(x)
    
    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# 3. Train and Evaluate the Model
def train_and_evaluate(csi_data, autoencoder, epochs=50, batch_size=32):
    """
    Train the autoencoder and evaluate its performance.
    
    Args:
        csi_data (np.ndarray): Input CSI data.
        autoencoder (tf.keras.Model): The autoencoder model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        tuple: Test data, reconstructed data, and training history.
    """
    # Reshape for convolutional input: (samples, subcarriers, real/imag)
    csi_data = csi_data.reshape(-1, 64, 2)
    train_size = int(0.8 * len(csi_data))
    x_train, x_test = csi_data[:train_size], csi_data[train_size:]
    
    # Efficient data pipeline with tf.data
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).shuffle(1000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(batch_size)
    
    # Train the model
    history = autoencoder.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=1)
    
    # Evaluate reconstruction error
    reconstructed = autoencoder.predict(x_test)
    mse = np.mean((x_test - reconstructed) ** 2)
    print(f"Mean Squared Error on Test Data: {mse:.6f}")
    return x_test, reconstructed, history

# 4. Benchmark with Uniform Quantization
def uniform_quantization(csi_data, bits=8):
    """
    Apply uniform quantization as a baseline compression method.
    
    Args:
        csi_data (np.ndarray): Input CSI data.
        bits (int): Number of bits for quantization.
    
    Returns:
        float: Mean Squared Error of the quantized reconstruction.
    """
    min_val = np.min(csi_data)
    max_val = np.max(csi_data)
    levels = 2 ** bits
    quantized = np.round((csi_data - min_val) / (max_val - min_val) * (levels - 1))
    reconstructed = quantized / (levels - 1) * (max_val - min_val) + min_val
    mse = np.mean((csi_data - reconstructed) ** 2)
    print(f"MSE with {bits}-bit Uniform Quantization: {mse:.6f}")
    return mse

# 5. Robustness Testing Across SNR Levels
def test_robustness(autoencoder, num_samples=200, snr_levels=[0, 10, 20, 30]):
    """
    Test the autoencoder's performance across different SNR levels.
    
    Args:
        autoencoder (tf.keras.Model): Trained autoencoder model.
        num_samples (int): Number of samples per SNR level.
        snr_levels (list): List of SNR values (in dB) to test.
    """
    mse_results = []
    for snr in snr_levels:
        csi_data = generate_csi_data(num_samples=num_samples, snr_db=snr)
        csi_data = csi_data.reshape(-1, 64, 2)
        reconstructed = autoencoder.predict(csi_data, verbose=0)
        mse = np.mean((csi_data - reconstructed) ** 2)
        mse_results.append(mse)
        print(f"SNR {snr} dB - MSE: {mse:.6f}")
    
    # Plot robustness results
    plt.figure(figsize=(8, 4))
    plt.plot(snr_levels, mse_results, marker='o')
    plt.title('Autoencoder Robustness Across SNR Levels')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('demo_plots/robustness.png')
    plt.close()

# 6. Visualize Results
def plot_results(original, reconstructed, history):
    """
    Visualize training loss and CSI reconstruction.
    
    Args:
        original (np.ndarray): Original test CSI data.
        reconstructed (np.ndarray): Reconstructed CSI data.
        history (tf.keras.callbacks.History): Training history.
    """
    # Training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('demo_plots/loss_curve.png')
    plt.close()

    # Original vs reconstructed CSI (real part, sample 0)
    plt.figure(figsize=(12, 4))
    plt.plot(original[0, :, 0], label='Original Real', alpha=0.7)
    plt.plot(reconstructed[0, :, 0], label='Reconstructed Real', alpha=0.7)
    plt.title('Original vs Reconstructed CSI (Real Part)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('demo_plots/csi_comparison_real.png')
    plt.close()

# Main Execution
if __name__ == "__main__":
    # Create output directory for plots
    import os
    os.makedirs('demo_plots', exist_ok=True)
    
    # Generate CSI data with realistic channel model
    print("Generating CSI data...")
    csi_data = generate_csi_data(num_samples=1000, num_antennas=64, snr_db=20)
    
    # Build and train the convolutional autoencoder
    print("Training the autoencoder...")
    autoencoder = build_cae(input_shape=(64, 2), encoding_dim=32)
    x_test, reconstructed, history = train_and_evaluate(csi_data, autoencoder, epochs=50, batch_size=32)
    
    # Benchmark against uniform quantization
    print("\nBenchmarking with uniform quantization...")
    quant_mse = uniform_quantization(csi_data, bits=8)
    
    # Test robustness across SNR levels
    print("\nTesting robustness across SNR levels...")
    test_robustness(autoencoder, num_samples=200, snr_levels=[0, 10, 20, 30])
    
    # Visualize results
    print("Generating plots...")
    plot_results(x_test, reconstructed, history)
    
    # Calculate compression ratio and energy savings
    original_size = 64 * 2  # 64 subcarriers, real + imaginary
    compressed_size = 32    # Encoding dimension
    compression_ratio = original_size / compressed_size
    energy_savings = (1 - 1 / compression_ratio) * 100
    print(f"\nCompression Ratio: {compression_ratio:.1f}x")
    print(f"Estimated Energy Savings: {energy_savings:.1f}%")