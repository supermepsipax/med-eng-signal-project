"""
Signal Artifact Visualization Tool

This script loads random chunks of raw EEG/EOG/EMG data and visualizes them
to help identify powerline noise, artifacts, and other signal quality issues.

Usage:
    python visualize_signal_artifacts.py

Features:
- Displays raw and filtered signals side-by-side
- Shows frequency spectrum to identify powerline noise (50/60 Hz)
- Randomly samples different epochs from the dataset
- Supports all channel types (EEG, EOG, EMG)
"""

import matplotlib
# Try to set an interactive backend
# Try multiple backends in order of preference
backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg']
backend_set = False

for backend in backends:
    try:
        matplotlib.use(backend)
        backend_set = True
        print(f"Using matplotlib backend: {backend}")
        break
    except:
        continue

if not backend_set:
    print("Warning: No interactive backend found. Plots may not display properly.")
    print("Consider installing: sudo apt-get install python3-tk")

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
from glob import glob
import config
from src.data_loader import load_training_data


def compute_spectrum(data, fs):
    """
    Compute frequency spectrum using FFT.

    Args:
        data (np.ndarray): Time-domain signal
        fs (float): Sampling frequency in Hz

    Returns:
        freqs (np.ndarray): Frequency bins
        magnitude (np.ndarray): Magnitude spectrum
    """
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1/fs)[:N//2]
    magnitude = 2.0/N * np.abs(yf[0:N//2])
    return xf, magnitude


def apply_basic_filter(data, cutoff, fs, order=5):
    """Apply a basic low-pass filter for comparison."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered = signal.lfilter(b, a, data)
    return filtered


def visualize_signal_chunk(chunk_data, fs, channel_name, epoch_idx,
                           cutoff_freq=40, show_seconds=30):
    """
    Visualize a chunk of signal data with time domain, filtered version,
    and frequency spectrum.

    Args:
        chunk_data (np.ndarray): 1D signal data
        fs (float): Sampling frequency in Hz
        channel_name (str): Name of the channel
        epoch_idx (int): Epoch index
        cutoff_freq (float): Low-pass filter cutoff frequency
        show_seconds (float): Number of seconds to display (if less than full epoch)
    """
    # Limit display to first N seconds if specified
    max_samples = int(show_seconds * fs)
    if len(chunk_data) > max_samples:
        display_data = chunk_data[:max_samples]
    else:
        display_data = chunk_data

    # Create time array
    time = np.arange(len(display_data)) / fs

    # Apply basic filter
    filtered_data = apply_basic_filter(display_data, cutoff_freq, fs)

    # Compute frequency spectrum (use full chunk for better frequency resolution)
    freqs, magnitude = compute_spectrum(chunk_data, fs)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'{channel_name} - Epoch {epoch_idx} (First {show_seconds}s shown)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Raw signal
    axes[0].plot(time, display_data, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Amplitude (µV)', fontsize=10)
    axes[0].set_title('Raw Signal', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([time[0], time[-1]])

    # Plot 2: Filtered signal
    axes[1].plot(time, filtered_data, 'g-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Amplitude (µV)', fontsize=10)
    axes[1].set_xlabel('Time (seconds)', fontsize=10)
    axes[1].set_title(f'Filtered Signal (Low-pass {cutoff_freq} Hz)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([time[0], time[-1]])

    # Plot 3: Frequency spectrum (0-100 Hz range for visibility)
    freq_limit = min(100, fs/2)
    freq_mask = freqs <= freq_limit
    axes[2].plot(freqs[freq_mask], magnitude[freq_mask], 'r-', linewidth=1)
    axes[2].set_ylabel('Magnitude', fontsize=10)
    axes[2].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[2].set_title('Frequency Spectrum (0-100 Hz)', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Highlight powerline frequencies
    axes[2].axvline(50, color='orange', linestyle='--', alpha=0.5, label='50 Hz (EU)')
    axes[2].axvline(60, color='purple', linestyle='--', alpha=0.5, label='60 Hz (US)')
    axes[2].legend(loc='upper right', fontsize=8)

    # Find and annotate peaks in spectrum
    peak_threshold = np.max(magnitude[freq_mask]) * 0.1  # 10% of max
    peaks, _ = signal.find_peaks(magnitude[freq_mask], height=peak_threshold)
    if len(peaks) > 0:
        peak_freqs = freqs[freq_mask][peaks]
        peak_mags = magnitude[freq_mask][peaks]
        # Show top 5 peaks
        top_peaks_idx = np.argsort(peak_mags)[-5:]
        for idx in top_peaks_idx:
            axes[2].plot(peak_freqs[idx], peak_mags[idx], 'ro', markersize=6)
            axes[2].annotate(f'{peak_freqs[idx]:.1f} Hz',
                           xy=(peak_freqs[idx], peak_mags[idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='red')

    plt.tight_layout()
    return fig


def load_and_visualize_random_epochs(n_samples=5, data_dir=None, show_seconds=10):
    """
    Load training data and visualize random epochs from each channel type.

    Args:
        n_samples (int): Number of random epochs to visualize
        data_dir (str): Path to training data directory (default: config.TRAINING_DIR)
        show_seconds (float): Number of seconds to display per epoch
    """
    if data_dir is None:
        data_dir = config.TRAINING_DIR

    # Find all EDF files
    edf_files = sorted(glob(os.path.join(data_dir, "*.edf")))

    if len(edf_files) == 0:
        raise FileNotFoundError(f"No .edf files found in {data_dir}")

    print(f"Found {len(edf_files)} EDF files")
    print(f"Visualizing {n_samples} random epochs from each channel type...\n")

    # Load first file to get data structure
    edf_file = edf_files[0]
    base_name = os.path.splitext(os.path.basename(edf_file))[0]
    xml_file = os.path.join(data_dir, f"{base_name}.xml")

    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"No annotation file found for {base_name}.edf")

    print(f"Loading {base_name}...\n")
    multi_channel_data, labels, channel_info = load_training_data(edf_file, xml_file)

    # Get available channel types
    channel_types = list(multi_channel_data.keys())
    print(f"Available channel types: {channel_types}\n")

    # Visualize random epochs for each channel type
    for channel_type in channel_types:
        data = multi_channel_data[channel_type]
        fs_key = f'{channel_type}_fs'
        names_key = f'{channel_type}_names'

        fs = channel_info.get(fs_key, 125)  # Default to 125 Hz
        channel_names = channel_info.get(names_key, [f'{channel_type.upper()}_ch{i}'
                                                      for i in range(data.shape[1])])

        n_epochs, n_channels, n_samples_per_epoch = data.shape

        print(f"\n{'='*60}")
        print(f"{channel_type.upper()} Channels")
        print(f"{'='*60}")
        print(f"Shape: {data.shape}")
        print(f"Sampling rate: {fs} Hz")
        print(f"Channels: {channel_names}")
        print(f"Samples per epoch: {n_samples_per_epoch} ({n_samples_per_epoch/fs:.1f} seconds)")

        # For each channel in this type
        for ch_idx, ch_name in enumerate(channel_names):
            print(f"\n--- Visualizing {ch_name} ---")

            # Select random epochs
            random_epochs = np.random.choice(n_epochs, size=min(n_samples, n_epochs),
                                            replace=False)

            for epoch_idx in random_epochs:
                epoch_data = data[epoch_idx, ch_idx, :]

                # Get label name
                label = labels[epoch_idx]
                label_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
                label_name = label_names[label] if label < len(label_names) else f'Unknown({label})'

                print(f"  Epoch {epoch_idx}: Stage={label_name}, "
                      f"Mean={np.mean(epoch_data):.2f}, Std={np.std(epoch_data):.2f}, "
                      f"Range=[{np.min(epoch_data):.2f}, {np.max(epoch_data):.2f}]")

                # Determine appropriate cutoff based on channel type
                if channel_type == 'eeg':
                    cutoff = 40  # Standard EEG cutoff
                elif channel_type == 'eog':
                    cutoff = 30  # Lower for EOG (preserve slow eye movements)
                elif channel_type == 'emg':
                    cutoff = 70  # Higher for EMG (preserve muscle activity)
                else:
                    cutoff = 40  # Default

                # Create visualization
                fig = visualize_signal_chunk(
                    epoch_data,
                    fs,
                    f"{ch_name} [{label_name}]",
                    epoch_idx,
                    cutoff_freq=cutoff,
                    show_seconds=show_seconds
                )

                plt.show(block=False)

                # Wait for user input to continue
                user_input = input("\n  Press Enter for next epoch, 'q' to skip remaining epochs, 'x' to exit: ")
                if user_input.lower() == 'q':
                    plt.close('all')
                    break
                elif user_input.lower() == 'x':
                    plt.close('all')
                    print("\nExiting visualization...")
                    return
                else:
                    plt.close(fig)

            # Ask if user wants to continue to next channel
            if ch_idx < len(channel_names) - 1:
                user_input = input(f"\n  Continue to next channel? (Enter=yes, 'q'=skip to next channel type, 'x'=exit): ")
                if user_input.lower() == 'q':
                    plt.close('all')
                    break
                elif user_input.lower() == 'x':
                    plt.close('all')
                    print("\nExiting visualization...")
                    return

    plt.close('all')
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


def analyze_powerline_noise(multi_channel_data, channel_info,
                           powerline_freq=60, tolerance=2):
    """
    Analyze the presence of powerline noise across all channels.

    Args:
        multi_channel_data (dict): Multi-channel data dictionary
        channel_info (dict): Channel information dictionary
        powerline_freq (float): Powerline frequency to check (50 or 60 Hz)
        tolerance (float): Frequency tolerance in Hz

    Returns:
        dict: Summary of powerline noise analysis per channel
    """
    print(f"\n{'='*60}")
    print(f"Powerline Noise Analysis ({powerline_freq} Hz ± {tolerance} Hz)")
    print(f"{'='*60}\n")

    noise_summary = {}

    for channel_type in multi_channel_data.keys():
        data = multi_channel_data[channel_type]
        fs_key = f'{channel_type}_fs'
        names_key = f'{channel_type}_names'

        fs = channel_info.get(fs_key, 125)
        channel_names = channel_info.get(names_key, [f'{channel_type.upper()}_ch{i}'
                                                      for i in range(data.shape[1])])

        print(f"{channel_type.upper()} Channels:")

        for ch_idx, ch_name in enumerate(channel_names):
            # Analyze multiple random epochs
            n_epochs = data.shape[0]
            sample_size = min(20, n_epochs)  # Analyze 20 random epochs
            random_epochs = np.random.choice(n_epochs, size=sample_size, replace=False)

            powerline_magnitudes = []

            for epoch_idx in random_epochs:
                epoch_data = data[epoch_idx, ch_idx, :]
                freqs, magnitude = compute_spectrum(epoch_data, fs)

                # Find magnitude at powerline frequency
                freq_mask = (freqs >= powerline_freq - tolerance) & (freqs <= powerline_freq + tolerance)
                if np.any(freq_mask):
                    powerline_mag = np.max(magnitude[freq_mask])
                    powerline_magnitudes.append(powerline_mag)

            avg_powerline = np.mean(powerline_magnitudes)
            max_powerline = np.max(powerline_magnitudes)

            # Calculate average magnitude in other frequency ranges for comparison
            baseline_mags = []
            for epoch_idx in random_epochs:
                epoch_data = data[epoch_idx, ch_idx, :]
                freqs, magnitude = compute_spectrum(epoch_data, fs)

                # Sample baseline from 5-40 Hz (avoiding powerline)
                baseline_mask = ((freqs >= 5) & (freqs <= 40) &
                               ((freqs < powerline_freq - 5) | (freqs > powerline_freq + 5)))
                if np.any(baseline_mask):
                    baseline_mags.append(np.mean(magnitude[baseline_mask]))

            avg_baseline = np.mean(baseline_mags)

            # Calculate signal-to-noise ratio
            snr = avg_powerline / avg_baseline if avg_baseline > 0 else 0

            noise_summary[ch_name] = {
                'avg_powerline_magnitude': avg_powerline,
                'max_powerline_magnitude': max_powerline,
                'avg_baseline_magnitude': avg_baseline,
                'powerline_to_baseline_ratio': snr
            }

            # Determine if powerline noise is significant
            if snr > 3:
                status = "⚠️  SIGNIFICANT powerline noise detected"
            elif snr > 1.5:
                status = "⚡ Moderate powerline noise detected"
            else:
                status = "✓  Low powerline noise"

            print(f"  {ch_name}: {status}")
            print(f"    Avg {powerline_freq}Hz magnitude: {avg_powerline:.6e}")
            print(f"    Max {powerline_freq}Hz magnitude: {max_powerline:.6e}")
            print(f"    Baseline magnitude (5-40Hz): {avg_baseline:.6e}")
            print(f"    Ratio: {snr:.2f}x")

        print()

    return noise_summary


if __name__ == '__main__':
    import sys

    # Parse command line arguments
    n_samples = 3  # Default: show 3 random epochs per channel
    show_seconds = 10  # Default: show first 10 seconds

    if len(sys.argv) > 1:
        try:
            n_samples = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid n_samples '{sys.argv[1]}', using default {n_samples}")

    if len(sys.argv) > 2:
        try:
            show_seconds = float(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid show_seconds '{sys.argv[2]}', using default {show_seconds}")

    print("="*60)
    print("Signal Artifact Visualization Tool")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Random epochs per channel: {n_samples}")
    print(f"  - Display duration: {show_seconds} seconds")
    print(f"  - Data directory: {config.TRAINING_DIR}")
    print("="*60)

    # Load and visualize
    try:
        # First, run powerline noise analysis
        edf_files = sorted(glob(os.path.join(config.TRAINING_DIR, "*.edf")))
        if len(edf_files) > 0:
            edf_file = edf_files[0]
            base_name = os.path.splitext(os.path.basename(edf_file))[0]
            xml_file = os.path.join(config.TRAINING_DIR, f"{base_name}.xml")

            if os.path.exists(xml_file):
                print(f"\nLoading data for powerline analysis from {base_name}...\n")
                multi_channel_data, labels, channel_info = load_training_data(edf_file, xml_file)

                # Analyze for both 50Hz and 60Hz
                print("\nChecking for 60 Hz powerline noise (US)...")
                noise_60hz = analyze_powerline_noise(multi_channel_data, channel_info,
                                                     powerline_freq=60, tolerance=2)

                print("\nChecking for 50 Hz powerline noise (EU)...")
                noise_50hz = analyze_powerline_noise(multi_channel_data, channel_info,
                                                     powerline_freq=50, tolerance=2)

        # Then run interactive visualization
        print("\n" + "="*60)
        print("Starting interactive visualization...")
        print("="*60)
        load_and_visualize_random_epochs(n_samples=n_samples, show_seconds=show_seconds)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
