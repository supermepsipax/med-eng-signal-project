import numpy as np
import scipy
from scipy import signal


def extract_time_domain_features(epoch):
    """
    This currently extracts 16 time domain features but is signal agnostic.

    Works for any signal type (EEG, EOG, EMG) but students should consider
    signal-specific features for optimal performance.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        dict: A dictionary of features.
    """
    # TODO: Current feature set for time-domain only, need to implement more features
    features = {
        "mean": np.mean(epoch),
        "median": np.median(epoch),
        "std": np.std(epoch),
        "variance": np.var(epoch),
        "rms": np.sqrt(np.mean(epoch**2)),
        "min": np.min(epoch),
        "max": np.max(epoch),
        "range": np.max(epoch) - np.min(epoch),
        "skewness": scipy.stats.skew(epoch),
        "kurtosis": scipy.stats.kurtosis(epoch),
        "zero_crossings": np.sum(np.diff(np.sign(epoch)) != 0),
        "hjorth_activity": np.var(epoch),
        "hjorth_mobility": np.sqrt(np.var(np.diff(epoch)) / np.var(epoch)),
        # This looks horrible but the formula is mobility(y')/mobility(y)
        "hjorth_complexity": (
            np.sqrt(np.var(np.diff(epoch, 2)) / np.var(np.diff(epoch)))
        )
        / (np.sqrt(np.var(np.diff(epoch)) / np.var(epoch))),
        "total_energy": np.sum(epoch**2),
        "mean_power": np.mean(epoch**2),
    }

    return features


def extract_frequency_domain_features(epoch, fs, signal_type="eeg"):
    """
    Extract 15 frequency-domain features from a single epoch.

    Frequency features are critical for sleep staging as different sleep stages
    have characteristic frequency patterns:
    - Delta (0.5-4 Hz): Deep sleep marker
    - Theta (4-8 Hz): Light sleep, drowsiness
    - Alpha (8-13 Hz): Relaxed wakefulness, eyes closed
    - Beta (13-30 Hz): Active thinking, alertness
    - Gamma (30-50 Hz): Cognitive processing

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.
        fs (float): Sampling frequency in Hz.
        signal_type (str): 'eeg' or 'eog' - affects feature interpretation.

    Returns:
        dict: A dictionary of 15 frequency-domain features.
    """
    # Compute power spectral density using Welch's method
    #
    # === WELCH METHOD PARAMETER SELECTION ===
    # (Based on PROJECT_GUIDE.md lines 1650-1684)
    #
    # Window selection: 'hann' (Hanning)
    #   - Standard for sleep EEG analysis
    #   - Good balance: frequency resolution vs. spectral leakage suppression
    #   - Main lobe width: 4 bins, First sidelobe: -32 dB
    #   - Tapers smoothly to zero at edges (reduces edge discontinuities)
    #
    # nperseg=256 samples:
    #   - EEG (125 Hz): 256 samples = 2.048 sec → 0.488 Hz resolution
    #   - EOG (50 Hz): 256 samples = 5.12 sec → 0.195 Hz resolution
    #   - Adequate for delta/theta separation (4 Hz boundary needs ~0.5 Hz resolution)
    #   - Provides ~14 overlapping windows in 30-second epoch → variance reduction
    #
    # noverlap=50% (128 samples):
    #   - Optimal for Hanning window (Welch, 1967)
    #   - Recovers information from window edges (heavily attenuated by taper)
    #   - Balances variance reduction vs. segment independence
    #
    # nfft=nperseg (no zero-padding):
    #   - Zero-padding only interpolates spectrum, doesn't improve resolution
    #   - Not needed for band power integration (wide frequency bands)
    #
    # scaling='density':
    #   - Power spectral density in W/Hz
    #   - Appropriate for integrating over frequency bands
    #
    nperseg = min(256, len(epoch))
    noverlap = nperseg // 2  # 50% overlap (explicit for documentation)

    freqs, psd = signal.welch(
        epoch,
        fs=fs,
        window='hann',      # Hanning window (standard for sleep EEG)
        nperseg=nperseg,    # ~2 sec for EEG → 0.5 Hz resolution
        noverlap=noverlap,  # 50% overlap → optimal variance reduction
        nfft=nperseg,       # No zero-padding (explicit)
        scaling='density'   # Power spectral density (W/Hz)
    )

    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    gamma_band = (30, 50)

    def band_power(freqs, psd, low, high):
        """Calculate power in a frequency band."""
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.trapz(psd[idx], freqs[idx]) if np.sum(idx) > 0 else 0

    delta_power = band_power(freqs, psd, *delta_band)
    theta_power = band_power(freqs, psd, *theta_band)
    alpha_power = band_power(freqs, psd, *alpha_band)
    beta_power = band_power(freqs, psd, *beta_band)
    gamma_power = band_power(freqs, psd, *gamma_band)
    total_power = band_power(freqs, psd, 0.5, 50)

    total_power = total_power if total_power > 0 else 1e-10

    features = {
        "delta_power": delta_power,
        "theta_power": theta_power,
        "alpha_power": alpha_power,
        "beta_power": beta_power,
        "gamma_power": gamma_power,

        "delta_rel": delta_power / total_power,
        "theta_rel": theta_power / total_power,
        "alpha_rel": alpha_power / total_power,
        "beta_rel": beta_power / total_power,
        "gamma_rel": gamma_power / total_power,

        "delta_beta_ratio": delta_power / beta_power if beta_power > 0 else 0,
        "theta_alpha_ratio": theta_power / alpha_power if alpha_power > 0 else 0,
        "spectral_edge_95": (
            freqs[np.where(np.cumsum(psd) >= 0.95 * np.sum(psd))[0][0]]
            if len(freqs) > 0
            else 0
        ),
        "spectral_entropy": -np.sum(
            (psd / np.sum(psd)) * np.log2(psd / np.sum(psd) + 1e-10)
        ),
        "peak_frequency": freqs[np.argmax(psd)] if len(psd) > 0 else 0,
    }

    return features


def extract_features(data, config):
    """
    STUDENT IMPLEMENTATION AREA: Extract features based on current iteration.

    This function should handle both single-channel (old format) and
    multi-channel data (new format with 2 EEG + 2 EOG + 1 EMG channels).

    Iteration 1: 16 time-domain features per EEG channel
    Iteration 2: 31+ features (time + frequency domain) per channel
    Iteration 3: Multi-signal features (EEG + EOG + EMG)
    Iteration 4: Optimized feature set (selected subset)

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        config (module): The configuration module.

    Returns:
        np.ndarray: A 2D array of features (n_epochs, n_features).
    """
    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")

    # Detect if we have multi-channel data structure
    is_multi_channel = isinstance(data, dict) and "eeg" in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return extract_multi_channel_features(data, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return extract_single_channel_features(data, config)


def extract_eeg_features(eeg_signal, fs=125, include_frequency=True):
    """
    Extract 31 features from EEG signal: 16 time-domain + 15 frequency-domain.

    EEG is the primary signal for sleep stage classification:
    - Time features capture signal morphology and statistical properties
    - Frequency features capture brain rhythms critical for sleep staging

    Args:
        eeg_signal (np.ndarray): 1D array of EEG signal data
        fs (float): Sampling frequency (default 125 Hz for EEG)
        include_frequency (bool): Whether to include frequency features

    Returns:
        dict: Combined time and frequency domain features (31 total)
    """
    # Extract 16 time-domain features
    features = extract_time_domain_features(eeg_signal)

    if include_frequency:
        # Add 15 frequency-domain features
        # Brain wave analysis is crucial for sleep staging
        freq_features = extract_frequency_domain_features(
            eeg_signal, fs, signal_type="eeg"
        )
        features.update(freq_features)

    return features


def extract_multi_channel_features(multi_channel_data, config):
    """
    Extract features from multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.

    Iteration 1: 16 time-domain features per EEG channel (backward compatible)
    Iteration 2+: 31 features (16 time + 15 freq) per channel for EEG and EOG
    """
    n_epochs = multi_channel_data["eeg"].shape[0]
    all_features = []

    include_frequency = config.CURRENT_ITERATION >= 2

    for epoch_idx in range(n_epochs):
        epoch_features = []

        eeg_fs = 125  # Default EEG sampling rate
        for ch in range(multi_channel_data["eeg"].shape[1]):
            eeg_signal = multi_channel_data["eeg"][epoch_idx, ch, :]
            eeg_features = extract_eeg_features(
                eeg_signal, fs=eeg_fs, include_frequency=include_frequency
            )
            epoch_features.extend(list(eeg_features.values()))

        if config.CURRENT_ITERATION >= 2:
            if "eog" in multi_channel_data:
                eog_fs = 50  # Default EOG sampling rate
                for ch in range(multi_channel_data["eog"].shape[1]):
                    eog_signal = multi_channel_data["eog"][epoch_idx, ch, :]
                    eog_features = extract_eog_features(
                        eog_signal, fs=eog_fs, include_frequency=include_frequency
                    )
                    epoch_features.extend(list(eog_features.values()))

        if config.CURRENT_ITERATION >= 3:
            if "emg" in multi_channel_data:
                emg_signal = multi_channel_data["emg"][epoch_idx, 0, :]
                emg_features = extract_emg_features(emg_signal)
                epoch_features.extend(list(emg_features.values()))

        all_features.append(epoch_features)

    features = np.array(all_features)

    # Print feature count summary
    if config.CURRENT_ITERATION == 1:
        expected = 2 * 16  # 2 EEG channels × 16 time features
        print(f"Iteration 1: {features.shape[1]} features (expected: {expected})")
        print("  - 2 EEG channels × 16 time-domain features")
    elif config.CURRENT_ITERATION == 2:
        expected = 2 * 31 + 2 * 31  # 2 EEG + 2 EOG × 31 features
        print(f"Iteration 2: {features.shape[1]} features (expected: {expected})")
        print("  - 2 EEG channels × 31 features (16 time + 15 freq)")
        print("  - 2 EOG channels × 31 features (16 time + 15 freq)")
    elif config.CURRENT_ITERATION >= 3:
        print(
            f"Iteration {config.CURRENT_ITERATION}: {features.shape[1]} total features"
        )
        print("  - 2 EEG channels × 31 features (16 time + 15 freq)")
        print("  - 2 EOG channels × 31 features (16 time + 15 freq)")
        print("  - 1 EMG channel × features")

    return features


def extract_single_channel_features(data, config):
    """
    Backward compatibility for single-channel data.
    """
    if config.CURRENT_ITERATION == 1:
        # Iteration 1: Time-domain features (TARGET: 16 features)
        # CURRENT: Only 3 features implemented - students must add 13 more!
        all_features = []
        for epoch in data:
            features = extract_time_domain_features(epoch)
            all_features.append(list(features.values()))
        features = np.array(all_features)

        print(
            f"WARNING: Only {features.shape[1]} features extracted, target is 16 for iteration 1"
        )
        print("Students must implement the remaining time-domain features!")

    elif config.CURRENT_ITERATION == 2:
        print("Target: ~31 features (time + frequency domain)")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  # Empty features - students must implement

    elif config.CURRENT_ITERATION >= 3:
        # TODO: Students must implement multi-signal features
        print("TODO: Students should use multi-channel data format for iteration 3+")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  # Empty features - students must implement

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return features


def extract_eog_features(eog_signal, fs=50, include_frequency=True):
    """
    Extract 31 features from EOG signal: 16 time-domain + 15 frequency-domain.

    EOG signals are used to detect:
    - Rapid eye movements (REM sleep indicator)
    - Slow eye movements (NREM sleep)
    - Eye blinks and artifacts

    Args:
        eog_signal (np.ndarray): 1D array of EOG signal data
        fs (float): Sampling frequency (default 50 Hz for EOG)
        include_frequency (bool): Whether to include frequency features

    Returns:
        dict: Combined time and frequency domain features
    """
    # Use the same 16 time-domain features as EEG
    # These are signal-agnostic and work well for EOG too
    features = extract_time_domain_features(eog_signal)

    if include_frequency:
        # Add 15 frequency-domain features
        # EOG frequency features help distinguish REM (rapid movements) from NREM
        freq_features = extract_frequency_domain_features(
            eog_signal, fs, signal_type="eog"
        )
        features.update(freq_features)

    # TODO: Students could add EOG-specific features:
    # - Eye movement detection (peak counting, saccade detection)
    # - Rapid vs slow movement discrimination (velocity metrics)
    # - Cross-channel correlations (left vs right eye for conjugate movements)

    return features


def extract_emg_features(emg_signal):
    """
    STUDENT TODO: Extract EMG-specific features for muscle tone detection.

    EMG signals are used to detect:
    - Muscle tone levels (high in wake, low in REM)
    - Muscle twitches and artifacts
    - Sleep-related muscle activity
    """
    features = {
        "emg_mean": np.mean(emg_signal),
        "emg_std": np.std(emg_signal),
        "emg_rms": np.sqrt(np.mean(emg_signal**2)),
    }

    # TODO: Students should add:
    # - High-frequency power (muscle activity indicator)
    # - Spectral edge frequency
    # - Muscle tone quantification

    return features
