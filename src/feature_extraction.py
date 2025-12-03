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
    # Time-domain features optimized for robustness across recordings
    # Using normalized/relative features instead of absolute power values
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
        # REMOVED: total_energy and mean_power (too large, causes scaling issues)
        # These absolute power features vary wildly across recordings
        # Use RMS and relative band powers instead
    }

    return features


def extract_frequency_domain_features(epoch, fs, signal_type="eeg"):
    """
    Extract 10 frequency-domain features from a single epoch (optimized for robustness).

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
        dict: A dictionary of 10 frequency-domain features (all normalized/relative).
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

    # Frequency features optimized for robustness across recordings
    # Using ONLY relative/normalized features - no absolute powers
    # Absolute band powers vary wildly across recordings (1e9 - 1e11 range!)
    features = {
        # REMOVED: Absolute band powers (delta_power, theta_power, etc.)
        # These cause extreme feature values that break StandardScaler

        # Relative band powers (normalized by total power) - ROBUST
        "delta_rel": delta_power / total_power,
        "theta_rel": theta_power / total_power,
        "alpha_rel": alpha_power / total_power,
        "beta_rel": beta_power / total_power,
        "gamma_rel": gamma_power / total_power,

        # Band power ratios - ROBUST
        "delta_beta_ratio": delta_power / beta_power if beta_power > 0 else 0,
        "theta_alpha_ratio": theta_power / alpha_power if alpha_power > 0 else 0,

        # N1-SPECIFIC: Theta/(Alpha+Beta) ratio - drowsiness marker
        # In N1: theta increases, alpha+beta decrease
        "theta_alertness_ratio": theta_power / (alpha_power + beta_power) if (alpha_power + beta_power) > 0 else 0,

        # N1-SPECIFIC: (Theta+Alpha)/(Delta+Beta) - transition indicator
        # N1 has mixed frequency, not dominated by any single band
        "transition_ratio": (theta_power + alpha_power) / (delta_power + beta_power) if (delta_power + beta_power) > 0 else 0,

        # Spectral shape features - ROBUST
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


def extract_features(data, config, channel_info=None):
    """
    Extract features from multi-channel data based on current iteration.

    Iteration 1: 14 time-domain features per EEG channel
    Iteration 2: 26 features (14 time + 12 freq) per EEG channel, 8 features per EOG channel
    Iteration 3: Multi-signal features (EEG + EOG + EMG)
    Iteration 4: Optimized feature set (selected subset)

    Args:
        data (dict): Multi-channel data with keys 'eeg', 'eog', 'emg'
        config (module): The configuration module.
        channel_info (dict): Channel metadata including sampling rates

    Returns:
        tuple: (features, feature_names) where:
            - features: A 2D array of features (n_epochs, n_features)
            - feature_names: List of feature names corresponding to columns
    """
    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")
    print("Processing multi-channel data (EEG + EOG + EMG)")
    return extract_multi_channel_features(data, config, channel_info)


def extract_eeg_features(eeg_signal, fs=125, include_frequency=True):
    """
    Extract 24 features from EEG signal: 14 time-domain + 10 frequency-domain.

    EEG is the primary signal for sleep stage classification:
    - Time features capture signal morphology and statistical properties
    - Frequency features capture brain rhythms critical for sleep staging

    Args:
        eeg_signal (np.ndarray): 1D array of EEG signal data
        fs (float): Sampling frequency (default 125 Hz for EEG)
        include_frequency (bool): Whether to include frequency features

    Returns:
        dict: Combined time and frequency domain features (24 total, all normalized)
    """
    # Extract 14 time-domain features
    features = extract_time_domain_features(eeg_signal)

    if include_frequency:
        # Add 10 frequency-domain features (all normalized/relative)
        # Brain wave analysis is crucial for sleep staging
        freq_features = extract_frequency_domain_features(
            eeg_signal, fs, signal_type="eeg"
        )
        features.update(freq_features)

    return features


def extract_multi_channel_features(multi_channel_data, config, channel_info=None):
    """
    Extract features from multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.

    Iteration 1: 14 time-domain features per EEG channel (backward compatible)
    Iteration 2+: 24 features (14 time + 10 freq) per channel for EEG and EOG
    Iteration 3+: Add EMG features

    Args:
        multi_channel_data (dict): Dictionary with 'eeg', 'eog', 'emg' keys
        config (module): Configuration module
        channel_info (dict): Channel metadata including sampling rates

    Returns:
        tuple: (features, feature_names) where:
            - features: A 2D array of features (n_epochs, n_features)
            - feature_names: List of feature names corresponding to columns
    """
    n_epochs = multi_channel_data["eeg"].shape[0]
    all_features = []
    feature_names = []

    include_frequency = config.CURRENT_ITERATION >= 2

    # Get sampling rates from channel_info, with fallback defaults
    eeg_fs = channel_info.get('eeg_fs', 125) if channel_info else 125
    eog_fs = channel_info.get('eog_fs', 50) if channel_info else 50
    emg_fs = channel_info.get('emg_fs', 125) if channel_info else 125

    # Get actual channel names if available
    eeg_channel_names = channel_info.get('eeg_channels', []) if channel_info else []
    eog_channel_names = channel_info.get('eog_channels', []) if channel_info else []
    emg_channel_names = channel_info.get('emg_channels', []) if channel_info else []

    # Build feature names (only needs to be done once)
    # Extract EEG features (always present)
    for ch in range(multi_channel_data["eeg"].shape[1]):
        # Get channel name or use default
        ch_name = eeg_channel_names[ch] if ch < len(eeg_channel_names) else f"ch{ch}"
        eeg_signal = multi_channel_data["eeg"][0, ch, :]
        eeg_features = extract_eeg_features(
            eeg_signal, fs=eeg_fs, include_frequency=include_frequency
        )
        for feature_name in eeg_features.keys():
            feature_names.append(f"eeg_{ch_name}_{feature_name}")

    # Extract EOG features (Iteration 2+)
    if config.CURRENT_ITERATION >= 2:
        if "eog" in multi_channel_data:
            for ch in range(multi_channel_data["eog"].shape[1]):
                ch_name = eog_channel_names[ch] if ch < len(eog_channel_names) else f"ch{ch}"
                eog_signal = multi_channel_data["eog"][0, ch, :]
                eog_features = extract_eog_features(
                    eog_signal, fs=eog_fs, include_frequency=include_frequency
                )
                for feature_name in eog_features.keys():
                    feature_names.append(f"eog_{ch_name}_{feature_name}")

    # Extract EMG features (Iteration 3+)
    if config.CURRENT_ITERATION >= 3:
        if "emg" in multi_channel_data:
            ch_name = emg_channel_names[0] if len(emg_channel_names) > 0 else "ch0"
            emg_signal = multi_channel_data["emg"][0, 0, :]
            emg_features = extract_emg_features(emg_signal, fs=emg_fs)
            for feature_name in emg_features.keys():
                feature_names.append(f"emg_{ch_name}_{feature_name}")

    # Now extract features for all epochs
    for epoch_idx in range(n_epochs):
        epoch_features = []

        # Extract EEG features (always present)
        for ch in range(multi_channel_data["eeg"].shape[1]):
            eeg_signal = multi_channel_data["eeg"][epoch_idx, ch, :]
            eeg_features = extract_eeg_features(
                eeg_signal, fs=eeg_fs, include_frequency=include_frequency
            )
            epoch_features.extend(list(eeg_features.values()))

        # Extract EOG features (Iteration 2+)
        if config.CURRENT_ITERATION >= 2:
            if "eog" in multi_channel_data:
                for ch in range(multi_channel_data["eog"].shape[1]):
                    eog_signal = multi_channel_data["eog"][epoch_idx, ch, :]
                    eog_features = extract_eog_features(
                        eog_signal, fs=eog_fs, include_frequency=include_frequency
                    )
                    epoch_features.extend(list(eog_features.values()))

        # Extract EMG features (Iteration 3+)
        if config.CURRENT_ITERATION >= 3:
            if "emg" in multi_channel_data:
                emg_signal = multi_channel_data["emg"][epoch_idx, 0, :]
                emg_features = extract_emg_features(emg_signal, fs=emg_fs)
                epoch_features.extend(list(emg_features.values()))

        all_features.append(epoch_features)

    features = np.array(all_features)

    # Print feature count summary
    if config.CURRENT_ITERATION == 1:
        expected = 2 * 14  # 2 EEG channels × 14 time features
        print(f"Iteration 1: {features.shape[1]} features (expected: {expected})")
        print("  - 2 EEG channels × 14 time-domain features")
    elif config.CURRENT_ITERATION == 2:
        expected = 2 * 24 + 2 * 6  # 2 EEG × 24 + 2 EOG × 6
        print(f"Iteration 2: {features.shape[1]} features (expected: {expected})")
        print("  - 2 EEG channels × 24 features (14 time + 10 freq, all normalized)")
        print("  - 2 EOG channels × 6 eye-movement-specific features")
    elif config.CURRENT_ITERATION >= 3:
        expected = 2 * 26 + 2 * 11 + 1 * 7  # 2 EEG × 26 + 2 EOG × 11 + 1 EMG × 7
        print(
            f"Iteration {config.CURRENT_ITERATION}: {features.shape[1]} features (expected: {expected})"
        )
        print("  - 2 EEG channels × 26 features (14 time + 12 freq with N1-specific ratios)")
        print("  - 2 EOG channels × 11 features (SEM detection + REM burst patterns)")
        print("  - 1 EMG channel × 7 features (muscle tone + atonia markers)")

    return features, feature_names


def extract_eog_features(eog_signal, fs=50, include_frequency=False):
    """
    Extract 11 EOG-specific features focused on eye movement detection.

    EOG signals are used to detect:
    - Slow eye movements (N1 sleep indicator - 0.25-0.5 Hz)
    - Rapid eye movements (REM sleep indicator - 1-10 Hz)
    - REM burst patterns (Iteration 4 - temporal structure)
    - Eye blinks and artifacts

    Args:
        eog_signal (np.ndarray): 1D array of EOG signal data
        fs (float): Sampling frequency (default 50 Hz for EOG)
        include_frequency (bool): Kept for API compatibility (ignored for EOG)

    Returns:
        dict: 11 EOG-specific features (8 from Iter 3 + 3 burst features from Iter 4)
    """
    # 1. Peak amplitude (max absolute value) - eye movement magnitude
    peak_amplitude = np.max(np.abs(eog_signal))

    # 2. Variance - signal variability indicator
    variance = np.var(eog_signal)

    # 3. RMS - signal power (normalized measure)
    rms = np.sqrt(np.mean(eog_signal**2))

    # 4. N1-SPECIFIC: Slow Eye Movement (SEM) detection score
    # Bandpass filter 0.25-0.5 Hz to isolate slow rolling eye movements
    # These are THE KEY MARKER for N1 sleep!
    nyquist = fs / 2
    if 0.5 < nyquist and 0.25 < nyquist:
        b, a = signal.butter(3, [0.25 / nyquist, 0.5 / nyquist], btype='band')
        sem_filtered = signal.filtfilt(b, a, eog_signal)

        # Count slow movements
        sem_threshold = 0.3 * np.std(sem_filtered)
        sem_peaks, _ = signal.find_peaks(np.abs(sem_filtered), height=sem_threshold, distance=int(fs))
        sem_score = len(sem_peaks)
    else:
        sem_score = 0

    # 5. REM detection score - count rapid deflections (1-10 Hz)
    # High-pass filter >1 Hz to isolate rapid movements
    if 1.0 < nyquist:
        b, a = signal.butter(3, 1.0 / nyquist, btype='high')
        rem_filtered = signal.filtfilt(b, a, eog_signal)

        # Count rapid movements
        rem_threshold = 0.5 * np.std(rem_filtered)
        rem_peaks, _ = signal.find_peaks(np.abs(rem_filtered), height=rem_threshold)
        rem_score = len(rem_peaks)
    else:
        rem_score = 0

    # 6. Zero-crossing rate - frequency of signal changes
    zero_crossings = np.sum(np.diff(np.sign(eog_signal)) != 0)

    # 7. Mean absolute value - overall activity level
    mean_abs_value = np.mean(np.abs(eog_signal))

    # 8. N1-SPECIFIC: SEM/REM ratio - distinguishes N1 from REM sleep
    # N1: high SEM, low REM
    # REM: low SEM, high REM
    sem_rem_ratio = sem_score / (rem_score + 1e-10)

    # 9-11. REM-SPECIFIC: Burst detection features (Iteration 4)
    # REM has characteristic bursts of eye movements, not continuous activity
    # These temporal features distinguish REM from other stages
    burst_count = 0
    burst_density = 0.0
    mean_burst_duration = 0.0

    if rem_score > 0 and 1.0 < nyquist:
        # Use REM-filtered signal and detected peaks
        # Cluster REM events that occur within 0.5 seconds to define bursts
        if len(rem_peaks) > 0:
            # Convert peak indices to time (seconds)
            peak_times = rem_peaks / fs

            # Find clusters of peaks (bursts) - peaks within 0.5s are in same burst
            burst_threshold = 0.5  # seconds
            bursts = []
            current_burst_start = peak_times[0]
            current_burst_end = peak_times[0]

            for i in range(1, len(peak_times)):
                if peak_times[i] - current_burst_end <= burst_threshold:
                    # Extend current burst
                    current_burst_end = peak_times[i]
                else:
                    # End current burst, start new one
                    bursts.append((current_burst_start, current_burst_end))
                    current_burst_start = peak_times[i]
                    current_burst_end = peak_times[i]

            # Add final burst
            bursts.append((current_burst_start, current_burst_end))

            # Calculate burst statistics
            burst_count = len(bursts)

            if burst_count > 0:
                # Total time in bursts (with padding for movement duration)
                total_burst_time = sum(end - start + 0.2 for start, end in bursts)
                epoch_duration = len(eog_signal) / fs
                burst_density = min(total_burst_time / epoch_duration, 1.0)

                # Mean burst duration
                burst_durations = [end - start for start, end in bursts]
                mean_burst_duration = np.mean(burst_durations)

    features = {
        "eog_peak_amplitude": peak_amplitude,
        "eog_variance": variance,
        "eog_rms": rms,
        "eog_sem_score": sem_score,           # N1-specific slow eye movements
        "eog_rem_score": rem_score,           # REM detection
        "eog_zero_crossings": zero_crossings,
        "eog_mean_abs_value": mean_abs_value,
        "eog_sem_rem_ratio": sem_rem_ratio,   # N1 vs REM discriminator
        "eog_burst_count": burst_count,       # REM-specific: number of bursts
        "eog_burst_density": burst_density,   # REM-specific: proportion in bursts
        "eog_mean_burst_duration": mean_burst_duration,  # REM-specific: avg burst length
    }

    return features


def extract_emg_features(emg_signal, fs=125):
    """
    Extract EMG-specific features for muscle tone detection.

    EMG signals are used to detect:
    - Muscle tone levels (high in wake, low in REM due to atonia)
    - REM muscle atonia (Iteration 4 - sustained low tone markers)
    - Muscle twitches and artifacts
    - Sleep-related muscle activity

    Args:
        emg_signal (np.ndarray): 1D array of EMG signal data
        fs (float): Sampling frequency (default 125 Hz for EMG)

    Returns:
        dict: 7 EMG-specific features (4 from Iter 3 + 3 atonia features from Iter 4)
    """
    # 1. RMS - signal power (low in REM, high in wake/NREM)
    emg_rms = np.sqrt(np.mean(emg_signal**2))

    # 2. Standard deviation - signal variability
    emg_std = np.std(emg_signal)

    # 3. Mean absolute value - overall activity level
    emg_mean_abs = np.mean(np.abs(emg_signal))

    # 4. High-frequency power ratio (20-40 Hz) - muscle activity indicator
    # Muscle activity concentrates in high-frequency band
    nperseg = min(256, len(emg_signal))
    freqs, psd = signal.welch(
        emg_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2
    )

    # Calculate power in high-frequency band (20-40 Hz) - muscle activity
    hf_idx = np.logical_and(freqs >= 20, freqs <= 40)
    total_idx = freqs <= 45  # Total band up to 45 Hz (avoids 50 Hz powerline)

    hf_power = np.trapz(psd[hf_idx], freqs[hf_idx]) if np.sum(hf_idx) > 0 else 0
    total_power = np.trapz(psd[total_idx], freqs[total_idx]) if np.sum(total_idx) > 0 else 1e-10

    hf_ratio = hf_power / total_power

    # 5-7. REM-SPECIFIC: Muscle atonia markers (Iteration 4)
    # REM has the lowest muscle tone of any sleep stage
    # These features capture sustained low tone, not just overall activity
    emg_abs = np.abs(emg_signal)

    # 5. 10th percentile - captures sustained low muscle tone
    # Robust to occasional twitches that occur even during REM atonia
    emg_percentile_10 = np.percentile(emg_abs, 10)

    # 6. Minimum to mean ratio - normalized measure of baseline tone
    # REM: very low ratio (minimum much lower than mean)
    # Wake: higher ratio (minimum closer to mean)
    emg_min = np.min(emg_abs)
    emg_min_ratio = emg_min / (emg_mean_abs + 1e-10)

    # 7. Low tone proportion - percentage of epoch with sustained low tone
    # Threshold: 30% of mean amplitude (empirically determined)
    low_tone_threshold = 0.3 * emg_mean_abs
    emg_low_tone_proportion = np.sum(emg_abs < low_tone_threshold) / len(emg_abs)

    features = {
        "emg_rms": emg_rms,
        "emg_std": emg_std,
        "emg_mean_abs": emg_mean_abs,
        "emg_hf_ratio": hf_ratio,
        "emg_percentile_10": emg_percentile_10,       # REM-specific: sustained low tone
        "emg_min_ratio": emg_min_ratio,               # REM-specific: baseline tone measure
        "emg_low_tone_proportion": emg_low_tone_proportion,  # REM-specific: temporal persistence
    }

    return features
