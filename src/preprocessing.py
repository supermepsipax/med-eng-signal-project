from scipy.signal import butter, filtfilt, iirnotch
import numpy as np

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Zero-phase low-pass Butterworth filter using filtfilt.

    filtfilt applies the filter forward and backward, eliminating phase distortion
    and providing better edge handling than lfilter. This is critical for
    biomedical signals where waveform shape must be preserved.

    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal with zero phase distortion.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # Ensure cutoff is below Nyquist frequency
    if normal_cutoff >= 1.0:
        raise ValueError(f"Cutoff frequency {cutoff} Hz must be less than Nyquist frequency {nyquist} Hz")

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def highpass_filter(data, cutoff, fs, order=5):
    """
    Zero-phase high-pass Butterworth filter to remove DC drift and slow artifacts.

    Useful for removing low-frequency noise and baseline drift in EEG signals.
    Typical cutoff frequencies for EEG: 0.5-1 Hz

    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter (Hz).
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal with zero phase distortion.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # Ensure cutoff is valid
    if normal_cutoff >= 1.0:
        raise ValueError(f"Cutoff frequency {cutoff} Hz must be less than Nyquist frequency {nyquist} Hz")

    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def notch_filter(data, center_freq, fs, Q=30):
    """
    Zero-phase notch filter to remove powerline interference at a specific frequency.

    Commonly used to remove 50 Hz or 60 Hz powerline noise from EEG signals.
    The Q factor determines the width of the notch (higher Q = narrower notch).

    Args:
        data (np.ndarray): The input signal.
        center_freq (float): The center frequency to remove (Hz), typically 50 or 60.
        fs (int): The sampling frequency of the signal.
        Q (float): Quality factor. Higher values = narrower notch. Default is 30.

    Returns:
        np.ndarray: The filtered signal with zero phase distortion.
    """
    # Design notch filter
    b, a = iirnotch(center_freq, Q, fs)
    y = filtfilt(b, a, data)
    return y

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Zero-phase bandpass Butterworth filter to retain frequencies within a specific range.

    Combines high-pass and low-pass filtering. Common ranges:
    - EEG: 0.5-40 Hz (preserves delta through beta bands)
    - EOG: 0.3-20 Hz (preserves slow eye movements)
    - EMG: 10-60 Hz (preserves muscle tone frequencies)

    Args:
        data (np.ndarray): The input signal.
        lowcut (float): The low cutoff frequency (Hz).
        highcut (float): The high cutoff frequency (Hz).
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal with zero phase distortion.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Ensure frequencies are valid
    if high >= 1.0:
        raise ValueError(f"High cutoff {highcut} Hz must be less than Nyquist frequency {nyquist} Hz")
    if low <= 0:
        raise ValueError(f"Low cutoff {lowcut} Hz must be greater than 0")
    if low >= high:
        raise ValueError(f"Low cutoff {lowcut} Hz must be less than high cutoff {highcut} Hz")

    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

def preprocess(data, channel_info, config, record_ids=None):
    """
    STUDENT IMPLEMENTATION AREA: Preprocess data based on current iteration.

    This function should handle both single-channel and multi-channel data
    (2 EEG + 2 EOG + 1 EMG channels) based on the data structure.

    IMPORTANT: Uses continuous signal filtering to avoid edge effects.
    When record_ids are provided, concatenates epochs by recording, filters
    the continuous signal, then re-segments into epochs. This eliminates
    inter-epoch edge artifacts and preserves slow oscillations.

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        channel_info (dict): Channel metadata including sampling rates and channel names
        config (module): The configuration module.
        record_ids (np.ndarray): Array mapping each epoch to its recording ID (optional)

    Returns:
        Same format as input: preprocessed data.
    """
    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")

    # Detect data format
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return preprocess_multi_channel(data, channel_info, config, record_ids)
    else:
        print("Processing single-channel data (backward compatibility)")
        return preprocess_single_channel(data, channel_info, config, record_ids)

def preprocess_multi_channel(multi_channel_data, channel_info, config, record_ids=None):
    """
    Preprocess multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.

    CONTINUOUS SIGNAL FILTERING APPROACH:
    - When record_ids provided: Concatenates epochs per recording, filters continuous signal, re-segments
    - When record_ids absent: Falls back to per-epoch filtering (less optimal but functional)

    This eliminates inter-epoch edge artifacts and properly handles slow oscillations.

    Args:
        multi_channel_data (dict): Multi-channel data with keys 'eeg', 'eog', 'emg'
        channel_info (dict): Channel metadata including sampling rates
        config (module): Configuration module
        record_ids (np.ndarray): Array mapping each epoch to its recording ID

    Returns:
        dict: Preprocessed multi-channel data in same format
    """
    preprocessed_data = {}

    # Get sampling rates
    eeg_fs = channel_info.get('eeg_fs', 125) if channel_info else 125
    eog_fs = channel_info.get('eog_fs', 50) if channel_info else 50
    emg_fs = channel_info.get('emg_fs', 125) if channel_info else 125

    # Determine filtering approach
    use_continuous_filtering = (record_ids is not None)

    if use_continuous_filtering:
        print("✓ Using continuous signal filtering (eliminates edge artifacts)")
        unique_recordings = np.unique(record_ids)
        print(f"  Processing {len(unique_recordings)} recordings separately")
    else:
        print("⚠ Warning: No record_ids provided, using per-epoch filtering")
        print("  (May have edge artifacts - consider passing record_ids)")

    # Process EEG channels (2 channels)
    eeg_data = multi_channel_data['eeg']
    if use_continuous_filtering:
        preprocessed_eeg = filter_continuous_multichannel(
            eeg_data, record_ids,
            config.EEG_BANDPASS_FILTER_FREQ[0],
            config.EEG_BANDPASS_FILTER_FREQ[1],
            eeg_fs,
            signal_type='EEG'
        )
    else:
        preprocessed_eeg = filter_epochs_multichannel(
            eeg_data,
            config.EEG_BANDPASS_FILTER_FREQ[0],
            config.EEG_BANDPASS_FILTER_FREQ[1],
            eeg_fs
        )
    preprocessed_data['eeg'] = preprocessed_eeg

    if config.CURRENT_ITERATION >= 2:  # EOG starts in iteration 2
        eog_data = multi_channel_data['eog']
        if use_continuous_filtering:
            preprocessed_eog = filter_continuous_multichannel(
                eog_data, record_ids,
                config.EOG_BANDPASS_FILTER_FREQ[0],
                config.EOG_BANDPASS_FILTER_FREQ[1],
                eog_fs,
                signal_type='EOG'
            )
        else:
            preprocessed_eog = filter_epochs_multichannel(
                eog_data,
                config.EOG_BANDPASS_FILTER_FREQ[0],
                config.EOG_BANDPASS_FILTER_FREQ[1],
                eog_fs
            )
        preprocessed_data['eog'] = preprocessed_eog

    if config.CURRENT_ITERATION >= 3:  # EMG starts in iteration 3
        emg_data = multi_channel_data['emg']
        if use_continuous_filtering:
            preprocessed_emg = filter_continuous_multichannel(
                emg_data, record_ids,
                config.EMG_BANDPASS_FILTER_FREQ[0],
                config.EMG_BANDPASS_FILTER_FREQ[1],
                emg_fs,
                signal_type='EMG'
            )
        else:
            preprocessed_emg = filter_epochs_multichannel(
                emg_data,
                config.EMG_BANDPASS_FILTER_FREQ[0],
                config.EMG_BANDPASS_FILTER_FREQ[1],
                emg_fs
            )
        preprocessed_data['emg'] = preprocessed_emg

    # Print summary
    if config.CURRENT_ITERATION >= 3:
        print(f"✓ Filtered EEG ({config.EEG_BANDPASS_FILTER_FREQ[0]}-{config.EEG_BANDPASS_FILTER_FREQ[1]} Hz), " +
              f"EOG ({config.EOG_BANDPASS_FILTER_FREQ[0]}-{config.EOG_BANDPASS_FILTER_FREQ[1]} Hz), " +
              f"EMG ({config.EMG_BANDPASS_FILTER_FREQ[0]}-{config.EMG_BANDPASS_FILTER_FREQ[1]} Hz)")
    elif config.CURRENT_ITERATION >= 2:
        print(f"✓ Filtered EEG and EOG")
    else:
        print(f"✓ Filtered EEG")

    return preprocessed_data


def filter_continuous_multichannel(data, record_ids, lowcut, highcut, fs, signal_type='Signal'):
    """
    Filter multi-channel data by concatenating epochs per recording,
    filtering continuous signal, then re-segmenting.

    Args:
        data (np.ndarray): Shape (n_epochs, n_channels, n_samples)
        record_ids (np.ndarray): Shape (n_epochs,) - recording ID for each epoch
        lowcut, highcut (float): Bandpass filter frequencies
        fs (float): Sampling frequency
        signal_type (str): Name for logging (e.g., 'EEG', 'EOG', 'EMG')

    Returns:
        np.ndarray: Filtered data in same shape as input
    """
    n_epochs, n_channels, samples_per_epoch = data.shape
    filtered_data = np.zeros_like(data)

    unique_recordings = np.unique(record_ids)

    for recording_id in unique_recordings:
        # Find epochs belonging to this recording
        epoch_mask = (record_ids == recording_id)
        epoch_indices = np.where(epoch_mask)[0]

        # Process each channel separately
        for ch in range(n_channels):
            # Concatenate epochs into continuous signal
            continuous_signal = data[epoch_mask, ch, :].flatten()

            # Filter the continuous signal (zero-phase with filtfilt)
            filtered_continuous = bandpass_filter(continuous_signal, lowcut, highcut, fs)

            # Re-segment into epochs
            for i, epoch_idx in enumerate(epoch_indices):
                start_sample = i * samples_per_epoch
                end_sample = start_sample + samples_per_epoch
                filtered_data[epoch_idx, ch, :] = filtered_continuous[start_sample:end_sample]

    return filtered_data


def filter_epochs_multichannel(data, lowcut, highcut, fs):
    """
    Fallback: Filter each epoch independently (may have edge artifacts).

    Args:
        data (np.ndarray): Shape (n_epochs, n_channels, n_samples)
        lowcut, highcut (float): Bandpass filter frequencies
        fs (float): Sampling frequency

    Returns:
        np.ndarray: Filtered data in same shape as input
    """
    n_epochs, n_channels, samples_per_epoch = data.shape
    filtered_data = np.zeros_like(data)

    for ch in range(n_channels):
        for epoch in range(n_epochs):
            signal = data[epoch, ch, :]
            filtered_data[epoch, ch, :] = bandpass_filter(signal, lowcut, highcut, fs)

    return filtered_data


def preprocess_single_channel(data, channel_info, config):
    """
    Backward compatibility for single-channel preprocessing.
    """
    if config.CURRENT_ITERATION == 1:
        # EXAMPLE: Very basic low-pass filter (students should expand)
        # Get actual sampling rate from channel_info, fallback to 125 Hz
        fs = channel_info.get('eeg_fs', 125) if channel_info else 125
        preprocessed_data = lowpass_filter(data, config.LOW_PASS_FILTER_FREQ, fs)

    elif config.CURRENT_ITERATION == 2:
        print("TODO: Implement enhanced preprocessing for iteration 2")
        preprocessed_data = data  # Placeholder

    elif config.CURRENT_ITERATION >= 3:
        print("TODO: Students should use multi-channel data format for iteration 3+")
        preprocessed_data = data  # Placeholder

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return preprocessed_data
