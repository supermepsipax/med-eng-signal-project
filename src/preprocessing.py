from scipy.signal import butter, lfilter, iirnotch
import numpy as np

def lowpass_filter(data, cutoff, fs, order=5):
    """
    EXAMPLE IMPLEMENTATION: Simple low-pass Butterworth filter.

    Students should understand this basic filter and consider:
    - Is 40Hz the right cutoff for EEG?
    - What about high-pass filtering?
    - Should you use bandpass instead?
    - What about notch filtering for powerline interference?

    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    # TODO: Students may want to implement additional filtering:
    # - High-pass filter to remove DC drift
    # - Notch filter for 50/60 Hz powerline noise
    # - Bandpass filter (e.g., 0.5-40 Hz for EEG)

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def highpass_filter(data, cutoff, fs, order=5):
    """
    High-pass Butterworth filter to remove DC drift and slow artifacts.

    Useful for removing low-frequency noise and baseline drift in EEG signals.
    Typical cutoff frequencies for EEG: 0.5-1 Hz

    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter (Hz).
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def notch_filter(data, center_freq, fs, Q=30):
    """
    Notch filter to remove powerline interference at a specific frequency.

    Commonly used to remove 50 Hz or 60 Hz powerline noise from EEG signals.
    The Q factor determines the width of the notch (higher Q = narrower notch).

    Args:
        data (np.ndarray): The input signal.
        center_freq (float): The center frequency to remove (Hz), typically 50 or 60.
        fs (int): The sampling frequency of the signal.
        Q (float): Quality factor. Higher values = narrower notch. Default is 30.

    Returns:
        np.ndarray: The filtered signal.
    """
    # Design notch filter
    b, a = iirnotch(center_freq, Q, fs)
    y = lfilter(b, a, data)
    return y

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Bandpass Butterworth filter to retain frequencies within a specific range.

    Combines high-pass and low-pass filtering. Common EEG bandpass: 0.5-40 Hz
    This is often more efficient than applying separate high-pass and low-pass filters.

    Args:
        data (np.ndarray): The input signal.
        lowcut (float): The low cutoff frequency (Hz).
        highcut (float): The high cutoff frequency (Hz).
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = lfilter(b, a, data)
    return y

def preprocess(data, channel_info, config):
    """
    STUDENT IMPLEMENTATION AREA: Preprocess data based on current iteration.

    This function should handle both single-channel and multi-channel data
    (2 EEG + 2 EOG + 1 EMG channels) based on the data structure.

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        channel_info (dict): Channel metadata including sampling rates and channel names
        config (module): The configuration module.

    Returns:
        Same format as input: preprocessed data.
    """
    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")

    # Detect data format
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return preprocess_multi_channel(data, channel_info, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return preprocess_single_channel(data, channel_info, config)

def preprocess_multi_channel(multi_channel_data, channel_info, config):
    """
    Preprocess multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.
    Each channel type may have different sampling rates and require different processing.
    """
    preprocessed_data = {}

    # Process EEG channels (2 channels)
    eeg_data = multi_channel_data['eeg']
    # Get actual sampling rate from channel_info, fallback to 125 Hz
    eeg_fs = channel_info.get('eeg_fs', 125) if channel_info else 125
    preprocessed_eeg = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]):
        for epoch in range(eeg_data.shape[0]):
            signal = eeg_data[epoch, ch, :]
            # Apply EEG-specific preprocessing
            # filtered_signal = lowpass_filter(signal, config.LOW_PASS_FILTER_FREQ, eeg_fs)
            filtered_signal = bandpass_filter(signal, config.EEG_BANDPASS_FILTER_FREQ[0], config.EEG_BANDPASS_FILTER_FREQ[1], eeg_fs)
            # TODO: Students should add bandpass filter, artifact removal
            preprocessed_eeg[epoch, ch, :] = filtered_signal

    preprocessed_data['eeg'] = preprocessed_eeg

    if config.CURRENT_ITERATION >= 2:  # EOG starts in iteration 2
        # Process EOG channels (2 channels) - may need different filtering
        eog_data = multi_channel_data['eog']
        # Get actual sampling rate from channel_info, fallback to 50 Hz
        eog_fs = channel_info.get('eog_fs', 50) if channel_info else 50
        preprocessed_eog = np.zeros_like(eog_data)

        for ch in range(eog_data.shape[1]):
            for epoch in range(eog_data.shape[0]):
                signal = eog_data[epoch, ch, :]
                # EOG may need different filter settings (preserve slow eye movements)
                # Cutoff must be < Nyquist frequency (eog_fs/2 = 25 Hz for 50 Hz sampling)
                filtered_signal = lowpass_filter(signal, 20, eog_fs)  # Lower cutoff for EOG
                preprocessed_eog[epoch, ch, :] = filtered_signal

        preprocessed_data['eog'] = preprocessed_eog

    if config.CURRENT_ITERATION >= 3:  # EMG starts in iteration 3
        # Process EMG channel (1 channel) - may need higher frequency preservation
        emg_data = multi_channel_data['emg']
        # Get actual sampling rate from channel_info, fallback to 125 Hz
        emg_fs = channel_info.get('emg_fs', 125) if channel_info else 125
        preprocessed_emg = np.zeros_like(emg_data)

        for epoch in range(emg_data.shape[0]):
            signal = emg_data[epoch, 0, :]
            # EMG needs higher frequency content preserved (muscle activity)
            # Cutoff must be < Nyquist frequency (emg_fs/2 = 62.5 Hz for 125 Hz sampling)
            filtered_signal = lowpass_filter(signal, 60, emg_fs)  # Higher cutoff for EMG
            preprocessed_emg[epoch, 0, :] = filtered_signal

        preprocessed_data['emg'] = preprocessed_emg
        print("Multi-channel preprocessing applied to EEG + EOG + EMG")
    elif config.CURRENT_ITERATION >= 2:
        print("Iteration 2: Processing EEG + EOG channels")
    else:
        print("Iteration 1: Processing EEG channels only")

    # TODO: Students should add:
    # - Channel-specific artifact removal
    # - Cross-channel artifact detection
    # - Signal quality assessment
    # - Normalization per channel type

    return preprocessed_data


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
