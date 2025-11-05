import config
from src.data_loader import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import train_classifier
from src.visualization import visualize_results
from src.report import generate_report
from src.utils import save_cache, load_cache
import os
import sys
import io
import numpy as np


def main():
    # # Create a string buffer
    # stdout_buffer = io.StringIO()
    #
    # # Save the original stdout
    # original_stdout = sys.stdout
    #
    # # Redirect stdout to the buffer
    # sys.stdout = stdout_buffer 

    print("\n=== PROCESSING LOG ===")

    print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")

    # 1. Load Data
    # Load ALL available data files from training directory
    print("\n=== STEP 1: DATA LOADING ===")

    # Use the load_all_training_data function to load all recordings at once
    multi_channel_data, labels, record_ids, channel_info = load_all_training_data(
        config.TRAINING_DIR,
        epoch_length=30
    )

    print("\nLoaded multi-channel data:")
    if 'eeg' in multi_channel_data:
        print(f"  EEG: {multi_channel_data['eeg'].shape}")
    if 'eog' in multi_channel_data:
        print(f"  EOG: {multi_channel_data['eog'].shape}")
    if 'emg' in multi_channel_data:
        print(f"  EMG: {multi_channel_data['emg'].shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Unique recordings: {len(np.unique(record_ids))}")

    print("\nChannel information:")
    if 'eeg_fs' in channel_info:
        print(f"  EEG sampling rate: {channel_info['eeg_fs']} Hz")
    if 'eog_fs' in channel_info:
        print(f"  EOG sampling rate: {channel_info['eog_fs']} Hz")
    if 'emg_fs' in channel_info:
        print(f"  EMG sampling rate: {channel_info['emg_fs']} Hz")

    # 2. Preprocessing
    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessed_data = None
    cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
        if preprocessed_data is not None:
            print("Loaded preprocessed data from cache")

    if preprocessed_data is None:
        preprocessed_data = preprocess(multi_channel_data, channel_info, config)
        # Display preprocessed data info
        if isinstance(preprocessed_data, dict):
            print("Preprocessed multi-channel data:")
            for signal_type in preprocessed_data.keys():
                print(f"  {signal_type.upper()}: {preprocessed_data[signal_type].shape}")
        else:
            print("Preprocessed data shape: {preprocessed_data.shape}")
        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")

    # 3. Feature Extraction
    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    features = None
    cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        features = load_cache(cache_filename_features, config.CACHE_DIR)
        if features is not None:
            print("Loaded features from cache")

    if features is None:
        features = extract_features(preprocessed_data, config)
        print(f"Extracted features shape: {features.shape}")
        if features.shape[1] == 0:
            print("⚠️  WARNING: No features extracted! Students must implement feature extraction.")
        if config.USE_CACHE:
            save_cache(features, cache_filename_features, config.CACHE_DIR)
            print("Saved features to cache")

    # 4. Feature Selection
    print("\n=== STEP 4: FEATURE SELECTION ===")
    selected_features = select_features(features, labels, config)
    print(f"Selected features shape: {selected_features.shape}")

    # 5. Classification
    print("\n=== STEP 5: CLASSIFICATION ===")
    if selected_features.shape[1] > 0:
        model = train_classifier(selected_features, labels, config)
        print(f"Trained {config.CLASSIFIER_TYPE} classifier")
    else:
        print("⚠️  WARNING: Cannot train classifier - no features available!")
        print("Students must implement feature extraction first.")
        model = None

    # Create a string buffer
    stdout_buffer = io.StringIO()

    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to the buffer
    sys.stdout = stdout_buffer 
    # 6. Visualization
    print("\n=== STEP 6: VISUALIZATION ===")
    if model is not None:
        visualize_results(model, selected_features, labels, config)
    else:
        print("Skipping visualization - no trained model")

    # 7. Report Generation
    print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===")

    # Restore the original stdout
    sys.stdout = original_stdout

    # Get the captured output from the buffer
    processing_log = stdout_buffer.getvalue()   
     
    if model is not None:
        generate_report(model, selected_features, labels, config, processing_log)
        model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(model, model_filename,config.CACHE_DIR )
    else:
        print("Skipping report - no trained model")

    print("\n" + "="*50)
    print("PIPELINE FINISHED")
    if model is None:
        print("⚠️  Students need to implement missing components!")
    print("="*50)

if __name__ == "__main__":
    main()
