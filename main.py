import config
from src.data_loader import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import train_classifier
from src.visualization import visualize_results, visualize_advanced_metrics
from src.report import generate_report
from src.utils import save_cache, load_cache
import os
import sys
import io
import numpy as np


def main():
    # Create a string buffer to capture processing log
    stdout_buffer = io.StringIO()

    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to the buffer
    sys.stdout = stdout_buffer 

    print("\n=== PROCESSING LOG ===")
    print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")

    # 1. Load Data
    print("\n=== STEP 1: DATA LOADING ===")
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
            print(f"Preprocessed data shape: {preprocessed_data.shape}")
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
    model = None
    metrics_dict = None
    model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
    metrics_filename = f"metrics_iter{config.CURRENT_ITERATION}.joblib"

    if selected_features.shape[1] > 0:
        # Try to load model and metrics from cache
        if config.USE_CACHE:
            model = load_cache(model_filename, config.CACHE_DIR)
            metrics_dict = load_cache(metrics_filename, config.CACHE_DIR)
            if model is not None and metrics_dict is not None:
                print(f"Loaded {config.CLASSIFIER_TYPE} classifier from cache")
                print(f"Loaded metrics from cache")

        # Train if not cached
        if model is None:
            model, metrics_dict = train_classifier(selected_features, labels, config)
            print(f"Trained {config.CLASSIFIER_TYPE} classifier")

            # Save model and metrics to cache
            if config.USE_CACHE:
                save_cache(model, model_filename, config.CACHE_DIR)
                save_cache(metrics_dict, metrics_filename, config.CACHE_DIR)
                print("Saved model and metrics to cache")

        # Display metrics
        print(f"\n📊 Final Test Metrics Summary:")
        print(f"  Accuracy: {metrics_dict['test_accuracy']:.3f}")
        print(f"  Macro F1: {metrics_dict['test_f1_macro']:.3f}")
        print(f"  Weighted F1: {metrics_dict['test_f1_weighted']:.3f}")
        if metrics_dict.get('test_kappa') is not None:
            print(f"  Cohen's Kappa: {metrics_dict['test_kappa']:.3f}")
    else:
        print("⚠️  WARNING: Cannot train classifier - no features available!")
        print("Students must implement feature extraction first.")

    # 6. Visualization
    print("\n=== STEP 6: VISUALIZATION ===")
    if model is not None:
        # Use test set predictions from metrics_dict for accurate confusion matrix
        visualize_results(metrics_dict, config)
        print("✓ Confusion matrix saved to: confusion_matrix.png")

        # Generate advanced visualizations
        visualize_advanced_metrics(metrics_dict, config)
        if metrics_dict.get('y_pred_proba') is not None:
            print("✓ ROC curves saved to: roc_curves_cv.png")
        if config.CURRENT_ITERATION >= 3:
            print("  (Feature importance available in Iteration 3+ with Random Forest)")
    else:
        print("Skipping visualization - no trained model")


    # Restore the original stdout
    sys.stdout = original_stdout

    # Get the captured output from the buffer
    processing_log = stdout_buffer.getvalue()

    if model is not None:
        generate_report(model, selected_features, labels, config, processing_log, metrics_dict)
        print("✓ Report saved to: report.txt")
    else:
        print("Skipping report - no trained model")

    print("\n" + "="*50)
    print("PIPELINE FINISHED")
    if model is None:
        print("⚠️  Students need to implement missing components!")
    print("="*50)

if __name__ == "__main__":
    main()