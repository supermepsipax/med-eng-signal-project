# -- Project Configuration --
import os

# Set the current iteration of the project (1-4). 
# This controls which parts of the pipeline are active.
CURRENT_ITERATION = 2

# Set to True to use cached data for preprocessing and feature extraction.
USE_CACHE = False  # Temporarily disabled for testing with real data

# -- File Paths --
DATA_DIR = 'data/'
TRAINING_DIR = f'{DATA_DIR}training/'
HOLDOUT_DIR = f'{DATA_DIR}holdout/'
SAMPLE_DIR = f'{DATA_DIR}sample/'
CACHE_DIR = 'cache/'

# Validate and create directories if needed
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}\nPlease ensure you are running from the correct directory.")
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

# -- Preprocessing --
LOW_PASS_FILTER_FREQ = 40  # Hz
EEG_BANDPASS_FILTER_FREQ = [0.5, 40]

# -- Feature Extraction --
# (Add feature-specific parameters here)

# -- Classification --
# Cross-validation settings (used only when hyperparameter tuning is enabled)
CV_FOLDS = 5  # Number of folds for GridSearchCV when KNN_TUNE_K=True or SVM_TUNE_HYPERPARAMS=True

# Iteration-specific parameters - students should modify these based on current iteration
if CURRENT_ITERATION == 1:
    # Iteration 1: Basic pipeline with k-NN
    CLASSIFIER_TYPE = 'knn'
    KNN_N_NEIGHBORS = 5

    # k-NN Hyperparameter Tuning (OPTIONAL)
    # Set to True to enable automatic k tuning via GridSearchCV
    # If False, uses fixed KNN_N_NEIGHBORS value above
    # Tuning adds computation time but may improve performance
    KNN_TUNE_K = False

elif CURRENT_ITERATION == 2:
    # Iteration 2: Enhanced EEG processing with SVM
    CLASSIFIER_TYPE = 'svm'

    # SVM Hyperparameter Tuning (OPTIONAL)
    # Set to False to train a single model with fixed parameters (faster, good for testing)
    # Set to True to enable GridSearchCV hyperparameter tuning (slower, better performance)
    SVM_TUNE_HYPERPARAMS = False  # Default: False for faster iteration during development

    # Fixed SVM parameters (used when SVM_TUNE_HYPERPARAMS = False)
    SVM_C = 1.0              # Regularization parameter (higher = less regularization)
    SVM_KERNEL = 'rbf'       # Kernel type: 'rbf', 'linear', 'poly', 'sigmoid'
    SVM_GAMMA = 'scale'      # Kernel coefficient: 'scale', 'auto', or float value

    # SVM Hyperparameter Tuning Grid (used when SVM_TUNE_HYPERPARAMS = True)
    # Uncomment and modify to customize the hyperparameter search space
    # Default searches: C=[0.1, 1.0, 10.0, 100.0], gamma=['scale', 'auto', 0.001, 0.01, 0.1], kernel=['rbf', 'linear']
    # Example for faster tuning (fewer combinations):
    # SVM_PARAM_GRID = {
    #     'C': [0.1, 1.0, 10.0],
    #     'gamma': ['scale', 0.01],
    #     'kernel': ['rbf']
    # }

elif CURRENT_ITERATION == 3:
    # Iteration 3: Multi-signal processing with Random Forest
    CLASSIFIER_TYPE = 'random_forest'

    # NOTE: The parameters below are DEPRECATED and no longer used
    # Random Forest now uses GridSearchCV for automatic hyperparameter tuning
    # See RF_PARAM_GRID below to customize the search space
    RF_N_ESTIMATORS = 100  # Deprecated - kept for backward compatibility
    RF_MAX_DEPTH = 10  # Deprecated - kept for backward compatibility

    # Random Forest Hyperparameter Tuning Grid (OPTIONAL)
    # Uncomment and modify to customize the hyperparameter search space
    # Default searches: n_estimators=[50, 100, 200], max_depth=[None, 10, 20, 30],
    #                   min_samples_split=[2, 5, 10], min_samples_leaf=[1, 2, 4]
    # Example for faster tuning (fewer combinations):
    # RF_PARAM_GRID = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 20],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2]
    # }

elif CURRENT_ITERATION == 4:
    # Iteration 4: Full system optimization
    CLASSIFIER_TYPE = 'random_forest'

    # NOTE: The parameters below are DEPRECATED and no longer used
    # Random Forest now uses GridSearchCV for automatic hyperparameter tuning
    # See RF_PARAM_GRID below to customize the search space
    RF_N_ESTIMATORS = 200  # Deprecated - kept for backward compatibility
    RF_MAX_DEPTH = None  # Deprecated - kept for backward compatibility
    RF_MIN_SAMPLES_SPLIT = 5  # Deprecated - kept for backward compatibility

    # Random Forest Hyperparameter Tuning Grid (OPTIONAL)
    # Uncomment and modify to customize the hyperparameter search space
    # For Iteration 4, you might want a more refined search based on Iteration 3 results
    # Example for comprehensive tuning:
    # RF_PARAM_GRID = {
    #     'n_estimators': [100, 150, 200, 250],
    #     'max_depth': [None, 20, 30, 40],
    #     'min_samples_split': [2, 5, 8],
    #     'min_samples_leaf': [1, 2, 3, 4]
    # }

else:
    raise ValueError(f"Invalid CURRENT_ITERATION: {CURRENT_ITERATION}. Must be 1-4.")

# -- Submission --
SUBMISSION_FILE = 'submission.csv'
