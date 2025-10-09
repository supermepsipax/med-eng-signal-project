# src/__init__.py

from .data_loader import (
    load_training_data,
    load_holdout_data,
    load_all_training_data
)
from .xml_parser import (
    parse_xml_annotations,
    create_epoch_labels
)

from .preprocessing import (
    preprocess,
    lowpass_filter
)

from .feature_extraction import (
    extract_features,
    extract_time_domain_features
)

from .feature_selection import (
    select_features
)

from .classification import (
    train_classifier
)

from .visualization import (
    visualize_results
)

from .report import (
    generate_report
)

from .utils import (
    save_cache,
    load_cache
)

__all__ = [
    # data_loader
    'load_training_data',
    'load_holdout_data',
    'load_all_training_data',
    # xml parser
    'parse_xml_annotations',
    'create_epoch_labels',
    # preprocessing
    'preprocess',
    'lowpass_filter',
    # feature_extraction
    'extract_features',
    'extract_time_domain_features',
    # feature_selection
    'select_features',
    # classification
    'train_classifier',
    # visualization
    'visualize_results',
    # report
    'generate_report',
    # utils
    'save_cache',
    'load_cache',
]
