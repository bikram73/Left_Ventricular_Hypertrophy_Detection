"""
Configuration file for LVH Detection Project
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
SPLITS_DIR = DATA_DIR / 'splits'

# Dataset specific paths
ECG_RAW_DIR = RAW_DATA_DIR / 'ecg' / 'heartbeat'
# Corrected paths based on your data location
MRI_RAW_DIR = RAW_DATA_DIR / 'mri'
CT_RAW_DIR = RAW_DATA_DIR / 'ct' / 'data'
CLINICAL_RAW_FILE = RAW_DATA_DIR / 'clinical' / 'heart_failure.csv'

# Processed data paths
ECG_PROCESSED_FILE = PROCESSED_DATA_DIR / 'ecg_features.csv'
MRI_PROCESSED_DIR = PROCESSED_DATA_DIR / 'mri_processed'
CT_PROCESSED_DIR = PROCESSED_DATA_DIR / 'ct_processed'
CLINICAL_PROCESSED_FILE = PROCESSED_DATA_DIR / 'clinical_processed.csv'

# Model paths
MODELS_DIR = BASE_DIR / 'models'
ECG_MODEL_PATH = MODELS_DIR / 'ecg_classifier.pkl'
CNN_MODEL_PATH = MODELS_DIR / 'cnn_model.h5'
ENSEMBLE_MODEL_PATH = MODELS_DIR / 'ensemble_model.pkl'
ECG_SCALER_PATH = MODELS_DIR / 'scalers' / 'ecg_scaler.pkl'
CLINICAL_SCALER_PATH = MODELS_DIR / 'scalers' / 'clinical_scaler.pkl'

# Flask app configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'lvh-detection-secret-key'
    UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'dcm'}

# Data processing parameters
ECG_PARAMS = {
    'sampling_rate': 360,
    'duration': 10,  # seconds
    'num_leads': 12,
    'target_length': 3600  # 10 seconds * 360 Hz
}

MRI_PARAMS = {
    'target_size': (128, 128),
    'num_slices': 20,
    'normalize': True
}

CT_PARAMS = {
    'target_size': (128, 128),
    'window_center': 40,
    'window_width': 400,
    'normalize': True
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Ensemble model weights
ENSEMBLE_WEIGHTS = {
    'ecg': 0.4,
    'mri': 0.3,
    'ct': 0.2,
    'clinical': 0.1
}

# Kaggle dataset information
KAGGLE_DATASETS = {
    'ecg': 'shayanfazeli/heartbeat',
    'mri': 'salikhussaini49/sunnybrook-cardiac-mri',
    'ct': 'nikhilroxtomar/ct-heart-segmentation',
    'clinical': 'fedesoriano/heart-failure-prediction'
}

# Create directories if they don't exist
def create_directories():
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, 
                 MODELS_DIR, MODELS_DIR / 'scalers', 
                 MRI_PROCESSED_DIR, CT_PROCESSED_DIR,
                 Config.UPLOAD_FOLDER]:
        path.mkdir(parents=True, exist_ok=True)

# Create directories when module is imported
create_directories()