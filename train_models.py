"""
Comprehensive Model Training Script for LVH Detection
Implements multiple algorithms: Random Forest, XGBoost, SVM, Neural Networks
Generates confusion matrix and performance metrics as per project requirements
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, classification_report, confusion_matrix, roc_curve)
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from pathlib import Path
import logging
import warnings
import json # Ensure json is imported
warnings.filterwarnings('ignore')

from config import (PROCESSED_DATA_DIR, MODELS_DIR, ECG_PROCESSED_FILE, 
                    CLINICAL_PROCESSED_FILE, RANDOM_STATE, TEST_SIZE)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LVHModelTrainer:
    """Comprehensive LVH Model Training with Multiple Algorithms"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.best_model = None
        self.best_score = 0
        self.thresholds = {}
        
    def load_processed_data(self):
        """Load all processed datasets from the new multimodal pipeline"""
        datasets = {}
        
        # Load clinical data
        if CLINICAL_PROCESSED_FILE.exists():
            clinical_data = pd.read_csv(CLINICAL_PROCESSED_FILE)
            datasets['clinical'] = clinical_data
            logger.info(f"Clinical data loaded: {clinical_data.shape}")
        
        # Load ECG features
        if ECG_PROCESSED_FILE.exists():
            ecg_data = pd.read_csv(ECG_PROCESSED_FILE)
            # Check if ECG data is too small - suggest using corrected processing
            if len(ecg_data) < 50:
                logger.warning(f"ECG data has only {len(ecg_data)} samples. Consider running 'python process_data_corrected.py' to generate more synthetic data.")
            datasets['ecg'] = ecg_data
            logger.info(f"ECG data loaded: {ecg_data.shape}")

        # Load MRI features
        mri_path = PROCESSED_DATA_DIR / 'mri_features.csv'
        if mri_path.exists():
            mri_data = pd.read_csv(mri_path)
            # Check for single-class issue
            if 'lvh_label' in mri_data.columns:
                unique_labels = mri_data['lvh_label'].nunique()
                if unique_labels < 2:
                    logger.warning(f"MRI data has only {unique_labels} class(es). Consider running 'python process_data_corrected.py' to fix labeling.")
            datasets['mri'] = mri_data
            logger.info(f"MRI data loaded: {mri_data.shape}")

        # Load CT features
        ct_path = PROCESSED_DATA_DIR / 'ct_features.csv'
        if ct_path.exists():
            ct_data = pd.read_csv(ct_path)
            # Check for single-class issue
            if 'lvh_label' in ct_data.columns:
                unique_labels = ct_data['lvh_label'].nunique()
                if unique_labels < 2:
                    logger.warning(f"CT data has only {unique_labels} class(es). Consider running 'python process_data_corrected.py' to fix labeling.")
            datasets['ct'] = ct_data
            logger.info(f"CT data loaded: {ct_data.shape}")

        if not datasets:
            logger.error("No datasets found. Please run process_data.py first.")
            return None
        
        return datasets
    
    def prepare_data(self, dataset, target_col='lvh_label'):
        """Prepare data for training"""
        
        # Flexible target column identification
        possible_targets = ['HeartDisease', 'target', 'label', 'class', 'lvh_label']
        target_col_found = None
        for col in possible_targets:
            if col in dataset.columns:
                target_col_found = col
                break
        
        if not target_col_found:
             raise ValueError(f"Target column not found in the dataset.")
        target_col = target_col_found

        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]
        
        # Note very small datasets (informational)
        if len(X) < 10:
            logger.info(f"Small dataset detected ({len(X)} samples).")
            
        # Check class distribution for stratified splitting
        class_counts = y.value_counts()
        stratify_param = None
        if len(class_counts) >= 2 and class_counts.min() >= 2:
            stratify_param = y
        elif len(class_counts) < 2:
            logger.info(f"Only one class present in target '{target_col}'. Cannot stratify.")
        else:
            logger.info("Class with very few samples detected. Using regular split without stratification.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_param
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Dimensionality reduction for very high-dimensional modalities (MRI/CT)
        pca = None
        PCA_THRESHOLD = 1000
        PCA_COMPONENTS = 100
        if X_train_scaled.shape[1] > PCA_THRESHOLD:
            n_comp = min(PCA_COMPONENTS, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1] - 1)
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)
                logger.info(f"Applied PCA ({n_comp} components) to reduce dimensionality for training.")
        
        logger.info(f"Data split successful - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        
        scaler_obj = {'scaler': scaler, 'pca': pca}
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler_obj

    def prepare_full_data(self, dataset, target_col='lvh_label'):
        """Prepares a full dataset (no train-test split) for analysis or cross-validation."""
        possible_targets = ['HeartDisease', 'target', 'label', 'class', 'lvh_label']
        target_col_found = None
        for col in possible_targets:
            if col in dataset.columns:
                target_col_found = col
                break

        if not target_col_found:
             raise ValueError(f"Target column not found in the dataset.")
        target_col = target_col_found

        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dimensionality reduction for very high-dimensional modalities (MRI/CT)
        pca = None
        PCA_THRESHOLD = 1000
        PCA_COMPONENTS = 100
        if X_scaled.shape[1] > PCA_THRESHOLD:
            # Ensure n_components is less than number of samples and features
            n_comp = min(PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1] - 1)
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
                X_scaled = pca.fit_transform(X_scaled)
                logger.info(f"Applied PCA ({n_comp} components) to reduce dimensionality for full-data training.")

        scaler_obj = {'scaler': scaler, 'pca': pca}
        return X_scaled, y, scaler_obj

    def train_with_cross_validation(self, name, model, X, y):
        """Train/evaluate model using cross-validation for small datasets and then fit on full data.

        Returns a result dict compatible with `train_and_evaluate`.
        """
        logger.info(f"Training {name} with cross-validation (small dataset)...")

        # Determine an appropriate CV splitter
        unique_classes = pd.Series(y).value_counts()
        if len(unique_classes) < 2:
            # Single-class: cannot do supervised CV
            raise ValueError("Single-class dataset passed to CV trainer.")

        # Prefer stratified splits where possible
        if unique_classes.min() >= 2:
            n_splits = min(5, int(unique_classes.min()))
            n_splits = max(2, n_splits)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        else:
            # Fall back to Leave-One-Out for very small imbalanced cases
            cv = LeaveOneOut()

        # Helper to safely compute metrics
        def _compute_metrics_safe(y_true, y_pred, y_pred_proba=None):
            accuracy = accuracy_score(y_true, y_pred)
            
            # If only one class is present in y_true, many metrics are undefined or misleading
            if len(np.unique(y_true)) < 2:
                precision = None
                recall = None
                f1 = None
                roc_auc = None
            else:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if y_pred_proba is None or len(np.unique(y_true)) < 2:
                    roc_auc = None # Explicitly None if not applicable
                else:
                    try:
                        roc_auc = roc_auc_score(y_true, y_pred_proba)
                    except Exception:
                        roc_auc = None
            return accuracy, precision, recall, f1, roc_auc

        # Helper to safely obtain predict_proba-like scores
        def _get_proba_from_model(estimator, X_in, fallback_pred=None):
            # Try predict_proba
            if hasattr(estimator, 'predict_proba'):
                try:
                    proba = estimator.predict_proba(X_in)
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        return proba[:, 1]
                    # single-column proba
                    return proba.ravel()
                except Exception:
                    pass
            # Try decision_function
            if hasattr(estimator, 'decision_function'):
                try:
                    scores = estimator.decision_function(X_in)
                    smin, smax = scores.min(), scores.max()
                    if smax > smin:
                        return (scores - smin) / (smax - smin)
                    return np.full_like(scores, 0.5, dtype=float)
                except Exception:
                    pass
            # Fall back to using fallback_pred (0/1) as probability
            if fallback_pred is not None:
                return (np.array(fallback_pred) == 1).astype(float)
            # Last resort
            return np.full((len(X_in),), 0.5, dtype=float)

        # Obtain cross-validated predictions (probabilities if possible)
        try:
            # Try to get cross-validated probabilities first, else get class predictions
            y_pred_proba = None
            y_pred = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        y_pred_proba = proba[:, 1]
                    else:
                        y_pred_proba = proba.ravel()
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                except Exception:
                    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
            else:
                y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
                # Try to get a decision_function-based score
                if hasattr(model, 'decision_function'):
                    try:
                        scores = cross_val_predict(model, X, y, cv=cv, method='decision_function')
                        smin, smax = scores.min(), scores.max()
                        if smax > smin:
                            y_pred_proba = (scores - smin) / (smax - smin)
                        else:
                            y_pred_proba = np.full_like(scores, 0.5, dtype=float)
                    except Exception:
                        y_pred_proba = None


        except Exception as e:
            logger.warning(f"Cross-validation prediction failed for {name}: {e}. Falling back to simple split training.")
            # As a last resort, fit on full data and evaluate on the same data
            model.fit(X, y)
            y_pred = model.predict(X)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
                smin, smax = scores.min(), scores.max()
                if smax > smin:
                    y_pred_proba = (scores - smin) / (smax - smin)
                else:
                    y_pred_proba = np.full_like(scores, 0.5, dtype=float)
            else:
                y_pred_proba = (np.array(y_pred) == 1).astype(float)

            threshold = self._find_best_threshold(y, y_pred_proba) if y_pred_proba is not None else 0.5
            y_pred_final = (y_pred_proba >= threshold).astype(int) if y_pred_proba is not None else y_pred

            # Fit model on full data (already done)
            trained_model = model

            accuracy, precision, recall, f1, roc_auc = _compute_metrics_safe(y, y_pred_final, y_pred_proba)
            self.thresholds[name] = threshold

            return {
                'model': trained_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred_proba': y_pred_proba,
                'y_test': y,
                'confusion_matrix': confusion_matrix(y, y_pred_final),
                'threshold': threshold
            }

        # Fit model on full data before returning
        model.fit(X, y)
        trained_model = model

        threshold = 0.5
        if y_pred_proba is not None and len(np.unique(y)) >= 2:
            threshold = self._find_best_threshold(y, y_pred_proba)
            y_pred = (y_pred_proba >= threshold).astype(int)

        accuracy, precision, recall, f1, roc_auc = _compute_metrics_safe(y, y_pred, y_pred_proba)
        self.thresholds[name] = threshold

        return {
            'model': trained_model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba,
            'y_test': y,
            'confusion_matrix': confusion_matrix(y, y_pred),
            'threshold': threshold
        }

    def train_and_evaluate_one_class(self, name, X_train, X_test, y_train, y_test):
        """Fallback training for single-class datasets using One-Class SVM."""
        logger.info(f"Training {name} with One-Class SVM (single-class dataset)...")

        # Identify the present class label (assume binary labels 0/1)
        present_vals = pd.Series(y_train).unique()
        present_label = int(present_vals[0]) if len(present_vals) >= 1 else 1

        oc_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        oc_model.fit(X_train)

        # Predict: +1 for inliers, -1 for outliers
        pred_inlier = oc_model.predict(X_test)
        # Map to class labels: inliers -> present_label, outliers -> 1-present_label
        alt_label = 1 - present_label
        y_pred = np.where(pred_inlier == 1, present_label, alt_label)

        # Decision function as a pseudo-probability (scale to 0..1)
        try:
            scores = oc_model.decision_function(X_test)
            scores_min, scores_max = scores.min(), scores.max()
            if scores_max > scores_min:
                y_pred_proba = (scores - scores_min) / (scores_max - scores_min)
            else:
                y_pred_proba = np.full_like(scores, 0.5, dtype=float)
        except Exception:
            y_pred_proba = np.full((len(y_test),), 0.5, dtype=float)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # ROC-AUC not applicable for true single-class test sets
        roc_auc = None

        logger.info(f"✓ {name} (One-Class) Accuracy: {accuracy:.4f} | ROC AUC: n/a")

        self.thresholds[name] = 0.5
        return {
            'model': oc_model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'is_one_class': True,
            'threshold': 0.5
        }
    
    def initialize_models(self):
        """Initialize a standard set of models for training"""
        quick = os.environ.get('QUICK_TRAIN', '0') in ('1', 'true', 'True')
        n_est = 20 if quick else 100
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=n_est, random_state=RANDOM_STATE),
            'XGBoost': xgb.XGBClassifier(n_estimators=n_est, random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False),
            'LightGBM': lgb.LGBMClassifier(n_estimators=n_est, random_state=RANDOM_STATE, verbose=-1),
            'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
            'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'MLP': MLPClassifier(random_state=RANDOM_STATE, max_iter=500 if quick else 1000),
            'AdaBoost': AdaBoostClassifier(n_estimators=n_est, random_state=RANDOM_STATE)
        }

    def _find_best_threshold(self, y_true, scores):
        """Find threshold that maximizes accuracy given probability scores."""
        if scores is None or len(np.unique(y_true)) < 2:
            return 0.5

        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = -np.inf

        for thr in thresholds:
            preds = (scores >= thr).astype(int)
            score = accuracy_score(y_true, preds)
            if score > best_score + 1e-6:
                best_score = score
                best_threshold = float(thr)

        return best_threshold

    def determine_optimal_threshold(self, model, X, y):
        """Run cross-validated predictions to determine an optimal probability threshold."""
        if len(np.unique(y)) < 2:
            return 0.5

        try:
            estimator = clone(model)
        except Exception:
            return 0.5

        if hasattr(estimator, 'predict_proba'):
            method = 'predict_proba'
        elif hasattr(estimator, 'decision_function'):
            method = 'decision_function'
        else:
            return 0.5

        class_counts = pd.Series(y).value_counts()
        if len(class_counts) < 2 or class_counts.min() < 1:
            return 0.5

        max_splits = int(class_counts.min())
        if max_splits < 2:
            return 0.5

        n_splits = int(min(5, max_splits))
        if n_splits < 2:
            return 0.5

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        try:
            cv_outputs = cross_val_predict(estimator, X, y, cv=cv, method=method)
        except Exception as exc:
            logger.debug(f"Threshold optimization skipped: {exc}")
            return 0.5

        if method == 'predict_proba':
            if cv_outputs.ndim == 2 and cv_outputs.shape[1] > 1:
                scores = cv_outputs[:, 1]
            else:
                scores = cv_outputs.ravel()
        else:
            scores = cv_outputs.ravel()
            smin, smax = scores.min(), scores.max()
            if smax > smin:
                scores = (scores - smin) / (smax - smin)
            else:
                return 0.5

        return self._find_best_threshold(y, scores)

    # ---- Helpers for stable metric computation ----
    def _compute_metrics_safe(self, y_true, y_pred, y_pred_proba=None):
        accuracy = accuracy_score(y_true, y_pred)
        
        # If only one class is present in y_true, many metrics are undefined or misleading
        if len(np.unique(y_true)) < 2:
            precision = None
            recall = None
            f1 = None
            roc_auc = None
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if y_pred_proba is None or len(np.unique(y_true)) < 2:
                roc_auc = None # Explicitly None if not applicable
            else:
                try:
                    roc_auc = roc_auc_score(y_true, y_pred_proba)
                except Exception:
                    roc_auc = None
        return accuracy, precision, recall, f1, roc_auc

    def _get_proba_from_estimator(self, estimator, X, fallback_pred=None):
        # Try predict_proba
        if hasattr(estimator, 'predict_proba'):
            try:
                proba = estimator.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]
                return proba.ravel()
            except Exception:
                pass
        # Try decision_function
        if hasattr(estimator, 'decision_function'):
            try:
                scores = estimator.decision_function(X)
                smin, smax = scores.min(), scores.max()
                if smax > smin:
                    return (scores - smin) / (smax - smin)
                return np.full_like(scores, 0.5, dtype=float)
            except Exception:
                pass
        # Fallback to predicted labels as probability
        if fallback_pred is not None:
            return (np.array(fallback_pred) == 1).astype(float)
        return np.full((len(X),), 0.5, dtype=float)

    def train_and_evaluate(self, name, model, X_train, X_test, y_train, y_test):
        """Train and evaluate a single model and return performance metrics"""
        logger.info(f"Training {name}...")
        threshold = self.determine_optimal_threshold(model, X_train, y_train)
        model.fit(X_train, y_train)
        y_pred_default = model.predict(X_test)
        # Safely attempt to get probability-like scores
        try:
            y_pred_proba = self._get_proba_from_estimator(model, X_test, fallback_pred=y_pred_default)
        except Exception:
            y_pred_proba = (np.array(y_pred_default) == 1).astype(float)

        if y_pred_proba is not None and len(np.unique(y_train)) >= 2:
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = y_pred_default
            threshold = 0.5

        accuracy, precision, recall, f1, roc_auc = self._compute_metrics_safe(y_test, y_pred, y_pred_proba)

        logger.info(f"✓ {name} Accuracy: {accuracy:.4f} | ROC AUC: {roc_auc:.4f} | Thr: {threshold:.2f}")
        self.thresholds[name] = threshold

        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'threshold': threshold
        }

    def create_ensemble_model(self, trained_models):
        """Creates a voting classifier from the best-performing models."""
        estimators = []
        for name, model in trained_models.items():
            if name in ['RandomForest', 'XGBoost', 'LightGBM', 'SVM']:
                estimators.append((name.lower(), model))
        
        if len(estimators) < 2:
            logger.warning("Not enough trained models to create an ensemble. Skipping ensemble training.")
            return None
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        return ensemble

    def train_multimodal_pipeline(self):
        """Main training pipeline for all datasets"""
        logger.info("Starting multimodal model training pipeline...")
        datasets = self.load_processed_data()
        
        if datasets is None:
            return

        trained_models = {}
        self.results = {}
        self.scalers = {}
        self.thresholds = {}
        modality_test_targets = {}
        
        # Train a separate model for each data modality
        for name, data in datasets.items():
            try:
                logger.info(f"\n--- Processing modality: {name.upper()} ---")
                # First try a full-dataset check to see if training is possible
                X_full, y_full, scaler_full = self.prepare_full_data(data)
                self.scalers[name] = scaler_full

                # Re-check class balance on the full dataset
                class_counts_full = pd.Series(y_full).value_counts()

                self.initialize_models()

                # Heuristics & chosen strategy:
                # - If full dataset has only one class: one-class fallback
                # - If dataset has >=3 and <50 samples AND has at least 2 samples per class: cross-validated training (CV/LOO)
                # - If dataset has <3 samples OR has <2 samples in any class: regular split (no CV possible) or one-class
                # - Otherwise: regular train/test split
                strategy = 'split' # default
                if len(class_counts_full) < 2 or class_counts_full.min() < 2:
                    strategy = 'one-class'
                elif len(X_full) < 3:
                    strategy = 'split'
                elif len(X_full) < 50:
                    strategy = 'cross-validate'
                
                logger.info(f"Selected training strategy for '{name}': {strategy} (n_samples={len(X_full)}, class_counts={class_counts_full.to_dict()})")

                # Loop over all models for this modality
                for model_name, model in self.models.items():
                    logger.info(f"\n--- Training {model_name} on {name.upper()} data ---")
                    result_key = f"{name}_{model_name}"
                    
                    if strategy == 'one-class':
                        # Use a split and One-Class fallback
                        X_train, X_test, y_train, y_test, scaler = self.prepare_data(data)
                        result = self.train_and_evaluate_one_class(
                            result_key,
                            X_train, X_test, y_train, y_test
                        )
                    elif strategy == 'cross-validate':
                        # Small but multi-class dataset: use cross-validation trainer
                        result = self.train_with_cross_validation(
                            result_key,
                            model,
                            X_full, y_full
                        )
                    else: # 'split'
                        # Use regular train/test split and supervised training
                        X_train, X_test, y_train, y_test, scaler = self.prepare_data(data)
                        result = self.train_and_evaluate(
                            result_key,
                            model,
                            X_train, X_test, y_train, y_test
                        )
                    
                    trained_models[result_key] = result['model']
                    self.results[result_key] = result
                    modality_test_targets[name] = result.get('y_test')

            except Exception as e:
                logger.error(f"Failed to complete training for modality {name}: {e}")
                continue

        if not trained_models:
            logger.error("No models were trained. Exiting.")
            return

        # Train a final ensemble model only if datasets are aligned and share indices
        logger.info("\n--- Training Final Ensemble Model (if data aligned) ---")
        try:
            target_candidates = []
            for name, data in datasets.items():
                for col in ['HeartDisease', 'target', 'label', 'class', 'lvh_label']:
                    if col in data.columns:
                        target_candidates.append((name, col))
                        break

            # Require at least two modalities with identical number of rows to attempt a naive ensemble
            valid_modalities = [n for n, _ in target_candidates if len(datasets[n]) == len(datasets[target_candidates[0][0]])]
            if len(valid_modalities) >= 2:
                X_list = []
                for n in valid_modalities:
                    tgt_col = [c for c in ['HeartDisease', 'target', 'label', 'class', 'lvh_label'] if c in datasets[n].columns][0]
                    X_modality = datasets[n].drop(columns=[tgt_col])
                    X_list.append(self.scalers[n].transform(X_modality))

                combined_X = np.hstack(X_list)
                # Use the first modality's split as reference
                ref_y_test = modality_test_targets[valid_modalities[0]]
                # Split combined_X to train/test using the same indices as ref modality
                # Since we don't have saved indices, skip ensemble to avoid data leakage
                logger.warning("Skipping ensemble due to lack of shared split indices across modalities.")
            else:
                logger.warning("Not enough aligned modalities for ensemble. Skipping.")
        except Exception as e:
            logger.warning(f"Ensemble training skipped due to error: {e}")

        self.generate_and_save_artifacts(trained_models, datasets)
        
    def generate_and_save_artifacts(self, trained_models, datasets):
        """Generates and saves all reports and visualizations."""
        # Ensure model dirs exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (MODELS_DIR / 'scalers').mkdir(parents=True, exist_ok=True)

        # Save models and scalers
        joblib.dump(trained_models, MODELS_DIR / 'all_models.pkl')
        joblib.dump(self.scalers, MODELS_DIR / 'scalers' / 'all_scalers.pkl')
        
        # Select best model and save it
        if self.results:
            best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
            self.best_model = trained_models[best_model_name]
            self.best_score = self.results[best_model_name]['accuracy']
            joblib.dump(self.best_model, MODELS_DIR / 'best_lvh_model.pkl')
        else:
            logger.warning("No models were successfully trained, so no 'best_model.pkl' was saved.")

        if self.thresholds:
            thresholds_path = MODELS_DIR / 'model_thresholds.json'
            with open(thresholds_path, 'w', encoding='utf-8') as f:
                json.dump(self.thresholds, f, indent=2)
            logger.info(f"Model probability thresholds saved: {thresholds_path}")

        # Generate and save plots and report
        modalities = set(key.split('_')[0] for key in self.results.keys())
        for modality in modalities:
            modality_results = {k: v for k, v in self.results.items() if k.startswith(modality)}
            if not modality_results:
                continue

            # Create a subdirectory for the modality's plots
            modality_plot_dir = MODELS_DIR / modality
            modality_plot_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Generating plots for modality '{modality}' in {modality_plot_dir}/")

            self.plot_confusion_matrices(
                modality_results,
                modality_plot_dir / 'confusion_matrices.png',
                modality
            )
            self.plot_model_comparison(
                modality_results,
                modality_plot_dir / 'model_comparison.png',
                modality
            )
            self.plot_roc_curves(
                modality_results,
                modality_plot_dir / 'roc_curves.png',
                modality
            )
        
        self.generate_report()

    def plot_confusion_matrices(self, results_to_plot, save_path, modality_name):
        """Plot confusion matrices for all models of a given modality."""
        n_models = len(results_to_plot)
        if n_models == 0:
            return
        cols = min(4, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle(f'Confusion Matrices for {modality_name.upper()} Models', fontsize=16, y=1.02)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, results) in enumerate(results_to_plot.items()):
            try:
                model_name_short = name.replace(f"{modality_name}_", "")
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
                axes[idx].set_title(f'{model_name_short}\nAccuracy: {results["accuracy"]:.3f}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            except Exception as e:
                logger.warning(f"Error plotting confusion matrix for {name}: {e}")
        
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path)
        plt.close()

    def plot_model_comparison(self, results_to_plot, save_path, modality_name):
        """Plot model performance comparison for a given modality."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = [key.replace(f"{modality_name}_", "") for key in results_to_plot.keys()]
        
        # Replace None values (e.g., roc_auc=None for one-class models) with sensible defaults for plotting
        metric_data = {}
        for metric in metrics:
            vals = []
            for key in results_to_plot.keys():
                v = results_to_plot[key].get(metric, 0)
                if v is None:
                    # For roc_auc, use 0.5 baseline; for others, use 0
                    v = 0.5 if metric == 'roc_auc' else 0
                vals.append(v)
            metric_data[metric] = vals
        
        x = np.arange(len(model_names))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, metric_data[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title(f'Model Performance Comparison for {modality_name.upper()} Data')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curves(self, results_to_plot, save_path, modality_name):
        """Plot ROC curves for all models of a given modality."""
        plt.figure(figsize=(12, 8))
        
        for name, results in results_to_plot.items():
            model_name_short = name.replace(f"{modality_name}_", "")
            if 'y_pred_proba' in results and results['y_pred_proba'] is not None and 'y_test' in results and len(np.unique(results['y_test'])) >= 2:
                try:
                    fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
                    auc_score = results.get('roc_auc')
                    auc_label = f"{auc_score:.3f}" if auc_score is not None else 'n/a'
                    plt.plot(fpr, tpr, label=f'{model_name_short} (AUC = {auc_label})')
                except Exception as e:
                    logger.warning(f"Could not plot ROC for {name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {modality_name.upper()} Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path)
        plt.close()

    def generate_report(self):
        """Generate comprehensive training report"""
        report_lines = [
            "=" * 80,
            "LVH DETECTION MODEL TRAINING REPORT",
            "=" * 80,
            "Per-modality test set sizes and class balance:",
            "",
            "MODEL PERFORMANCE RESULTS:",
            "-" * 50
        ]
        
        # Add results for each model
        if not self.results:
            report_lines.append("\nNo models were successfully trained. Please review the training logs.")
        else:
            for name, results in self.results.items():
                roc_val = results.get('roc_auc')
                roc_str = f"{roc_val:.4f}" if roc_val is not None else 'n/a'
                report_lines.extend([
                    f"\n{name.upper()}:",
                    f"   Accuracy:     {results.get('accuracy', 0):.4f}",
                    f"   Precision:    {results.get('precision', 0):.4f}",
                    f"   Recall:       {results.get('recall', 0):.4f}",
                    f"   F1-Score:     {results.get('f1_score', 0):.4f}",
                    f"   ROC-AUC:      {roc_str}"
                ])

                if 'y_test' in results:
                    y_test_mod = results['y_test']
                    report_lines.extend([
                        f"   Test samples: {len(y_test_mod)}",
                        f"   Positives:    {int(np.sum(y_test_mod))} ({(np.sum(y_test_mod)/len(y_test_mod)*100):.1f}%)",
                        f"   Negatives:    {int(len(y_test_mod)-np.sum(y_test_mod))} ({((len(y_test_mod)-np.sum(y_test_mod))/len(y_test_mod)*100):.1f}%)",
                    ])
                    if 'threshold' in results:
                        report_lines.append(f"   Threshold:    {results['threshold']:.2f}")
        
        if self.results:
            best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
            report_lines.extend([
                "",
                f"BEST MODEL: {best_model_name}",
                f"BEST ACCURACY: {self.best_score:.4f}",
            ])
        else:
            report_lines.append("")

        if self.thresholds:
            report_lines.extend([
                "",
                "MODEL THRESHOLDS:",
                "-" * 50
            ])
            for model_name, threshold in sorted(self.thresholds.items()):
                report_lines.append(f"   {model_name:<40} {threshold:.2f}")

        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = MODELS_DIR / 'training_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        logger.info(f"Training report saved: {report_path}")

def main():
    """Main function"""
    trainer = LVHModelTrainer()
    trainer.train_multimodal_pipeline()
    
    print(f"\n🎉 Training Complete!")
    print(f"Best Model Accuracy: {trainer.best_score:.4f}")
    print(f"Models and results saved in: {MODELS_DIR}")
    print(f"Ready to run: python run.py")

if __name__ == "__main__":
    main()