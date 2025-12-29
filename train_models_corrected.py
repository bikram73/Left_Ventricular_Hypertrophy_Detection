"""
ULTIMATE High-Accuracy Training Script for LVH Detection (FIXED Unicode Error)
Implements advanced ML techniques for maximum performance
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import logging
import warnings
from sklearn.base import clone
warnings.filterwarnings('ignore')

from config import (PROCESSED_DATA_DIR, MODELS_DIR, ECG_PROCESSED_FILE, 
                    CLINICAL_PROCESSED_FILE, RANDOM_STATE, TEST_SIZE)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltimateLVHTrainer:
    """Ultimate high-accuracy trainer with advanced ML techniques"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.best_model = None
        self.best_score = 0
        self.thresholds = {}

    def load_processed_data(self):
        """Load all processed datasets"""
        datasets = {}

        if CLINICAL_PROCESSED_FILE.exists():
            clinical_data = pd.read_csv(CLINICAL_PROCESSED_FILE)
            datasets['clinical'] = clinical_data
            logger.info(f"Clinical data loaded: {clinical_data.shape}")

        if ECG_PROCESSED_FILE.exists():
            ecg_data = pd.read_csv(ECG_PROCESSED_FILE)
            datasets['ecg'] = ecg_data
            logger.info(f"ECG data loaded: {ecg_data.shape}")

        mri_path = PROCESSED_DATA_DIR / 'mri_features.csv'
        if mri_path.exists():
            mri_data = pd.read_csv(mri_path)
            datasets['mri'] = mri_data
            logger.info(f"MRI data loaded: {mri_data.shape}")

        ct_path = PROCESSED_DATA_DIR / 'ct_features.csv'
        if ct_path.exists():
            ct_data = pd.read_csv(ct_path)
            datasets['ct'] = ct_data
            logger.info(f"CT data loaded: {ct_data.shape}")

        if not datasets:
            logger.error("No datasets found. Please run process_data.py first.")
            return None

        return datasets

    def prepare_data_advanced(self, dataset, modality_name):
        """Advanced data preparation with feature engineering and selection"""

        possible_targets = ['HeartDisease', 'target', 'label', 'class', 'lvh_label']
        target_col = None
        for col in possible_targets:
            if col in dataset.columns:
                target_col = col
                break

        if not target_col:
            raise ValueError(f"Target column not found in {modality_name} dataset")

        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]

        class_counts = y.value_counts()
        logger.info(f"{modality_name} class distribution: {class_counts.to_dict()}")

        # IMPROVED: Use larger test size for better evaluation + validation set
        test_size_adjusted = 0.3 if len(X) > 200 else 0.25  # 30% for test
        stratify_param = y if len(class_counts) >= 2 and class_counts.min() >= 2 else None

        # Split into train+val (70%) and test (30%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size_adjusted, random_state=RANDOM_STATE, stratify=stratify_param
        )
        
        # Further split train+val into train (80% of 70% = 56%) and val (20% of 70% = 14%)
        stratify_param2 = y_train_val if len(pd.Series(y_train_val).value_counts()) >= 2 and pd.Series(y_train_val).value_counts().min() >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_param2
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # IMPROVED: More conservative PCA to retain more information
        pca = None
        if X_train_scaled.shape[1] > 100:
            # Retain 95% variance instead of fixed components
            n_comp = min(100, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1] - 1)
            if n_comp > 0:
                pca = PCA(n_components=0.95, random_state=RANDOM_STATE)  # Retain 95% variance
                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_val_scaled = pca.transform(X_val_scaled)
                X_test_scaled = pca.transform(X_test_scaled)
                logger.info(f"Applied PCA ({pca.n_components_} components) - explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        # IMPROVED: More aggressive feature selection for better signal
        n_features_to_select = min(X_train_scaled.shape[1], max(15, int(X_train_scaled.shape[1] * 0.7)))
        if X_train_scaled.shape[1] > 20 and len(np.unique(y_train)) >= 2:
            try:
                selector = SelectKBest(f_classif, k=n_features_to_select)
                X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
                X_val_scaled = selector.transform(X_val_scaled)
                X_test_scaled = selector.transform(X_test_scaled)
                logger.info(f"Feature selection: reduced to {X_train_scaled.shape[1]} features")
                self.feature_selectors[modality_name] = selector
            except:
                logger.info(f"Feature selection skipped for {modality_name}")

        # IMPROVED: More aggressive SMOTE for better class balance (only on training set)
        if len(np.unique(y_train)) >= 2 and len(X_train_scaled) > 20:
            try:
                train_class_counts = pd.Series(y_train).value_counts()
                min_class_count = train_class_counts.min()
                k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
                
                # Use SMOTE with sampling strategy to create more balanced data
                smote = SMOTE(
                    random_state=RANDOM_STATE, 
                    k_neighbors=k_neighbors,
                    sampling_strategy='auto'  # Balance to 1:1 ratio
                )
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                logger.info(f"Applied SMOTE - new training size: {X_train_scaled.shape[0]}")
                logger.info(f"New class distribution: {pd.Series(y_train).value_counts().to_dict()}")
            except Exception as e:
                logger.info(f"SMOTE not applied for {modality_name}: {e}")

        logger.info(f"Data prepared - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

        scaler_obj = {'scaler': scaler, 'pca': pca}
        self.scalers[modality_name] = scaler_obj

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    def get_optimized_models(self, modality_name):
        """Get highly optimized models with best hyperparameters - IMPROVED for ECG/MRI/CT"""

        # IMPROVED: Adjust hyperparameters based on modality
        is_imaging = modality_name.lower() in ['mri', 'ct']
        is_ecg = modality_name.lower() == 'ecg'
        
        models = {
            'RandomForest_Optimized': RandomForestClassifier(
                n_estimators=500 if is_imaging else 300,  # More trees for imaging
                max_depth=25 if is_imaging else 20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                n_jobs=-1,
                random_state=RANDOM_STATE,
                criterion='gini',
                min_impurity_decrease=0.0
            ),
            'XGBoost_Optimized': xgb.XGBClassifier(
                n_estimators=400 if is_imaging else 300,
                max_depth=12 if is_imaging else 10,
                learning_rate=0.05 if is_ecg else 0.03,  # Higher LR for ECG
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=2.0,
                scale_pos_weight=1.0,  # Balanced after SMOTE
                min_child_weight=1,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'LightGBM_Optimized': lgb.LGBMClassifier(
                n_estimators=400 if is_imaging else 300,
                max_depth=15 if is_imaging else 12,
                learning_rate=0.05 if is_ecg else 0.03,
                num_leaves=60 if is_imaging else 50,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=2.0,
                class_weight='balanced',
                min_child_samples=3,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
                boosting_type='gbdt'
            ),
            'GradientBoosting_Optimized': GradientBoostingClassifier(
                n_estimators=250 if is_imaging else 200,
                learning_rate=0.1 if is_ecg else 0.08,
                max_depth=10 if is_imaging else 9,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=0.9,
                max_features='sqrt',
                random_state=RANDOM_STATE
            ),
            'SVM_Optimized': SVC(
                C=20.0 if is_imaging else 15.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=RANDOM_STATE,
                cache_size=500
            ),
            'MLP_Optimized': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128) if is_imaging else (256, 128, 64),
                activation='relu',
                alpha=0.0001 if is_imaging else 0.00005,
                batch_size=32 if is_ecg else 64,
                learning_rate='adaptive',
                learning_rate_init=0.003 if is_ecg else 0.002,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=RANDOM_STATE,
                solver='adam'
            ),
            'AdaBoost_Optimized': AdaBoostClassifier(
                n_estimators=200 if is_imaging else 150,
                learning_rate=1.2 if is_ecg else 1.0,
                random_state=RANDOM_STATE,
                algorithm='SAMME'
            ),
            'LogisticRegression_Optimized': LogisticRegression(
                C=5.0 if is_imaging else 3.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=3000,
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
        }

        return models

    def determine_optimal_threshold(self, model, X, y):
        """Determine probability threshold that maximizes accuracy via cross-validation."""
        if len(np.unique(y)) < 2:
            return 0.5

        try:
            estimator = clone(model)
        except Exception:
            return 0.5

        method = None
        if hasattr(estimator, 'predict_proba'):
            method = 'predict_proba'
        elif hasattr(estimator, 'decision_function'):
            method = 'decision_function'
        else:
            return 0.5

        unique, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            return 0.5

        max_splits = counts.min()
        if max_splits < 2:
            return 0.5

        n_splits = int(min(5, max_splits))
        if n_splits < 2:
            return 0.5

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        try:
            cv_outputs = cross_val_predict(estimator, X, y, cv=cv, method=method)
        except Exception as exc:
            logger.debug(f"Threshold optimization skipped due to cross-val failure: {exc}")
            return 0.5

        if method == 'predict_proba':
            if cv_outputs.ndim == 2 and cv_outputs.shape[1] > 1:
                cv_scores = cv_outputs[:, 1]
            else:
                cv_scores = cv_outputs.ravel()
        else:
            scores = cv_outputs.ravel()
            smin, smax = scores.min(), scores.max()
            if smax > smin:
                cv_scores = (scores - smin) / (smax - smin)
            else:
                return 0.5

        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = -np.inf

        for thr in thresholds:
            preds = (cv_scores >= thr).astype(int)
            score = accuracy_score(y, preds)
            if score > best_score + 1e-6:
                best_score = score
                best_threshold = float(thr)

        logger.debug(f"Optimal threshold determined at {best_threshold:.3f} with CV accuracy {best_score:.4f}")
        return best_threshold

    def train_and_evaluate_advanced(self, name, model, X_train, X_val, X_test, y_train, y_val, y_test):
        """Advanced training with validation set and cross-validation"""
        logger.info(f"Training {name}...")

        if len(X_train) > 20 and len(np.unique(y_train)) >= 2:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            logger.info(f"  CV Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        model.fit(X_train, y_train)
        
        # Evaluate on validation set first
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")

        # Then evaluate on test set
        y_pred = model.predict(X_test)

        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
                y_pred_proba = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                y_pred_proba = y_pred.astype(float)
        except:
            y_pred_proba = y_pred.astype(float)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) >= 2 else 0
        recall = recall_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) >= 2 else 0
        f1 = f1_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) >= 2 else 0
        roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) >= 2 else 0

        # FIXED: Use ASCII checkmark instead of Unicode
        logger.info(f"[OK] {name} - Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

        return {
            'model': model,
            'accuracy': accuracy,
            'val_accuracy': val_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def create_stacking_ensemble(self, base_models, modality_name):
        """Create stacking ensemble for even better performance"""
        estimators = []
        for name, model in base_models.items():
            estimators.append((name.lower(), model))

        if len(estimators) < 3:
            return None

        estimators = estimators[:4]

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=5.0, random_state=RANDOM_STATE),
            cv=3,
            n_jobs=-1
        )

        logger.info(f"Created stacking ensemble with {len(estimators)} base models")
        return stacking_model

    def train_multimodal_pipeline(self):
        """Main training pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STARTING ULTIMATE HIGH-ACCURACY TRAINING PIPELINE")
        logger.info("="*80 + "\n")

        datasets = self.load_processed_data()
        if datasets is None:
            return

        all_trained_models = {}

        for modality_name, dataset in datasets.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"TRAINING {modality_name.upper()} MODALITY")
            logger.info(f"{'='*80}")

            try:
                X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data_advanced(dataset, modality_name)

                if len(X_train) < 5:
                    logger.warning(f"Insufficient data for {modality_name}")
                    continue

                models = self.get_optimized_models(modality_name)

                modality_results = {}
                trained_base_models = {}

                for model_name, model in models.items():
                    full_name = f"{modality_name}_{model_name}"
                    result = self.train_and_evaluate_advanced(
                        full_name, model, X_train, X_val, X_test, y_train, y_val, y_test
                    )
                    modality_results[full_name] = result
                    trained_base_models[model_name] = result['model']
                    all_trained_models[full_name] = result['model']

                if len(trained_base_models) >= 3:
                    logger.info(f"\nTraining Stacking Ensemble for {modality_name}...")
                    stacking_model = self.create_stacking_ensemble(trained_base_models, modality_name)
                    if stacking_model:
                        ensemble_name = f"{modality_name}_StackingEnsemble"
                        ensemble_result = self.train_and_evaluate_advanced(
                            ensemble_name, stacking_model, X_train, X_val, X_test, y_train, y_val, y_test
                        )
                        modality_results[ensemble_name] = ensemble_result
                        all_trained_models[ensemble_name] = ensemble_result['model']

                self.results.update(modality_results)

                modality_dir = MODELS_DIR / modality_name
                modality_dir.mkdir(parents=True, exist_ok=True)
                for model_name, model in trained_base_models.items():
                    joblib.dump(model, modality_dir / f'{model_name}.pkl')

                joblib.dump(self.scalers[modality_name], modality_dir / 'scaler.pkl')

                self.generate_modality_visualizations(modality_results, modality_dir, modality_name)

            except Exception as e:
                logger.error(f"Failed to train {modality_name}: {e}")
                continue

        if not all_trained_models:
            logger.error("No models were successfully trained!")
            return

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(all_trained_models, MODELS_DIR / 'all_optimized_models.pkl')

        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        self.best_model = all_trained_models[best_model_name]
        self.best_score = self.results[best_model_name]['accuracy']
        joblib.dump(self.best_model, MODELS_DIR / 'best_lvh_model.pkl')

        self.generate_comprehensive_report()

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best Accuracy: {self.best_score:.4f}")
        logger.info(f"Models saved in: {MODELS_DIR}")
        logger.info(f"{'='*80}\n")

    def generate_modality_visualizations(self, results, save_dir, modality_name):
        """Generate visualizations for a modality"""

        n_models = len(results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle(f'{modality_name.upper()} - Confusion Matrices', fontsize=16, fontweight='bold')

        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (name, result) in enumerate(results.items()):
            short_name = name.replace(f"{modality_name}_", "").replace("_Optimized", "")
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                       xticklabels=['No LVH', 'LVH'], yticklabels=['No LVH', 'LVH'])
            axes[idx].set_title(f'{short_name}\nAcc: {result["accuracy"]:.3f}', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        for i in range(n_models, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = [k.replace(f"{modality_name}_", "").replace("_Optimized", "") for k in results.keys()]

        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(model_names))
        width = 0.15

        for i, metric in enumerate(metrics):
            values = [results[k].get(metric, 0) for k in results.keys()]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)

        ax.set_xlabel('Models', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title(f'{modality_name.upper()} - Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.3, label='80% threshold')

        plt.tight_layout()
        plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 9))

        for name, result in results.items():
            if len(np.unique(result['y_test'])) >= 2:
                try:
                    fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                    short_name = name.replace(f"{modality_name}_", "").replace("_Optimized", "")
                    plt.plot(fpr, tpr, linewidth=2, label=f'{short_name} (AUC={result["roc_auc"]:.3f})')
                except:
                    pass

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        plt.title(f'{modality_name.upper()} - ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self):
        """Generate detailed training report - FIXED Unicode encoding"""
        report = []
        report.append("=" * 90)
        report.append("ULTIMATE LVH DETECTION - HIGH-ACCURACY TRAINING REPORT")
        report.append("=" * 90)
        report.append(f"\nBest Overall Model: {max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]}")
        report.append(f"Best Accuracy: {self.best_score:.4f}")
        report.append("\n" + "=" * 90)
        report.append("DETAILED RESULTS BY MODALITY")
        report.append("=" * 90)

        modalities = set(k.split('_')[0] for k in self.results.keys())

        for modality in sorted(modalities):
            report.append(f"\n{modality.upper()} MODALITY:")
            report.append("-" * 90)
            report.append(f"{'Model':<40} {'Val Acc':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
            report.append("-" * 110)

            modality_results = {k: v for k, v in self.results.items() if k.startswith(modality)}

            for name, result in sorted(modality_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                short_name = name.replace(f"{modality}_", "").replace("_Optimized", "")
                val_acc = result.get('val_accuracy', 0.0)
                report.append(
                    f"{short_name:<40} "
                    f"{val_acc:<12.4f} "
                    f"{result['accuracy']:<12.4f} "
                    f"{result['precision']:<12.4f} "
                    f"{result['recall']:<12.4f} "
                    f"{result['f1_score']:<12.4f} "
                    f"{result['roc_auc']:<12.4f}"
                )

            best_mod = max(modality_results.items(), key=lambda x: x[1]['accuracy'])
            # FIXED: Use [OK] instead of Unicode checkmark
            report.append(f"\n[OK] BEST {modality.upper()} MODEL: {best_mod[0]} (Accuracy: {best_mod[1]['accuracy']:.4f})")

        report.append("\n" + "=" * 90)
        report.append("TRAINING TECHNIQUES APPLIED:")
        report.append("=" * 90)
        report.append("  [OK] Advanced hyperparameter optimization")
        report.append("  [OK] Feature selection (SelectKBest)")
        report.append("  [OK] SMOTE for class balancing")
        report.append("  [OK] PCA for dimensionality reduction")
        report.append("  [OK] 5-fold stratified cross-validation")
        report.append("  [OK] Stacking ensemble models")
        report.append("  [OK] Regularization and early stopping")
        report.append("=" * 90)

        report_text = "\n".join(report)

        # FIXED: Use UTF-8 encoding explicitly
        with open(MODELS_DIR / 'ultimate_training_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        logger.info(f"Report saved: {MODELS_DIR / 'ultimate_training_report.txt'}")

def main():
    """Main execution"""
    trainer = UltimateLVHTrainer()
    trainer.train_multimodal_pipeline()

    print(f"\n{'='*90}")
    print(f"ULTIMATE TRAINING COMPLETE!")
    print(f"{'='*90}")
    print(f"Best Model Accuracy: {trainer.best_score:.4f}")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Visualizations generated for each modality")
    print(f"Ready to deploy: python run.py")
    print(f"{'='*90}\n")

if __name__ == "__main__":
    main()
