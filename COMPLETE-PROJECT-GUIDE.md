# 🏥 Complete LVH Detection System - Comprehensive Guide

**Version:** 2.2.0  
**Last Updated:** November 22, 2025  
**Status:** ✅ Production Ready

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Complete Directory Structure](#complete-directory-structure)
3. [System Requirements](#system-requirements)
4. [Installation Guide](#installation-guide)
5. [Data Pipeline](#data-pipeline)
6. [Model Training](#model-training)
7. [Web Application](#web-application)
8. [API Documentation](#api-documentation)
9. [Database Architecture & Analytics System](#database-architecture--analytics-system)
10. [Testing Guide](#testing-guide)
11. [Performance Metrics](#performance-metrics)
12. [Troubleshooting](#troubleshooting)
13. [Academic Integration](#academic-integration)

---

## 🎯 Project Overview

### What is This System?

The **LVH Detection System** is a comprehensive AI-powered medical diagnostic tool that detects Left Ventricular Hypertrophy (LVH) using multiple data modalities:

- **Clinical Data**: Patient demographics, vital signs, risk factors
- **ECG Signals**: Electrocardiogram analysis with Sokolow-Lyon criteria
- **MRI Images**: Cardiac magnetic resonance imaging
- **CT Scans**: Computed tomography cardiac imaging

### Key Features

✅ **Multi-Modal Analysis**: Supports 4 different data types  
✅ **9 ML Algorithms**: Random Forest, XGBoost, LightGBM, GradientBoosting, SVM, Neural Networks (MLP), AdaBoost, Logistic Regression, Stacking Ensemble  
✅ **Interactive Web Interface**: Professional UI with real-time predictions  
✅ **Smart Validation**: Clinical data optional when files uploaded  
✅ **Confidence Scoring**: 0-100% confidence with each prediction  
✅ **Comprehensive Visualizations**: Charts, graphs, and performance metrics  
✅ **API Endpoints**: RESTful API for integration  

### Performance Summary

| Modality | Best Model | Accuracy | ROC-AUC | Status |
|----------|-----------|----------|---------|--------|
| Clinical | GradientBoosting | 89.13% | 0.9411 | ✅ Excellent |
| ECG | XGBoost | 82.00% | 0.9024 | ✅ Excellent |
| MRI | SVM | 81.43% | 0.8505 | ✅ Good |
| CT | Stacking Ensemble | 78.80% | 0.8477 | ✅ Good |

---


## 📁 Complete Directory Structure

```
lvh-detection/
│
├── 📄 Core Application Files
│   ├── run.py                          # Main entry point - START HERE
│   ├── app.py                          # Flask web application (2000+ lines)
│   ├── config.py                       # Configuration and paths
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # Project documentation
│
├── 📊 Data Processing
│   ├── download_data.py                # Kaggle dataset downloader
│   ├── process_data.py                 # Feature extraction pipeline
│   ├── process_data_corrected.py       # Fixed version
│   ├── process_ecg_only.py            # ECG-specific processing
│   └── create_input_samples.py        # Sample data generator
│
├── 🧠 Model Training
│   ├── train_models.py                 # Main training script
│   ├── train_models_corrected.py      # Fixed version
│   ├── train_models_improved.py       # Optimized version
│   ├── train_ecg_quick.py             # Quick ECG training
│   └── run_fixed_training.py          # Training runner
│
├── 🔮 Prediction & Testing
│   ├── predict_single.py              # Single file prediction
│   ├── batch_predict_all.py           # Batch predictions
│   ├── test_web_app.py                # Web app tests
│   ├── test_enhanced_ui.py            # UI tests
│   ├── test_clinical_validation.py    # Validation tests
│   └── test_fixes.py                  # Fix verification
│
├── 📊 Analysis & Reporting
│   ├── generate_performance_report.py  # Performance analysis
│   ├── compare_results.py             # Model comparison
│   ├── setup_enhanced_system.py       # System setup
│   ├── dashboard_service.py           # 📊 NEW: Analytics backend service
│   ├── metrics_collector.py           # 📊 NEW: Background metrics collection
│   ├── dashboard_metrics.db           # 📊 NEW: System metrics database
│   ├── predictions_history.db         # 📊 NEW: Predictions tracking database
│   └── fix_analytics_display.py       # 📊 NEW: Analytics display fixes
│
├── 📁 data/                           # Data directory
│   ├── raw/                           # Raw Kaggle datasets
│   │   ├── ecg/                       # ECG heartbeat data
│   │   ├── mri/                       # MRI cardiac images
│   │   ├── ct/                        # CT heart scans
│   │   └── clinical/                  # Clinical datasets
│   ├── processed/                     # Processed features
│   │   ├── ecg_features.csv          # ECG features
│   │   ├── mri_features.csv          # MRI features
│   │   ├── ct_features.csv           # CT features
│   │   └── clinical_processed.csv    # Clinical features
│   └── splits/                        # Train/test splits
│
├── 📁 input_data/                     # Sample input files
│   ├── ecg/                           # Sample ECG files
│   │   ├── patient_001_ecg_features.csv
│   │   ├── patient_002_ecg_signal.csv
│   │   └── ...
│   ├── mri/                           # Sample MRI images
│   │   ├── patient_002_mri.dcm
│   │   └── ...
│   ├── ct/                            # Sample CT images
│   │   ├── patient_010_ct.dcm
│   │   └── ...
│   ├── clinical.txt                   # Clinical data format
│   └── README.md                      # Input data guide
│
├── 📁 patient_ecg_data/              # Generated patient data
│   ├── patient1.csv to patient4.csv  # Individual patients
│   └── predection.txt                # Prediction results
│
├── 📁 models/                         # Trained models
│   ├── clinical/                      # Clinical models
│   │   ├── RandomForest_Optimized.pkl
│   │   ├── XGBoost_Optimized.pkl
│   │   ├── LightGBM_Optimized.pkl
│   │   ├── GradientBoosting_Optimized.pkl
│   │   ├── SVM_Optimized.pkl
│   │   ├── MLP_Optimized.pkl
│   │   ├── AdaBoost_Optimized.pkl
│   │   ├── LogisticRegression_Optimized.pkl
│   │   ├── scaler.pkl
│   │   ├── confusion_matrices.png
│   │   ├── model_comparison.png
│   │   └── roc_curves.png
│   ├── ecg/                           # ECG models (same structure)
│   ├── mri/                           # MRI models (same structure)
│   ├── ct/                            # CT models (same structure)
│   ├── scalers/                       # Feature scalers
│   │   └── feature_scalers.pkl
│   ├── best_lvh_model.pkl            # Best overall model
│   ├── all_models.pkl                # All trained models
│   ├── all_optimized_models.pkl      # Optimized models
│   ├── all_improved_models.pkl       # Improved models
│   ├── ecg_clinical_stacking.pkl     # Stacking ensemble
│   ├── model_thresholds.json         # Optimal thresholds
│   ├── training_report.txt           # Training results
│   ├── ultimate_training_report.txt  # Ultimate results
│   ├── improved_training_report.txt  # Improved results
│   ├── super_optimized_training_report.txt  # Super optimized results
│   ├── confusion_matrices.png        # Confusion matrices
│   ├── performance_chart.png         # Performance comparison
│   └── lvh_pipeline.png              # Pipeline visualization
│
├── 📁 templates/                      # HTML templates
│   ├── index.html                     # Home page
│   ├── upload.html                    # Upload interface
│   ├── results.html                   # Results display
│   ├── analytics.html                 # 📊 Analytics dashboard
│   ├── help.html                      # 📚 NEW: Help & user guide
│   ├── document.html                  # documentation
│   └── api.html                       # api
│
├── 📁 static/                         # Static assets
│   ├── css/                           # Stylesheets
│   │   ├── style.css                  # Main styles
│   │   └── dashboard.css              # 📊 NEW: Dashboard styles
│   ├── js/                            # JavaScript
│   │   ├── main.js                    # Main JavaScript
│   │   └── dashboard.js               # 📊 NEW: Dashboard interactions
│   └── uploads/                       # File uploads
│
├── 📁 visualizations/                 # Generated charts
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── *_feature_importance.png      # Feature importance plots
│
├── 📁 performance_reports/            # Performance analysis
│   ├── clinical_analysis.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── roc_curve.png
│   └── performance_report.txt
```

---


## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.14+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Disk Space**: 5GB free space
- **Internet**: Required for initial setup and Kaggle downloads

### Recommended Requirements
- **OS**: Windows 11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.9 or 3.10
- **RAM**: 16GB
- **Disk Space**: 10GB free space
- **GPU**: Optional (CUDA-compatible for faster training)

### Software Dependencies

**Core Libraries:**
```
Flask==2.3.0                 # Web framework
numpy==1.24.3               # Numerical computing
pandas==2.0.2               # Data manipulation
scikit-learn==1.3.0         # Machine learning
xgboost==1.7.6              # Gradient boosting
lightgbm==4.0.0             # Light gradient boosting
tensorflow==2.13.0          # Deep learning
```

**Data Processing:**
```
scipy==1.11.1               # Scientific computing
opencv-python==4.8.0        # Image processing
pydicom==2.4.2              # DICOM file handling
Pillow==10.0.0              # Image manipulation
```

**Visualization:**
```
matplotlib==3.7.2           # Plotting
seaborn==0.12.2             # Statistical visualization
plotly==5.15.0              # Interactive plots
```

**Web & API:**
```
werkzeug==2.3.6             # WSGI utilities
jinja2==3.1.2               # Template engine
requests==2.31.0            # HTTP library
```

**Utilities:**
```
joblib==1.3.1               # Model serialization
tqdm==4.65.0                # Progress bars
kaggle==1.5.16              # Kaggle API
```

---

## 🚀 Installation Guide

### Step 1: Clone or Download Project

```bash
# Option 1: If using Git
git clone <repository-url>
cd lvh-detection

# Option 2: Download ZIP and extract
# Then navigate to the folder
cd lvh-detection
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import flask, numpy, pandas, sklearn, xgboost; print('✓ All core libraries installed')"
```

### Step 4: Configure Kaggle API (Optional)

**For downloading real datasets:**

1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it in the correct location:

**Windows:**
```bash
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\
```

**Linux/Mac:**
```bash
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 5: Verify Installation

```bash
# Test import
python -c "from app import app; print('✓ App imports successfully')"

# Check models directory
python -c "from pathlib import Path; print('✓ Models found' if Path('models').exists() else '✗ Models missing')"
```

---

## 📊 Data Pipeline

### Data Flow Architecture

```
Raw Data Sources
       ↓
┌──────────────────────────────────────┐
│  1. Data Download (download_data.py) │
│     - Kaggle API integration         │
│     - Dataset validation             │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  2. Feature Extraction               │
│     (process_data.py)                │
│     - ECG: Signal processing         │
│     - MRI: Image features            │
│     - CT: Texture analysis           │
│     - Clinical: Data cleaning        │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  3. Data Preprocessing               │
│     - Normalization                  │
│     - Feature scaling                │
│     - Train/test split               │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  4. Model Training                   │
│     (train_models.py)                │
│     - 9 algorithms × 4 modalities    │
│     - Cross-validation               │
│     - Hyperparameter tuning          │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  5. Model Evaluation                 │
│     - Performance metrics            │
│     - Confusion matrices             │
│     - ROC curves                     │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  6. Deployment                       │
│     (app.py)                         │
│     - Web interface                  │
│     - API endpoints                  │
│     - Real-time predictions          │
└──────────────────────────────────────┘
```

### Data Processing Details

#### 1. ECG Processing
**File:** `process_data.py` - ECG section

**Features Extracted:**
- R-peak detection and intervals
- Heart rate variability (HRV)
- QRS complex duration
- PR and QT intervals
- Sokolow-Lyon voltage (key LVH indicator)
- Cornell voltage criteria
- Frequency domain features
- Signal quality metrics

**Code Example:**
```python
# ECG feature extraction
def extract_ecg_features(signal):
    features = {}
    
    # R-peak detection
    r_peaks = detect_r_peaks(signal)
    
    # Sokolow-Lyon voltage
    features['sokolow_lyon'] = calculate_sokolow_lyon(signal, r_peaks)
    
    # Heart rate
    features['heart_rate'] = 60 / np.mean(np.diff(r_peaks))
    
    return features
```

#### 2. MRI Processing
**File:** `process_data.py` - MRI section

**Features Extracted:**
- Texture features (GLCM)
- Shape descriptors
- Intensity statistics
- Edge detection
- Cardiac chamber segmentation
- Wall thickness measurements

#### 3. CT Processing
**File:** `process_data.py` - CT section

**Features Extracted:**
- Density analysis (Hounsfield units)
- Morphological features
- Edge characteristics
- Texture patterns
- Cardiac calcium scoring

#### 4. Clinical Processing
**File:** `process_data.py` - Clinical section

**Features Used:**
- Age, Gender
- Blood Pressure (Resting, Max)
- Cholesterol levels
- Chest Pain Type
- Exercise-induced Angina
- ST Depression (Oldpeak)
- ST Slope
- Fasting Blood Sugar
- Resting ECG results

---


## 🧠 Model Training

### Training Pipeline

**Main Script:** `train_models.py`

**Training Process:**
1. Load processed features
2. Apply SMOTE for class balancing
3. Feature selection (SelectKBest)
4. Split data (80% train, 20% test)
5. Train 9 algorithms per modality (36 total models)
6. 5-fold stratified cross-validation
7. Hyperparameter optimization
8. Evaluate performance with multiple metrics
9. Save best models with optimal thresholds
10. Generate visualizations and reports

### Algorithms Implemented

| Algorithm | Type | Best For | Best Modality |
|-----------|------|----------|---------------|
| Random Forest | Ensemble | Robust, interpretable | All modalities |
| XGBoost | Gradient Boosting | High accuracy | ECG (82%) |
| LightGBM | Gradient Boosting | Fast training | All modalities |
| GradientBoosting | Gradient Boosting | Highest accuracy | Clinical (89.13%) |
| SVM (RBF) | Kernel Method | Non-linear patterns | MRI (81.43%) |
| MLP (Neural Network) | Deep Learning | Complex patterns | All modalities |
| AdaBoost | Boosting | Weak learners | All modalities |
| Logistic Regression | Linear | Baseline | CT (78.75%) |
| Stacking Ensemble | Meta-learning | Combined predictions | CT (78.80%) |

### Training Commands

```bash
# Full training (all modalities)
python train_models.py

# Quick ECG training
python train_ecg_quick.py

# Improved training with optimization
python train_models_improved.py

# Fixed version
python train_models_corrected.py
```

### Model Performance

**Clinical Models:**
- Best: GradientBoosting (89.13% accuracy, 0.9411 ROC-AUC)
- Runner-up: RandomForest (88.41% accuracy)
- Training time: ~2 minutes
- Features: 11 clinical parameters
- Techniques: SMOTE, Feature Selection, Cross-validation

**ECG Models:**
- Best: XGBoost (82.00% accuracy, 0.9024 ROC-AUC)
- Runner-up: GradientBoosting (78.00% accuracy)
- Training time: ~3 minutes
- Features: 19 ECG parameters
- Techniques: Signal processing, R-peak detection

**MRI Models:**
- Best: SVM (81.43% accuracy, 0.8505 ROC-AUC)
- Runner-up: LightGBM (80.00% accuracy)
- Training time: ~5 minutes
- Features: Image texture and shape features
- Techniques: GLCM, Edge detection

**CT Models:**
- Best: Stacking Ensemble (78.80% accuracy, 0.8477 ROC-AUC)
- Runner-up: LogisticRegression (78.75% accuracy)
- Training time: ~5 minutes
- Features: Density and texture features
- Techniques: Hounsfield units, Morphological analysis

---

## 🌐 Web Application & User Interface Architecture

### Starting the Application

```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Start the app
python run.py

# Access at: http://localhost:5000
```

### UI Architecture Overview

**Frontend Technologies:**
- Bootstrap 5.1.3 - Responsive framework
- Font Awesome 6.0 - Icon library  
- Chart.js - Interactive charts
- Prism.js - Code syntax highlighting
- Custom CSS - Gradient backgrounds, animations

**Backend Technologies:**
- Flask 2.3.0 - Web framework
- Jinja2 - Template engine
- Werkzeug - File handling
- Python ML Stack - Prediction engine
- RESTful API - JSON responses

### Complete Page Structure (8 Pages)

#### 1. Home Page (index.html) - Route: `/`

**Purpose:** Landing page with system overview and feature highlights

**Key Sections:**
- Hero section with gradient background
- System statistics (94.2% accuracy, 91.8% sensitivity, 95.9% specificity)
- Feature cards (ECG Analysis, Medical Imaging, Clinical Data, Ensemble Models)
- About section with project goals
- Quick navigation to all pages

**UI Features:**
- Animated gradient background (purple to blue)
- Hover effects on feature cards
- Responsive grid layout
- Fixed navigation bar
- Call-to-action buttons

#### 2. Upload Page (upload.html) - Route: `/upload`

**Purpose:** Interactive data upload interface with smart validation

**Clinical Data Form (11 Fields):**
1. Age (Required*) - Integer 1-120
2. Gender (Required*) - 0=Female, 1=Male
3. Chest Pain Type (Required*) - 0-3
4. Resting BP (Required*) - 50-300 mmHg
5. Cholesterol (Required*) - 100-600 mg/dl
6. Max Heart Rate (Required*) - 60-220 bpm
7. Fasting Blood Sugar - Normal/Elevated
8. Resting ECG - Normal/ST-T Wave/LVH
9. Exercise Angina - Yes/No
10. Oldpeak (ST Depression) - 0-10
11. ST Slope - Upsloping/Flat/Downsloping

*Required only if no files uploaded

**File Upload Sections:**
- **ECG:** CSV files (R-wave, HRV, QRS, Sokolow-Lyon)
- **MRI:** PNG/JPG/DICOM (Wall thickness, chamber volume)
- **CT:** PNG/JPG/DICOM (Density, texture, calcification)

**Smart Validation:**
- Option 1: Upload files - Clinical data optional
- Option 2: Fill 5 required fields - No files needed
- Real-time validation with visual feedback

#### 3. Results Page (results.html) - Route: `/results`

**Purpose:** Comprehensive prediction results with interactive visualizations

**Key Sections:**
- Main result card with confidence score
- Data sources summary (ECG/MRI/CT/Clinical indicators)
- Multi-modal predictions display
- Dynamic performance metrics
- Clinical data summary
- Interactive visualizations (5 tabs)
- Analysis details and recommendations

**Interactive Tabs:**
1. Multi-Modal Analysis - Comparison across modalities
2. Risk Factors Analysis - Bar chart with severity levels
3. Risk Profile - Radar chart (360° view)
4. Performance Metrics - Accuracy, Precision, Recall
5. ECG Analysis - Signal visualization (if uploaded)

#### 4. 📊 Analytics Dashboard Page (analytics.html) - Route: `/analytics`

**Purpose:** Real-time system analytics and prediction tracking dashboard

**Key Sections:**
- **System Status Overview**: Real-time health indicators
- **Prediction Analytics**: Total predictions, LVH detection rate, confidence trends
- **Data Type Distribution**: Usage breakdown by modality (ECG, MRI, CT, Clinical)
- **Performance Metrics**: Processing times, accuracy trends, system performance
- **Daily/Weekly Trends**: Prediction volume over time
- **Recent Predictions**: Latest prediction history with details
- **System Metrics**: CPU, RAM, disk usage monitoring

**Interactive Features:**
- **Real-time Updates**: All metrics update automatically with new predictions
- **Dynamic Charts**: Interactive charts using Chart.js
- **Time Period Filtering**: View data for different time ranges
- **Export Functionality**: Download analytics data as CSV/JSON
- **Responsive Design**: Mobile-friendly dashboard layout

**Analytics Cards:**
1. **Total Predictions** - Running count of all predictions
2. **LVH Detection Rate** - Percentage of positive LVH cases
3. **Average Confidence** - Mean confidence score across predictions
4. **Processing Time** - Average time per prediction
5. **Data Type Usage** - Distribution chart of modality usage
6. **Accuracy Trend** - 7-day rolling accuracy trend
7. **Daily Predictions** - Predictions per day chart
8. **System Health** - CPU, RAM, disk usage indicators

#### 5. 📚 Help Guide Page (help.html) - Route: `/help`

**Purpose:** Comprehensive user manual and troubleshooting guide

**Key Sections:**
- **Getting Started**: System overview, requirements, and quick start guide
- **Home Page Guide**: Navigation and feature explanations
- **Upload & Predict Guide**: Step-by-step upload process and input options
- **Understanding Results**: Detailed explanation of prediction results and confidence scores
- **Documentation Guide**: How to use the technical documentation
- **API Usage Guide**: REST API integration examples
- **System Health Guide**: Monitoring and status checking
- **Troubleshooting**: Common issues and solutions

**Interactive Features:**
- **Sidebar Navigation**: Sticky navigation with smooth scrolling
- **Dark Mode Toggle**: Light/dark theme switching
- **Step-by-Step Instructions**: Numbered guides with visual indicators
- **Screenshot Gallery**: Visual examples of each page
- **Quick Reference Cards**: File formats, performance metrics, navigation links
- **Color-coded Sections**: Info boxes, tip boxes, warning boxes

**Help Topics Covered:**
1. **File Upload Issues**: Format validation, size limits, troubleshooting
2. **Prediction Problems**: Data format requirements, processing times
3. **System Compatibility**: Browser requirements, JavaScript/cookies
4. **API Integration**: Endpoint usage, response formats
5. **Performance Optimization**: Best practices for accurate predictions

#### 6. Documentation Page (document.html) - Route: `/document`

**Purpose:** Complete system documentation with interactive navigation

**14 Sections:**
1. Project Overview
2. Directory Structure
3. System Requirements
4. Installation Guide
5. Data Pipeline (with animated visualization)
6. Model Training
7. Web Application
8. 📊 Analytics Dashboard (NEW)
9. API Documentation
10. Testing Guide
11. Performance Metrics
12. Training Visualizations (clickable gallery)
13. Troubleshooting
14. Academic Integration

**UI Features:**
- Sticky sidebar navigation
- Smooth scroll with active highlighting
- Code syntax highlighting
- Image modal viewer
- Print-friendly styles

#### 7. API Documentation Page (api.html) - Route: `/api`

**Purpose:** Complete REST API reference with code examples

**Key Sections:**
- API Overview
- Endpoints (POST /predict, GET /health, Analytics APIs)
- Code examples (Python, cURL, JavaScript)
- Error handling
- Model information cards

#### 8. Health Check Page (health.html) - Route: `/health`

**Purpose:** Real-time system health monitoring with interactive modals

**Interactive Cards:**
- Models Status (36 models) - Click for details
- Predictor Status - Click for pipeline info
- Available Modalities (4) - Click for features
- Individual modality cards (Clinical, ECG, MRI, CT)

**Modal System:**
- Click any card to open detailed modal
- Smooth animations and transitions
- Scrollable content
- ESC key or click outside to close

### Common UI Elements

**Navigation Bar:**
- Fixed to top with blur effect
- 7 links: Home, Predict, Analytics, Documentation, API, Health, Help
- Responsive hamburger menu (mobile)

**Color Scheme:**
- Primary: Purple (#667eea) to Blue (#764ba2) gradient
- Success: Green, Warning: Yellow, Danger: Red, Info: Cyan

**Responsive Design:**
- Mobile-first approach
- Breakpoints: 576px, 768px, 992px, 1200px
- Touch-friendly buttons (min 44px)

**Animations:**
- Fade-in effects (0.5s)
- Hover scale transforms
- Smooth scrolling
- Progress bar animations
- Pulsing indicators

### Technical Implementation

**Flask Routes:**
```python
@app.route('/')  # Home
@app.route('/upload')  # Upload interface
@app.route('/predict', methods=['POST'])  # Prediction API
@app.route('/results')  # Results display
@app.route('/analytics')  # Analytics dashboard
@app.route('/help')  # Help & user guide
@app.route('/document')  # Documentation
@app.route('/api')  # API docs
@app.route('/health')  # Health check
```

**File Upload Handling:**
- Werkzeug secure_filename()
- 16MB file size limit
- MIME type checking
- Temporary storage with auto-cleanup

**Session Management:**
- Flask session for results
- Flash messages for errors
- Temporary file paths

### User Workflows

**Workflow 1: Clinical Data Only**
1. Navigate to /upload
2. Fill 5 required fields
3. Click "Analyze LVH Risk"
4. View results

**Workflow 2: File Upload Only**
1. Navigate to /upload
2. Upload ECG/MRI/CT file
3. Click "Analyze LVH Risk"
4. View file-specific analysis

**Workflow 3: Combined Analysis**
1. Fill clinical data
2. Upload files
3. Get comprehensive multi-modal results
4. Compare predictions

**Workflow 4: Analytics Monitoring**
1. Navigate to /analytics
2. View real-time system metrics
3. Monitor prediction trends
4. Export analytics data

**Workflow 5: Getting Help**
1. Navigate to /help
2. Browse comprehensive user guide
3. Follow step-by-step instructions
4. Troubleshoot common issues

**Workflow 6: API Integration**
1. Check /health
2. POST to /predict
3. Receive JSON response
4. Monitor via /api/dashboard/analytics

### Browser Compatibility

**Fully Supported:**
- Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

**Required Features:**
- JavaScript enabled
- Cookies enabled
- Modern CSS support
- File API support

### Security Features

**Input Validation:**
- Client-side (JavaScript) + Server-side (Python)
- File type/size restrictions
- SQL injection prevention
- XSS protection

**Data Privacy:**
- Temporary storage only
- Secure file handling
- No third-party tracking
- HTTPS recommended

### Conclusion

The LVH Detection System features a comprehensive, professional web interface with:
- 8 fully-functional pages (including Analytics Dashboard and Help Guide)
- Interactive visualizations and real-time analytics
- Smart validation system
- Multi-modal support
- Responsive design
- Accessibility compliance
- Modern UI/UX patterns
- Production-ready code
- Complete database integration
- Real-time system monitoring
- Comprehensive user documentation and help system

---

## 🔌 API Documentation

### Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "version": "2.2.0",
  "models_loaded": true,
  "available_modalities": ["ecg", "mri", "ct", "clinical"],
  "total_models": 36,
  "algorithms": 9
}
```

### Analytics Dashboard API

```bash
# Get system analytics
GET /api/dashboard/analytics

# Get recent predictions
GET /api/dashboard/recent-predictions

# Get system metrics
GET /api/dashboard/metrics

# Get performance metrics
GET /api/dashboard/performance-metrics

# Export analytics data as CSV
GET /api/dashboard/export/csv

# Export analytics data as JSON
GET /api/dashboard/export/json
```

### Prediction Endpoint

```bash
POST /predict

Content-Type: multipart/form-data

Parameters:
- ecg_file: ECG CSV file (optional)
- mri_file: MRI image file (optional)
- ct_file: CT image file (optional)
- age: Patient age (optional)
- sex: Gender 0=Female, 1=Male (optional)
- chest_pain_type: 0-3 (optional)
- resting_bp: Blood pressure (optional)
- cholesterol: Cholesterol level (optional)
... (other clinical fields)

Response:
{
  "prediction": "LVH Positive" or "LVH Negative",
  "confidence": 0.85,
  "confidence_pct": "85.0%",
  "modality": "ECG",
  "risk_level": "High Risk",
  "details": {...}
}
```

---

## 🗄️ Database & Analytics System

### Database Overview

The system uses **SQLite databases** for data storage and analytics:

#### 1. Predictions History (`predictions_history.db`)
- **Purpose**: Tracks all predictions for analytics
- **Key Data**: Prediction results, confidence scores, data types, timestamps
- **Tables**: `predictions`, `daily_stats`

#### 2. System Metrics (`dashboard_metrics.db`)  
- **Purpose**: Stores system performance data
- **Key Data**: CPU usage, RAM usage, API response times
- **Tables**: `system_metrics`

### Analytics Features

#### Dashboard Service (`dashboard_service.py`)
- Real-time analytics data collection
- System performance monitoring
- Prediction history tracking
- Data export functionality (CSV/JSON)

#### Background Metrics (`metrics_collector.py`)
- Automatic data collection every 30 seconds
- System health monitoring
- Database cleanup and maintenance

### Key Analytics Tracked
- **Prediction Volume**: Total predictions, daily trends
- **Accuracy Metrics**: Confidence scores, success rates
- **System Performance**: Processing times, resource usage
- **Data Usage**: Distribution by modality (ECG, MRI, CT, Clinical)

### Database Management
```bash
# Check database structure
python check_database_structure.py

# View analytics data
python -c "from dashboard_service import dashboard_service; print(dashboard_service.get_analytics_data())"
```

### Security & Privacy
- Patient data is anonymized (hashed IDs)
- No personal health information stored
- Local SQLite storage with file system security

---

## 🧪 Testing Guide

### Unit Tests

```bash
# Test clinical validation
python test_clinical_validation.py

# Test UI features
python test_enhanced_ui.py

# Test web app
python test_web_app.py

# Test fixes
python test_fixes.py
```

### Integration Tests

```bash
# Test single file prediction
python predict_single.py input_data/ecg/patient_001_ecg_features.csv ecg

# Test batch predictions
python batch_predict_all.py
```

### Manual Testing Checklist

See `TEST_CHECKLIST.txt` for complete testing scenarios:

1. ✓ Upload ECG file only
2. ✓ Upload MRI file only
3. ✓ Upload CT file only
4. ✓ Fill clinical data only
5. ✓ Upload multiple files
6. ✓ Combined file + clinical data
7. ✓ Empty submission (should error)
8. ✓ Partial clinical data (should error)

---

## 📊 Performance Metrics

### Model Evaluation Metrics

**For Each Model:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

**Visualizations Generated:**
- `confusion_matrices.png` - All confusion matrices
- `model_comparison.png` - Performance comparison
- `roc_curves.png` - ROC curves
- `*_feature_importance.png` - Feature importance

### Current Performance

**Clinical Models:**
```
GradientBoosting:      89.13% accuracy, 0.9411 ROC-AUC  ⭐ BEST
RandomForest:          88.41% accuracy, 0.9366 ROC-AUC
MLP:                   87.32% accuracy, 0.9115 ROC-AUC
StackingEnsemble:      87.32% accuracy, 0.9363 ROC-AUC
LightGBM:              86.59% accuracy, 0.9343 ROC-AUC
XGBoost:               85.51% accuracy, 0.9329 ROC-AUC
```

**ECG Models:**
```
XGBoost:               82.00% accuracy, 0.9024 ROC-AUC  ⭐ BEST
GradientBoosting:      78.00% accuracy, 0.9120 ROC-AUC
AdaBoost:              78.00% accuracy, 0.8720 ROC-AUC
LogisticRegression:    78.00% accuracy, 0.8736 ROC-AUC
StackingEnsemble:      78.00% accuracy, 0.8768 ROC-AUC
RandomForest:          76.00% accuracy, 0.8792 ROC-AUC
```

**MRI Models:**
```
SVM:                   81.43% accuracy, 0.8505 ROC-AUC  ⭐ BEST
LightGBM:              80.00% accuracy, 0.8121 ROC-AUC
GradientBoosting:      80.00% accuracy, 0.8234 ROC-AUC
LogisticRegression:    80.00% accuracy, 0.8733 ROC-AUC
StackingEnsemble:      80.00% accuracy, 0.8339 ROC-AUC
RandomForest:          78.57% accuracy, 0.8322 ROC-AUC
```

**CT Models:**
```
StackingEnsemble:      78.80% accuracy, 0.8477 ROC-AUC  ⭐ BEST
LogisticRegression:    78.75% accuracy, 0.8275 ROC-AUC
SVM:                   78.52% accuracy, 0.8206 ROC-AUC
XGBoost:               78.46% accuracy, 0.8253 ROC-AUC
GradientBoosting:      78.07% accuracy, 0.7981 ROC-AUC
LightGBM:              78.01% accuracy, 0.8471 ROC-AUC
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**2. Model Not Found**
```bash
# Solution: Train models first
python train_models.py
```

**3. Port Already in Use**
```bash
# Solution: Change port in run.py or kill process
# Windows: netstat -ano | findstr :5000
# Linux: lsof -i :5000
```

**4. Memory Error**
```bash
# Solution: Reduce batch size or use smaller dataset
# Edit config.py: BATCH_SIZE = 16
```

**5. Indentation Error**
```bash
# Solution: Check Python version and file encoding
python --version  # Should be 3.8+
```

### Performance Issues

**Slow Training:**
- Reduce number of algorithms
- Use smaller dataset
- Enable GPU if available

**Slow Predictions:**
- Use lighter models (Logistic Regression)
- Reduce image resolution
- Cache predictions

---

## 🎓 Academic Integration

### Project Credits

**Team Members:**
- Anusha T E
- Ashwitha K
- Bikram Manna P
- Chithra Pragathi

**Supervisor:**
- MS. Chaitanya V, Assistant Professor

**Institution:**
- BMS Institute of Technology and Management

### Research Contributions

1. **Multi-Modal LVH Detection**
   - Novel approach combining 4 data types
   - Ensemble learning methods
   - Real-time web interface

2. **Clinical Validation**
   - Smart validation logic
   - Optional clinical data
   - Confidence scoring

3. **Performance Analysis**
   - Comprehensive model comparison
   - 36 trained models (9 algorithms × 4 modalities)
   - Detailed visualizations
   - Advanced techniques: SMOTE, Feature Selection, Cross-validation

### Deliverables

✅ Working web application  
✅ Trained ML models  
✅ Performance reports  
✅ Complete documentation  
✅ Test datasets  
✅ API endpoints  
✅ Visualization charts  

---


## 🚀 Quick Start Commands

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Train models (optional)
python train_models.py 
# and
python train_models_corrected.py

# 3. Start application
python run.py

# 4. Access application
# Open browser: http://localhost:5000

# 5. Test prediction
python predict_single.py input_data/ecg/patient_001_ecg_features.csv ecg
```

---

## 📞 Support & Contact

### Getting Help

1. Check documentation files
2. Review troubleshooting section
3. Check `/health` endpoint
4. Review console logs

### Reporting Issues

Include:
- Error message
- Steps to reproduce
- System information
- Python version

---

## ✅ System Status

**Version:** 2.2.0  
**Status:** ✅ Production Ready  
**Last Updated:** November 22, 2025  

**Components:**
- ✅ Data Pipeline: Working
- ✅ Model Training: Complete
- ✅ Web Application: Running
- ✅ API Endpoints: Active
- ✅ Documentation: Complete
- ✅ Testing: Verified

---

## 🎉 Conclusion

This LVH Detection System is a complete, production-ready application that demonstrates:

- Advanced machine learning techniques
- Multi-modal data processing
- Professional web interface
- Comprehensive documentation
- Academic research quality

**Ready to use for:**
- Medical research
- Academic projects
- Clinical demonstrations
- Educational purposes
- Further development

---

**Start Now:**
```bash
python run.py
```

**Access at:** http://localhost:5000

**Enjoy your LVH Detection System!** 🏥💙

---

*End of Complete Project Guide*
