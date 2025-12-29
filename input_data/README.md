# Input Data for Real-Time LVH Prediction

This folder contains sample patient data for real-time LVH detection predictions.

## Folder Structure

```
input_data/
├── ecg/          # ECG signal files (CSV format)
├── mri/          # MRI cardiac images
└── ct/           # CT cardiac images
```

## File Formats

### ECG Files
- Format: CSV
- Naming: patient_XXX_ecg.csv
- Content: ECG signal data (one row per patient)
- Columns: ECG signal values

### MRI Files
- Format: PNG, JPG, or DICOM
- Naming: patient_XXX_mri.[ext]
- Content: Cardiac MRI images

### CT Files
- Format: PNG, JPG, or DICOM
- Naming: patient_XXX_ct.[ext]
- Content: Cardiac CT scan images

## Usage

To make predictions on these samples, run:

```bash
python predict.py --input input_data/ecg/patient_001_ecg.csv --modality ecg
python predict.py --input input_data/mri/patient_001_mri.png --modality mri
python predict.py --input input_data/ct/patient_001_ct.png --modality ct
```

Or use the batch prediction script:

```bash
python batch_predict.py --input-dir input_data
```

## Notes

- These are sample files extracted from the training dataset
- For real clinical use, replace with actual patient data
- Ensure data privacy and HIPAA compliance when using real patient data
- All predictions should be reviewed by qualified medical professionals
