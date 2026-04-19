# Speech Command Classification using MFCC and Multiple ML Models

This project implements a modular speech command recognition system that classifies spoken commands:

"yes", "no", "up", "down", "stop"

It uses MFCC-based feature extraction and multiple models:

* KNN
* Logistic Regression
* SVM
* CNN (PyTorch)

---

# Project Pipeline

```text
Audio (.wav)
      ↓
MFCC Feature Extraction
      ↓
Saved as .npy files
      ↓
Model Training / Inference
      ↓
Predicted Command
```

---

# Project Structure

```text
.
├── speech_data/              # Training dataset (organized by labels)
├── custom_audio_files/       # Raw custom test audio (.wav)
├── test_data/                # Processed features + models
│   ├── X.npy
│   ├── Y.npy
│   ├── X_cnn.npy
│   ├── *.npy (custom processed files)
│   ├── *_cnn.npy
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   ├── logreg_model.pkl
│   ├── logreg_scaler.pkl
│   ├── svm_model.pkl
│   ├── svm_scaler.pkl
│   ├── cnn_model.pth
│   ├── cnn_scaler.pkl
│   ├── cnn_labels.npy
│
├── mfcc.py                   # Training feature extraction
├── preprocess_custom.py      # Custom file preprocessing
├── knn_model.py
├── logistic_model.py
├── svm_model.py
├── cnn_model.py
```

---

# Setup Instructions

## 1. Create Virtual Environment

```bash
python3 -m venv myenv
source myenv/bin/activate        # Linux / Mac
.\myenv\Scripts\activate         # Windows PowerShell
```

## 2. Install Dependencies

```bash
pip install numpy librosa scikit-learn torch joblib
```

---

# Workflow

## Step 1: Extract Training Features

```bash
python mfcc.py
```

This generates:

* X.npy, Y.npy (for classical models)
* X_cnn.npy (for CNN)

---

## Step 2: Preprocess Custom Audio Files

```bash
python preprocess_custom.py
```

This script:

* Reads all `.wav` files from `custom_audio_files/`
* Converts each file into:

  * `.npy` (for ML models)
  * `_cnn.npy` (for CNN)
* Saves them into `test_data/`

### Optimization

```text
if file.npy exists:
    skip
else:
    process and save
```

This avoids recomputation and speeds up repeated testing.

---

## Step 3: Train or Load Models

```bash
python knn_model.py
python logistic_model.py
python svm_model.py
python cnn_model.py
```

Each script:

* Trains the model if it does not exist
* Otherwise loads the saved model

---

## Step 4: Run Model

Each model provides:

```text
1 - Predict custom file
2 - Compute accuracy
```

---

## Step 5: Prediction Example

```text
Enter .npy file name: yes1.npy
```

Output:

```text
Prediction: yes

Class Probabilities:
yes: 0.82
no: 0.05
up: 0.04
down: 0.03
stop: 0.06
```

---

# Models Used

| Model               | Type           | Notes                 |
| ------------------- | -------------- | --------------------- |
| KNN                 | Distance-based | Simple, interpretable |
| Logistic Regression | Linear         | Fast, lightweight     |
| SVM                 | Kernel-based   | High accuracy         |
| CNN                 | Deep Learning  | Best performance      |

---

# Key Design Idea

```text
Audio → MFCC → .npy → Model
```

* Decoupled preprocessing and inference
* No repeated feature extraction
* Easy to plug in new models

---

# Notes

* Ensure MFCC parameters remain consistent across scripts
* Re-run `mfcc.py` if feature extraction settings change
* Custom `.npy` files must match training format

---

# Author

Madhur Zanwar, Mahiman Bhardwaj
