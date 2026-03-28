import librosa
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "./speech_data"
ALLOWED_LABELS = ["yes", "no", "up", "down", "stop"]
MAX_LEN = 16000  # 1 second audio
N_MFCC = 17
MODEL_PATH = "./knn_model.pkl"
SCALER_PATH = "./scaler.pkl"

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=16000)

    # 1. Trim silence
    signal, _ = librosa.effects.trim(signal)

    # 2. Fix length
    if len(signal) < MAX_LEN:
        signal = np.pad(signal, (0, MAX_LEN - len(signal)))
    else:
        signal = signal[:MAX_LEN]

    # 3. MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    # 4. Delta features
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 5. Combine features
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1)
    ])

    return features

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("Loading saved model...")
    
    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

else:
    print("Training model ...")

    # ==============================
    # LOAD DATASET
    # ==============================
    X = []
    Y = []

    for label in os.listdir(DATASET_PATH):
        folder = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(folder):
            continue

        if label not in ALLOWED_LABELS:
            continue

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)

                try:
                    features = extract_features(path)
                    X.append(features)
                    Y.append(label)

                except Exception as e:
                    print(f"Error with file {path}: {e}")

    X = np.array(X)
    Y = np.array(Y)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # ==============================
    # NORMALIZATION
    # ==============================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ==============================
    # TRAIN TEST SPLIT
    # ==============================
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # ==============================
    # TRAIN KNN MODEL
    # ==============================
    knn = KNeighborsClassifier(
        n_neighbors=22,
        metric='cosine',
        weights='distance'
    )

    knn.fit(X_train, Y_train)

    # ==============================
    # EVALUATION
    # ==============================
    Y_pred = knn.predict(X_test)

    print("\nAccuracy:", accuracy_score(Y_test, Y_pred))
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

    # ==============================
    # SAVE MODEL + SCALER
    # ==============================
    joblib.dump(knn, "knn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("\nModel and scaler saved!")

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_audio(file_path):
    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")

    features = extract_features(file_path)

    features = scaler.transform(features.reshape(1, -1))

    prediction = knn.predict(features)

    return prediction[0]


# ==============================
# TEST WITH NEW AUDIO
# ==============================
if __name__ == "__main__":
    test_file = "./custom_tests/yes.wav"  # change this path

    if os.path.exists(test_file):
        result = predict_audio(test_file)
        print(f"\nPrediction for {test_file}: {result}")
    else:
        print("\nPut a test.wav file in this directory to test prediction")