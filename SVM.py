import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# =========================
# CONFIG
# =========================
DATA_DIR = "test_data"

MODEL_PATH = os.path.join(DATA_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "svm_scaler.pkl")

X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "Y.npy")

# =========================
# LOAD OR TRAIN MODEL
# =========================
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("Loading saved SVM model...")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

else:
    print("Training SVM model...")

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise Exception("Run mfcc_extractor.py first!")

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    print("Loaded dataset:", X.shape)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # Train SVM
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, Y_train)

    # Evaluate
    Y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(Y_test, Y_pred))
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

    # Save model + scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model saved!")


# =========================
# PREDICT FROM NPY FILE
# =========================
def predict_from_npy(npy_path):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return

    features = np.load(npy_path).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    print(f"\nFile: {npy_path}")
    print("Prediction:", prediction)
    print("\nClass Probabilities:")

    for cls, prob in zip(classes, probabilities):
        print(f"{cls}: {prob:.4f}")

    return prediction


# =========================
# COMPUTE ACCURACY
# =========================
def compute_accuracy():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Dataset not found! Run mfcc first.")
        return

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    X = scaler.transform(X)

    # Same split as training
    _, X_test, _, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    Y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")


# =========================
# CLI MENU
# =========================
if __name__ == "__main__":
    print("\nChoose option:")
    print("1 - Predict custom file")
    print("2 - Compute accuracy")

    choice = input("Enter choice: ")

    # OPTION 1 → PREDICT
    if choice == "1":
        test_npy = input("Enter .npy file name (e.g., up1.npy): ")
        test_path = os.path.join(DATA_DIR, test_npy)

        if os.path.exists(test_path):
            result = predict_from_npy(test_path)
            print(f"\nPrediction for {test_path}: {result}")
        else:
            print("File not found!")

    # OPTION 2 → ACCURACY
    elif choice == "2":
        compute_accuracy()

    else:
        print("Invalid choice!")