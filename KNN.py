import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# CONFIG
# ==============================
X_PATH = "./test_data/X.npy"
Y_PATH = "./test_data/Y.npy"

MODEL_PATH = "./test_data/knn_model.pkl"
SCALER_PATH = "./test_data/scaler.pkl"

# ==============================
# LOAD OR TRAIN MODEL
# ==============================
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("Loading saved KNN model...")

    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

else:
    print("Training KNN model...")

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise Exception("Run mfcc_extractor.py first!")

    # Load dataset
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

    # Train KNN
    knn = KNeighborsClassifier(
        n_neighbors=22,
        metric='cosine',
        weights='distance'
    )

    knn.fit(X_train, Y_train)

    # Evaluate
    Y_pred = knn.predict(X_test)

    print("\nAccuracy:", accuracy_score(Y_test, Y_pred))
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

    # Save model + scaler
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model saved!")


# ==============================
# PREDICT FROM NPY FILE
# ==============================
def predict_from_npy(npy_path):
    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return

    features = np.load(npy_path).reshape(1, -1)
    features = scaler.transform(features)

    prediction = knn.predict(features)[0]
    probabilities = knn.predict_proba(features)[0]
    classes = knn.classes_

    print(f"\nFile: {npy_path}")
    print("Prediction:", prediction)
    print("\nClass Probabilities:")

    for cls, prob in zip(classes, probabilities):
        print(f"{cls}: {prob:.4f}")

    return prediction


# ==============================
# COMPUTE ACCURACY
# ==============================
def compute_accuracy():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Dataset not found! Run mfcc first.")
        return

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    scaler = joblib.load(SCALER_PATH)
    knn = joblib.load(MODEL_PATH)

    X = scaler.transform(X)

    # Same split as training
    _, X_test, _, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    Y_pred = knn.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")


# ==============================
# CLI MENU
# ==============================
if __name__ == "__main__":
    print("\nChoose option:")
    print("1 - Predict custom file")
    print("2 - Compute accuracy")

    choice = input("Enter choice: ")

    # OPTION 1 → PREDICT
    if choice == "1":
        test_npy = input("Enter .npy file name (e.g., Yes1.npy): ")
        test_path = os.path.join("./test_data", test_npy)

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