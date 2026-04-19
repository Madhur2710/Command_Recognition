import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =========================
# CONFIG
# =========================
DATA_DIR = "test_data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_PATH = os.path.join(DATA_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "svm_scaler.pkl")
LABELS_PATH = os.path.join(DATA_DIR, "svm_labels.npy")
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "Y.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test_svm.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test_svm.npy")

# =========================
# TRAIN IF MODEL NOT EXISTS
# =========================
if not os.path.exists(MODEL_PATH):
    print("Model not found. Training SVM...")
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    labels = sorted(list(set(Y)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_int = np.array([label_to_idx[label] for label in Y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)

    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    np.save(LABELS_PATH, labels)
    print("Training complete.")

# =========================
# LOAD MODEL
# =========================
print("Loading SVM model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)

# =========================
# USER OPTION
# =========================
print("\nChoose option:")
print("1 → Predict custom file")
print("2 → Compute accuracy")
choice = input("Enter choice: ")

# =========================
# PREDICTION
# =========================
if choice == "1":
    while True:
        file_name = input("Enter the .npy filename (with or without extension): ").strip()

        # Strip .npy extension if user included it, then re-add cleanly
        if file_name.endswith(".npy"):
            file_name = file_name[:-4]

        path = os.path.join(DATA_DIR, f"{file_name}.npy")

        if os.path.exists(path):
            break
        else:
            print(f"ERROR: '{file_name}.npy' not found in '{DATA_DIR}/'. Please try again.")

    X_sample = np.load(path)
    X_sample = scaler.transform(X_sample)
    pred = model.predict(X_sample)
    print("Predicted command:", labels[pred[0]])

# =========================
# ACCURACY
# =========================
elif choice == "2":
    if not os.path.exists(X_TEST_PATH):
        print("ERROR: Test set not found. Retrain model.")
        exit()
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    print(f"Test Accuracy: {accuracy:.4f}")

else:
    print("Invalid choice.")