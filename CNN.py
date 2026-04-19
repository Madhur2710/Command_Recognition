import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
DATA_DIR = "test_data"

MODEL_PATH = os.path.join(DATA_DIR, "cnn_model.pth")
SCALER_PATH = os.path.join(DATA_DIR, "cnn_scaler.pkl")
LABELS_PATH = os.path.join(DATA_DIR, "cnn_labels.npy")

X_PATH = os.path.join(DATA_DIR, "X_cnn.npy")
Y_PATH = os.path.join(DATA_DIR, "Y.npy")

# =========================
# MODEL
# =========================
class SpeechCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpeechCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.fc1 = nn.Linear(64 * 2 * 23, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# =========================
# LOAD OR TRAIN MODEL
# =========================
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("Loading saved CNN model...")

else:
    print("Training CNN model...")

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    print("Loaded dataset:", X.shape)

    labels = sorted(list(set(Y)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_int = np.array([label_to_idx[label] for label in Y])

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    # NORMALIZE
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    X_train = X_train_flat.reshape(-1, 1, 17, 100)
    X_test = X_test_flat.reshape(-1, 1, 17, 100)

    # SAVE TEST SET
    np.save(os.path.join(DATA_DIR, "X_test_cnn.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test_cnn.npy"), y_test)

    # TO TENSOR
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    model = SpeechCNN(len(labels))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TRAIN
    for epoch in range(10):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # SAVE
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    np.save(LABELS_PATH, labels)

    print("Model saved!")


# =========================
# LOAD MODEL
# =========================
labels = np.load(LABELS_PATH, allow_pickle=True)
scaler = joblib.load(SCALER_PATH)

model = SpeechCNN(len(labels))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# =========================
# PREDICT FROM NPY
# =========================
def predict_from_npy(npy_path):
    if not os.path.exists(npy_path):
        print("File not found:", npy_path)
        return

    cnn_input = np.load(npy_path)

    cnn_flat = cnn_input.reshape(1, -1)
    cnn_flat = scaler.transform(cnn_flat)
    cnn_input = cnn_flat.reshape(1, 1, 17, 100)

    cnn_input = torch.tensor(cnn_input, dtype=torch.float32)

    with torch.no_grad():
        output = model(cnn_input)
        probs = torch.softmax(output, dim=1)
        predicted = torch.argmax(probs, 1)

    print(f"\nFile: {npy_path}")
    print("Prediction:", labels[predicted.item()])
    print("\nClass Probabilities:")

    for i, cls in enumerate(labels):
        print(f"{cls}: {probs[0][i].item():.4f}")

    return labels[predicted.item()]


# =========================
# COMPUTE ACCURACY
# =========================
def compute_accuracy():
    X_test = np.load(os.path.join(DATA_DIR, "X_test_cnn.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test_cnn.npy"))

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

        accuracy = (predicted == y_test).float().mean()

    print(f"\nTest Accuracy: {accuracy:.4f}")


# =========================
# CLI MENU
# =========================
if __name__ == "__main__":
    print("\nChoose option:")
    print("1 - Predict custom file")
    print("2 - Compute accuracy")

    choice = input("Enter choice: ")

    if choice == "1":
        test_npy = input("Enter CNN .npy file (e.g., up1_cnn.npy): ")
        path = os.path.join(DATA_DIR, test_npy)

        if os.path.exists(path):
            result = predict_from_npy(path)
            print(f"\nPrediction for {path}: {result}")
        else:
            print("File not found!")

    elif choice == "2":
        compute_accuracy()

    else:
        print("Invalid choice!")