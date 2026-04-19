import librosa
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "./speech_data"
ALLOWED_LABELS = ["yes", "no", "up", "down", "stop"]
MAX_LEN = 16000
N_MFCC = 17
MAX_FRAMES = 100   # for CNN

OUTPUT_X = "./test_data/X.npy"          # classical ML
OUTPUT_X_CNN = "./test_data/X_cnn.npy"  # CNN
OUTPUT_Y = "./test_data/Y.npy"

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def extract_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1)
    ])

    return features


def extract_cnn_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    # FIX LENGTH (time dimension)
    if mfcc.shape[1] > MAX_FRAMES:
        mfcc = mfcc[:, :MAX_FRAMES]
    else:
        pad_width = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))

    return mfcc

# ==============================
# DATASET PROCESSING
# ==============================
def create_dataset():
    X = []        # classical features
    X_cnn = []    # CNN features
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
                    signal, sr = librosa.load(path, sr=16000)

                    # remove silence
                    signal, _ = librosa.effects.trim(signal)

                    # FIX LENGTH (raw signal)
                    if len(signal) < MAX_LEN:
                        signal = np.pad(signal, (0, MAX_LEN - len(signal)))
                    else:
                        signal = signal[:MAX_LEN]

                    # -------- classical features --------
                    features = extract_features(signal, sr)
                    X.append(features)

                    # -------- CNN features --------
                    mfcc_full = extract_cnn_features(signal, sr)
                    X_cnn.append(mfcc_full)

                    Y.append(label)

                except Exception as e:
                    print(f"Error with {path}: {e}")

    X = np.array(X)
    X_cnn = np.array(X_cnn)
    Y = np.array(Y)

    print("Classical X shape:", X.shape)        # (N, 68)
    print("CNN X shape:", X_cnn.shape)          # (N, 17, 100)
    print("Y shape:", Y.shape)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_X_CNN, X_cnn)
    np.save(OUTPUT_Y, Y)

    print("Saved: X.npy, X_cnn.npy, Y.npy")


if __name__ == "__main__":
    create_dataset()