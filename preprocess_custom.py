import librosa
import numpy as np
import os

# ==============================
# CONFIG (must match training!)
# ==============================
SR = 16000
MAX_LEN = 16000
N_MFCC = 17
MAX_FRAMES = 100

INPUT_FOLDER = "./custom_audio_files"
OUTPUT_FOLDER = "./test_data"

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def preprocess_wav(file_path):
    signal, sr = librosa.load(file_path, sr=SR)

    signal, _ = librosa.effects.trim(signal)

    if len(signal) < MAX_LEN:
        signal = np.pad(signal, (0, MAX_LEN - len(signal)))
    else:
        signal = signal[:MAX_LEN]

    # -------- CLASSICAL --------
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    classical_features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1)
    ])

    classical_features = classical_features.reshape(1, -1)

    # -------- CNN --------
    mfcc_full = mfcc.copy()

    if mfcc_full.shape[1] > MAX_FRAMES:
        mfcc_full = mfcc_full[:, :MAX_FRAMES]
    else:
        pad_width = MAX_FRAMES - mfcc_full.shape[1]
        mfcc_full = np.pad(mfcc_full, ((0, 0), (0, pad_width)))

    cnn_input = mfcc_full.reshape(1, 1, N_MFCC, MAX_FRAMES)

    return classical_features, cnn_input


# ==============================
# PROCESS ALL FILES
# ==============================
def process_all_files():
    if not os.path.exists(INPUT_FOLDER):
        print("ERROR: Input folder not found!")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".wav")]

    if len(files) == 0:
        print("No .wav files found!")
        return

    print(f"Found {len(files)} files\n")

    for file in files:
        input_path = os.path.join(INPUT_FOLDER, file)
        base_name = os.path.splitext(file)[0]

        classical_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.npy")
        cnn_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_cnn.npy")

        # SKIP IF ALREADY EXISTS
        if os.path.exists(classical_path):
            print(f"Skipping {file} (already processed)")
            continue

        try:
            classical, cnn = preprocess_wav(input_path)

            np.save(classical_path, classical)
            np.save(cnn_path, cnn)

            print(f"Processed {file}")
            print(f"  → {classical_path}")
            print(f"  → {cnn_path}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    print("\nDone processing all files!")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    process_all_files()