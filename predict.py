import torch
import librosa
import numpy as np
import sys

from model import DeepfakeCNN


# -------------------------------
# Load Model
# -------------------------------
model = DeepfakeCNN()

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features(audio_path):

    signal, sr = librosa.load(
        audio_path,
        sr=16000
    )

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    target_length = 100

    current_length = mel_tensor.shape[1]

    if current_length < target_length:

        padding = target_length - current_length

        mel_tensor = torch.nn.functional.pad(
            mel_tensor,
            (0, padding)
        )

    elif current_length > target_length:

        mel_tensor = mel_tensor[
            :,
            :target_length
        ]

    return mel_tensor


# -------------------------------
# Prediction Function
# -------------------------------
def predict(audio_path):

    mel_tensor = extract_features(
        audio_path
    )

    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        outputs = model(mel_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    prediction = prediction.item()

    confidence = confidence.item()

    if prediction == 0:

        print("\nPrediction: FAKE AUDIO")

    else:

        print("\nPrediction: REAL AUDIO")

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python predict.py <audio_file.wav>"
        )

    else:

        audio_path = sys.argv[1]

        predict(audio_path)