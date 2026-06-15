import streamlit as st
import torch
import librosa
import numpy as np
import tempfile

from model import DeepfakeCNN


# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Deepfake Audio Detection",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Deepfake Audio Detection")
st.write("Upload a WAV audio file to determine whether it is REAL or FAKE.")


# --------------------------------
# Load Model
# --------------------------------
@st.cache_resource
def load_model():

    model = DeepfakeCNN()

    model.load_state_dict(
        torch.load(
            "model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


model = load_model()


# --------------------------------
# Feature Extraction
# --------------------------------
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


# --------------------------------
# Upload Audio
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload a WAV file",
    type=["wav"]
)


if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        temp_path = temp_file.name


    # Extract features
    mel_tensor = extract_features(
        temp_path
    )


    # Prepare input for CNN
    input_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)


    # Prediction
    with torch.no_grad():

        outputs = model(
            input_tensor
        )

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


    # Display Result
    st.subheader("🔍 Prediction")

    if prediction == 0:

        st.error(
            f"Prediction: FAKE AUDIO\n\n"
            f"Confidence Score: {confidence * 100:.2f}%"
        )

    else:

        st.success(
            f"Prediction: REAL AUDIO\n\n"
            f"Confidence Score: {confidence * 100:.2f}%"
        )