import os
import librosa
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.audio_files = []
        self.labels = []

        # Process fake audio files
        fake_dir = os.path.join(root_dir, "fake")

        for file in os.listdir(fake_dir):
            if file.endswith(".wav"):
                self.audio_files.append(
                    os.path.join(fake_dir, file)
                )
                self.labels.append(0)  # Fake = 0

        # Process real audio files
        real_dir = os.path.join(root_dir, "real")

        for file in os.listdir(real_dir):
            if file.endswith(".wav"):
                self.audio_files.append(
                    os.path.join(real_dir, file)
                )
                self.labels.append(1)  # Real = 1

    def __len__(self):
        return len(self.audio_files)
    def __getitem__(self, idx):

        # Get audio file path and label
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load audio
        signal, sr = librosa.load(audio_path, sr=16000)

        # Generate Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_mels=128
        )

        # Convert to decibels
        mel_db = librosa.power_to_db(
            mel,
            ref=mel.max()
        )

        # Convert to PyTorch tensor
        mel_tensor = torch.tensor(
        mel_db,
           dtype=torch.float32
        )

        # Fix spectrogram width to 100
        target_length = 100
        current_length = mel_tensor.shape[1]
        if current_length < target_length:
    # Pad with zeros
         padding = target_length - current_length
         mel_tensor = torch.nn.functional.pad(
        mel_tensor,
        (0, padding)
    )
        elif current_length > target_length:
    # Crop extra columns
         mel_tensor = mel_tensor[:, :target_length]
        return mel_tensor, label 