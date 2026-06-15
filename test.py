
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from audio_dataset import AudioDataset
from model import DeepfakeCNN


# -------------------------------
# Load Test Dataset
# -------------------------------
test_dataset = AudioDataset("dataset/test")

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


# -------------------------------
# Load Trained Model
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
# Testing Variables
# -------------------------------
correct = 0
total = 0

all_labels = []
all_predictions = []
all_scores = []


# -------------------------------
# Testing Loop
# -------------------------------
with torch.no_grad():

    for spectrograms, labels in test_loader:

        # Add channel dimension
        spectrograms = spectrograms.unsqueeze(1)

        # Forward pass
        outputs = model(spectrograms)

        # Predicted class
        _, predicted = torch.max(outputs, 1)

        # Accuracy calculation
        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        # Store predictions
        all_labels.extend(labels.numpy())

        all_predictions.extend(predicted.numpy())

        # Store probabilities for EER
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        real_scores = probabilities[:, 1]

        all_scores.extend(real_scores.numpy())


# -------------------------------
# Overall Accuracy
# -------------------------------
accuracy = 100 * correct / total

print(
    f"\nTest Accuracy: {accuracy:.2f}%"
)


# -------------------------------
# Confusion Matrix
# -------------------------------
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)


# -------------------------------
# Classification Report
# -------------------------------
print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Fake", "Real"]
    )
)


# -------------------------------
# Equal Error Rate (EER)
# -------------------------------
fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_scores
)

eer = brentq(
    lambda x: 1.0 - x - interp1d(fpr, tpr)(x),
    0.0,
    1.0
)

print(
    f"EER: {eer * 100:.2f}%"
)
