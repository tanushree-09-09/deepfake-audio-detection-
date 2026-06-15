import torch
from torch.utils.data import DataLoader

from audio_dataset import AudioDataset
from model import DeepfakeCNN

# Load validation dataset
val_dataset = AudioDataset("dataset/val")

# Create dataloader
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# Create model
model = DeepfakeCNN()

# Load trained weights
model.load_state_dict(torch.load("model.pth"))

# Evaluation mode
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for spectrograms, labels in val_loader:

        spectrograms = spectrograms.unsqueeze(1)

        outputs = model(spectrograms)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Validation Accuracy:", accuracy, "%")