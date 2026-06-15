import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from audio_dataset import AudioDataset
from model import DeepfakeCNN

# Create dataset
train_dataset = AudioDataset("dataset/train")

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Create model
model = DeepfakeCNN()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Number of epochs
epochs = 5

for epoch in range(epochs):

    running_loss = 0.0

    for spectrograms, labels in train_loader:

        # Add channel dimension
        spectrograms = spectrograms.unsqueeze(1)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(spectrograms)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )
    torch.save(model.state_dict(), "model.pth")

    print("Model saved successfully!")