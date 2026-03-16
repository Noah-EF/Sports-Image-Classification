import torch
import torch.nn as nn
import torch.optim as optim

from src.data_processing.data_processing import get_dataloaders
from src.models.simple_cnn import SimpleCNN
from src.training.train import train_model, evaluate

import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, classes = get_dataloaders()

    model = SimpleCNN(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=25
    )

    test_acc = evaluate(model, test_loader, device)

    print("Test Accuracy:", test_acc)

if __name__ == "__main__":
    main()

