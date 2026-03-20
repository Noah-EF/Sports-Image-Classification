import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.utils.logger import ExperimentLogger
from src.data_processing.data_processing import get_dataloaders


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, dataloader, device):

    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    loss = running_loss / len(dataloader)

    return loss, accuracy


def main():

    config = {
        "experiment": "baseline_resnet18",
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "num_classes": 100
    }

    device = get_device()

    print(f"Using device: {device}")

    logger = ExperimentLogger(config["experiment"], config)

    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=config["batch_size"])

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, config["num_classes"])

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            if epoch == 0 and batch_idx == 0:
                logger.log_image("sample_image", images[0], epoch)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)

        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)

        logger.log_metrics(
            {
                "loss/train": train_loss,
                "accuracy/train": train_acc,
                "loss/val": val_loss,
                "accuracy/val": val_acc
            },
            epoch
        )

        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f}"
        )

    logger.close()

    print("Experiment complete.")


if __name__ == "__main__":
    main()