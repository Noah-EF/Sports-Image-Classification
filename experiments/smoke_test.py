import sys
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from src.utils.logger import ExperimentLogger


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():

    config = {
        "experiment": "pipeline_smoke_test",
        "model": "resnet18",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "subset_size": 500,
        "val_size": 100,
        "num_classes": 5
    }

    device = get_device()
    print(f"Using device: {device}")

    logger = ExperimentLogger(config["experiment"], config)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.FakeData(
        size=1000,
        image_size=(3, 224, 224),
        num_classes=config["num_classes"],
        transform=transform
    )

    # split into training and validation subsets
    train_subset = Subset(dataset, range(config["subset_size"]))
    val_subset = Subset(dataset, range(config["subset_size"], config["subset_size"] + config["val_size"]))

    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False)

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
        train_accuracy = correct / total

        # validation step
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = val_correct / val_total

        # log metrics
        logger.log_metrics(
            {
                "loss/train": train_loss,
                "accuracy/train": train_accuracy,
                "accuracy/val": val_accuracy
            },
            epoch
        )

        print(
            f"Epoch {epoch+1}/{config['epochs']} "
            f"Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.3f} "
            f"Val Acc: {val_accuracy:.3f}"
        )

        model.train()  # switch back to training mode

    logger.close()
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()

    