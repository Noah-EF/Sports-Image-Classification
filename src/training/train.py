import time
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one training epoch.
    """

    model.train()

    running_loss = 0
    total_samples = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples

    return epoch_loss


def evaluate(model, loader, device):
    """
    Computes accuracy on a dataset.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=5
):
    """
    Full training loop.
    """

    history = []

    for epoch in range(epochs):

        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_acc = evaluate(model, val_loader, device)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time
        })

    return history