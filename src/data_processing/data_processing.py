import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(image_size=224):
    """
    Returns training and validation transforms.
    """

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    return train_transform, eval_transform


def get_datasets(data_dir="../data/raw", image_size=224):
    """
    Creates train/validation/test datasets using ImageFolder.
    """

    train_transform, eval_transform = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(
        f"{data_dir}/train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        f"{data_dir}/valid",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        f"{data_dir}/test",
        transform=eval_transform
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    data_dir="../data/raw",
    batch_size=32,
    image_size=224,
    num_workers=2
):
    """
    Returns dataloaders and class names.
    """

    train_dataset, val_dataset, test_dataset = get_datasets(
        data_dir=data_dir,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    classes = train_dataset.classes

    return train_loader, val_loader, test_loader, classes

