import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(dir, image_size, batch_size, num_workers):
    if isinstance(image_size, int):
        image_size = [image_size, image_size]

    transform = transforms.Compose([
        transforms.Resize((image_size[0], image_size[1])),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(
        root=dir,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    return dataloader, dataset


if __name__ == '__main__':
    dataloader, dataset = create_dataloaders('data/train', 224, 8, os.cpu_count())
    print(dataloader, dataset)