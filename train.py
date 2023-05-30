import os
import argparse

import torch
import torch.nn as nn

from model import MLPMixer
from utils.dataloader import create_dataloader
from utils.save_model import save_entire_model


def train(args):
    if args.device == None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    if isinstance(args.image_size, int):
        args.image_size = [args.image_size, args.image_size]
        
    train_data_dir = os.path.join('data', 'train')
    train_dataloader, train_dataset = create_dataloader(
        dir=train_data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers if args.num_workers != -1 else os.cpu_count()
    )
    
    val_data_dir = os.path.join('data', 'val') if os.path.exists(os.path.join('data', 'val')) else None
    if val_data_dir is not None:
        val_dataloader, _ = create_dataloader(
            dir=val_data_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers if args.num_workers != -1 else os.cpu_count()
        )
    
    print('[!] Trainning model')
    print('- Trainning config: ')
    print(f'\t - Epochs: {args.epochs}')
    print(f'\t - Learning rate: {args.learning_rate}')
    print(f'\t - Batch size: {args.batch_size}')
    print(f'\t - Device: {device}')
    print()
    print('- Model config: ')
    print(f'\t - Num classes: {len(train_dataset.classes)}')
    print(f'\t - Image size: ({args.image_size[0]}, {args.image_size[1]})')
    print(f'\t - Patch size: {args.patch_size}')
    print(f'\t - Num MLP blocks: {args.num_mlp_blocks}')
    print(f'\t - Projection dim: {args.projection_dim}')
    print(f'\t - Token mixing dim: {args.token_mixing_dim}')
    print(f'\t - Channel mixing dim: {args.channel_mixing_dim}')
    print()
    
    model = MLPMixer(
        num_classes=len(train_dataset.classes),
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_mlp_blocks=args.num_mlp_blocks,
        projection_dim=args.projection_dim, 
        token_mixing_dim=args.token_mixing_dim,
        channel_mixing_dim=args.channel_mixing_dim
    ).to(device)
            
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print('- Trainning: ')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (torch.sum(torch.argmax(logits, dim=1) == labels) / len(labels)).item()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        if val_data_dir is None:
            print(f'\t- Epoch: {epoch+1} - loss: {train_loss:.4f} - acc: {train_acc:.4f}')
        else:
            model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    logits = model(images)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    val_acc += (torch.sum(torch.argmax(logits, dim=1) == labels) / len(labels)).item()

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

            print(f'\t- Epoch: {epoch+1} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

    print()
    
    save_entire_model(model, 'last')
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, nargs='+', default=300)
    parser.add_argument('--patch-size', type=int, default=100)
    parser.add_argument('--num-mlp-blocks', type=int, default=8)
    parser.add_argument('--projection-dim', type=int, default=512)
    parser.add_argument('--token-mixing-dim', type=int, default=2048)
    parser.add_argument('--channel-mixing-dim', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--device', type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()