import os
import yaml
import argparse

import torch
import torch.nn as nn

from model.mlp_mixer import MLPMixer
from utils.engine import train_step, val_step
from utils.dataloaders import create_dataloaders
from utils.save_model import save_entire_model


DATA_CONFIG = yaml.load(open('data/config.yaml', 'r'), Loader=yaml.loader.SafeLoader)


def train(opt):
    if opt.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(opt.device)
        
    train_dir = os.path.join(DATA_CONFIG['path'], DATA_CONFIG['train'])
    train_dataloader, train_dataset = create_dataloaders(
        dir=train_dir,
        image_size=opt.image_size,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers if opt.num_workers != -1 else os.cpu_count()
    )
    
    if DATA_CONFIG['val'] is not None:
        var_dir = os.path.join(DATA_CONFIG['path'], DATA_CONFIG['val'])
        val_dataloader, _ = create_dataloaders(
            dir=var_dir,
            image_size=opt.image_size,
            batch_size=opt.batch_size
        )
    
    image_size = opt.image_size
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
        
    model = MLPMixer(
        num_classes=len(train_dataset.classes),
        image_size=image_size,
        patch_size=opt.patch_size,
        num_mlp_blocks=opt.num_mlp_blocks,
        projection_dim=opt.projection_dim, 
        token_mixing_dim=opt.token_mixing_dim,
        channel_mixing_dim=opt.channel_mixing_dim
    ).to(device)
            
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    
    for epoch in range(opt.epochs):
        train_loss, train_acc = train_step(model, train_dataloader, optimizer, criterion, device)
        
        if DATA_CONFIG['val'] is not None:
            val_loss, val_acc = val_step(model, val_dataloader, criterion, device)
            print(f'\t- Epoch: {epoch+1}', end=' ')
            print(f'- loss: {train_loss:.4f} - acc: {train_acc:.4f}', end=' ')
            print(f'- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
        else:
            print(f'\t- Epoch: {epoch+1}', end=' ')
            print(f'- loss: {train_loss:.4f} - acc: {train_acc:.4f}')
    print()
    
    save_entire_model(model, 'last')
    
    
def main(opt):
    try:
        if DATA_CONFIG['download']:
            download_dataset()
    except:
        pass

    train(opt)
    
    
def parse_opt():
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
    
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)