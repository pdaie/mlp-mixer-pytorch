import yaml

import torch

from utils.metrics import AverageMeter


def train_step(model, dataloader, optimizer, criterion, device):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        
        logits = model(X)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        n = X.shape[0]
        loss_meter.update(loss.item(), n)
        acc = torch.sum(logits.argmax(dim=1) == y)
        acc_meter.update(acc.item() / n, n)

    return loss_meter.avg, acc_meter.avg


def val_step(model, dataloader, criterion, device):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            loss = criterion(logits, y)
            
            n = X.shape[0]
            loss_meter.update(loss.item(), n)
            acc = torch.sum(logits.argmax(dim=1) == y)
            acc_meter.update(acc.item() / n, n)

    return loss_meter.avg, acc_meter.avg