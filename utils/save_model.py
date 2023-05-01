import os
import torch


def save_entire_model(model, name):
    print('[!] Save model')
    save_dir = 'runs'
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)
    
    i = 1
    while True:
        save_exp_dir = os.path.join(save_dir, f'exp_{i}')
        
        if os.path.isdir(save_exp_dir) is False:
            os.mkdir(save_exp_dir)
            break
        
        i += 1
    
    save_model_path = os.path.join(save_exp_dir, f'{name}.pt')
    torch.save(model, save_model_path)
    print(f'- Model saved to {save_exp_dir}')
    
    
def save_state_dict(model, name):
    save_dir = 'runs'
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)
    
    i = 1
    while True:
        save_exp_dir = os.path.join(save_dir, f'exp_{i}')
        
        if os.path.isdir(save_exp_dir) is False:
            os.mkdir(save_exp_dir)
            break
        
        i += 1
    
    save_model_path = os.path.join(save_exp_dir, f'{name}.pt')
    torch.save(model.state_dict(), save_model_path)