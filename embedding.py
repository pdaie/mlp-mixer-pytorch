import torch
import torch.nn as nn


class PatchEmbedding():
    def __init__(self, patch_size=100): 
        self.patch_size = patch_size
    
    def __call__(self, x):
        batch_size, channel, height, width = x.shape
        num_patches_height = height / self.patch_size
        num_patches_width = width / self.patch_size
        num_patches = int(num_patches_height * num_patches_width)
        
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # B C H W -> B C NPH NPW P P
        patches = patches.reshape(batch_size, channel, num_patches, self.patch_size, self.patch_size) # B C NPH NPW P P -> B C NP P P
        patches = patches.permute(0, 2, 1, 3, 4)  # B C NP P P -> B NP C P P
        patches_embedding = patches.reshape(batch_size, num_patches, -1) # B NP C P P -> B NP C*P*P
        
        return patches_embedding