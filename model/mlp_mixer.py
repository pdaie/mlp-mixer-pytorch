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


class MixerBlock(nn.Module):
    def __init__(
        self, 
        num_patches, 
        embedding_dim, 
        token_mixing_dim,
        channel_mixing_dim
    ):
        super().__init__()
        self.norm_token_mixing = nn.LayerNorm(embedding_dim)
        self.norm_channel_mixing = nn.LayerNorm(embedding_dim)
        
        token_mixing_encoder = torch.cat([nn.Parameter(torch.randn(num_patches)).reshape(-1, 1)]*token_mixing_dim, dim=1)
        token_mixing_decoder = torch.cat([nn.Parameter(torch.randn(token_mixing_dim)).reshape(-1, 1)]*num_patches, dim=1)
        self.register_buffer('token_mixing_encoder', token_mixing_encoder)
        self.register_buffer('token_mixing_decoder', token_mixing_decoder)
        
        channel_mixing_encoder = torch.cat([nn.Parameter(torch.randn(embedding_dim)).reshape(-1, 1)]*channel_mixing_dim, dim=1)
        channel_mixing_decoder = torch.cat([nn.Parameter(torch.randn(channel_mixing_dim)).reshape(-1, 1)]*embedding_dim, dim=1)
        self.register_buffer('channel_mixing_encoder', channel_mixing_encoder)
        self.register_buffer('channel_mixing_decoder', channel_mixing_decoder)
    
    def token_mixing_mlp(self, x):
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, self.token_mixing_encoder)
        x = torch.nn.functional.gelu(x)
        x = torch.matmul(x, self.token_mixing_decoder)
        x = x.permute(0, 2, 1)
        
        return x
    
    def channel_mixing_mlp(self, x):
        x = torch.matmul(x, self.channel_mixing_encoder)
        x = torch.nn.functional.gelu(x)
        x = torch.matmul(x, self.channel_mixing_decoder)
        
        return x
    
    def token_mixing_block(self, x):
        x = self.norm_token_mixing(x)
        x_skip_connection = x.clone().detach()
        x = self.token_mixing_mlp(x)
        x = x + x_skip_connection
        
        return x
    
    def channel_mixing_block(self, x):
        x = self.norm_channel_mixing(x)
        x_skip_connection = x.clone().detach()
        x = self.channel_mixing_mlp(x)
        x = x + x_skip_connection
        
        return x
        
    def forward(self, x):
        x = self.token_mixing_block(x)
        x = self.channel_mixing_block(x)
        
        return x
    
    
class MLPMixer(nn.Module):
    def __init__(
        self,
        num_classes,
        image_size,
        patch_size,
        num_mlp_blocks,
        projection_dim, 
        token_mixing_dim,
        channel_mixing_dim
    ):
        super().__init__()
        
        self.embedding = PatchEmbedding(patch_size)
        
        self.projection = nn.Linear(3*patch_size*patch_size, projection_dim)
        
        self.num_patches = int((image_size[0] * image_size[1]) / (patch_size ** 2))
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(
                    self.num_patches, 
                    projection_dim, 
                    token_mixing_dim, 
                    channel_mixing_dim
                )
            ]*num_mlp_blocks
        )
                
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.projection(x)
        
        for block in self.mixer_blocks:
            x = block(x)
            
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.avg_pool1d(x, kernel_size=self.num_patches)
        x = x.squeeze()
        x = self.classifier(x)
        
        return x
    
    
if __name__ == '__main__':
    device = torch.device('cuda:0')
    
    model = MLPMixer(
        num_classes=2,
        image_size=[300, 300],
        patch_size=100,
        num_mlp_blocks=8,
        projection_dim=512,
        token_mixing_dim=2048,
        channel_mixing_dim=256
    ).to(device)
    
    x = torch.randn(4, 3, 300, 300).to(device)
    
    y = model(x)
    
    print(y)
    print(y.argmax(dim=1))