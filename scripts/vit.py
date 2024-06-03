import torch
from torch import nn


IMG_SIZE = 128
PS = 8
C = 3
D_MODEL_VIT = PS**2*C
NUM_PATCHES = (IMG_SIZE**2)//(PS**2) 
NUM_HEADS = 4
NUM_ENCODERS = 4

class PatchEmbeddings(nn.Module):
    
    def __init__(self, 
                patch_size: int=PS,
                channels: int=C,
                d_model: int=D_MODEL_VIT) -> None:
        
        super().__init__()
        
        self.conv_patch_layer = nn.Conv2d(in_channels = channels,
                                         out_channels = d_model,
                                         kernel_size = patch_size,
                                         stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        
        # images -> B, C, H, W
        patched_images = self.conv_patch_layer(images) # B, D_MODEL, IMG_SIZE/PS, IMG_SIZE/PS
        flattened_patches = self.flatten(patched_images) # B, D_MODEL, N
        permuted_patches = flattened_patches.permute(0, 2, 1) # B, N, D_MODEL
        return permuted_patches

class Embedding(nn.Module):
    
    def __init__(self, 
                num_patches: int=NUM_PATCHES,
                patch_size: int=PS,
                channels: int=C,
                d_model: int=D_MODEL_VIT) -> None:
        
        super().__init__()
        
        self.class_token_embedding = nn.Parameter(
            data = torch.randn(size=(1, 1, d_model)),
            requires_grad = True
        )
        self.positional_embedding = nn.Parameter(
            data = torch.randn(size=(1, num_patches+1, d_model)),
            requires_grad = True
        )
        self.patch_embeddings_layer = PatchEmbeddings(
            patch_size = patch_size,
            channels = channels,
            d_model = d_model
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images -> B, C, H, W
        patch_embeddings = self.patch_embeddings_layer(images) # B, NUM_PATCHES, D_MODEL
        patch_embeddings_with_class_token = torch.cat(
            tensors = (self.class_token_embedding, patch_embeddings), 
            dim = 1
        ) # B, NUM_PATCHES+1, D_MODEL
        return patch_embeddings_with_class_token + self.positional_embedding
    

class MSABlock(nn.Module):
    
    def __init__(self, 
                d_model: int = D_MODEL_VIT,
                num_heads: int = NUM_HEADS) -> None:
        
        super().__init__()
        
        self.attn_block = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x -> B, NUM_PATCHES+1, D_MODEL
        attn_output, _ = self.attn_block(x, x, x)
        return self.layer_norm(x + attn_output) 
    
class MLPBlock(nn.Module):
    
    def __init__(self, 
                d_model: int = D_MODEL_VIT) -> None:
        
        super().__init__()
        
        self.dense_net = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.layer_norm(x + self.dense_net(x))

    
class EncoderBlock(nn.Module):
    
    def __init__(self, 
                d_model: int = D_MODEL_VIT,
                num_heads: int = NUM_HEADS) -> None:
        
        super().__init__()
        self.msa_block = MSABlock(d_model=d_model, num_heads=num_heads)
        self.mlp_block = MLPBlock(d_model=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.mlp_block(self.msa_block(x))
    

class Encoder(nn.Module):
    
    def __init__(self, 
                num_encoders: int = NUM_ENCODERS,
                d_model: int = D_MODEL_VIT,
                num_heads: int = NUM_HEADS) -> None: 
        
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_encoders)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for block in self.blocks:
            x = block(x)
        
        return x
    
class ViT(nn.Module):
    
    def __init__(self, 
                num_encoders: int=NUM_ENCODERS,
                num_heads: int = NUM_HEADS,
                num_patches: int=NUM_PATCHES,
                patch_size: int=PS,
                channels: int=C,
                d_model: int=D_MODEL_VIT) -> None:
        
        super().__init__()
        
        self.embedding_layer = Embedding(
            num_patches, patch_size, channels, d_model
        )
        self.encoder = Encoder(
            num_encoders, d_model, num_heads
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # B, C, H, W
        embeddings = self.embedding_layer(images) # B, NUM_PATCHES+1, D_MODEL
        encoded_vectors = self.encoder(embeddings) # B, NUM_PATCHES+1, D_MODEL
        return encoded_vectors[:, :1, :]