import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List
import math
import timm
import os

class PatchEmbeddings(nn.Module):
    
    """Class for creating the patches of an image using a convolutional layer"""

    def __init__(self, config) -> None:
        
        super().__init__()
        self.conv_patch_layer = nn.Conv2d(in_channels = config['channels'],
                                         out_channels = config['d_model'],
                                         kernel_size = config['patch_size'],
                                         stride = config['patch_size'])
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        
        # images -> B, C, H, W
        patched_images = self.conv_patch_layer(images) # B, D_MODEL, IMG_SIZE/PS, IMG_SIZE/PS
        flattened_patches = self.flatten(patched_images) # B, D_MODEL, N
        permuted_patches = flattened_patches.permute(0, 2, 1) # B, N, D_MODEL
        return permuted_patches


class ViTEmbedding(nn.Module):
    
    """Creates the input embeddings for the ViT class by combining both patch and positional embeddings"""

    def __init__(self, config) -> None:
        
        super().__init__()
        self.class_token_embedding = nn.Parameter(
            data = torch.randn(size=(1, 1, config['d_model'])),
            requires_grad = True
        )
        self.positional_embedding = nn.Parameter(
            data = torch.randn(size=(1, config['num_patches']+1, config["d_model"])),
            requires_grad = True
        )
        self.patch_embeddings_layer = PatchEmbeddings(config)
        self.dropout = nn.Dropout(p=config['emb_dropout'])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images -> B, C, H, W
        patch_embeddings = self.patch_embeddings_layer(images) # B, NUM_PATCHES, D_MODEL
        patch_embeddings_with_class_token = torch.cat(
            tensors = (self.class_token_embedding.repeat(patch_embeddings.shape[0], 1, 1), patch_embeddings), 
            dim = 1
        ) # B, NUM_PATCHES+1, D_MODEL
        return self.dropout(patch_embeddings_with_class_token + self.positional_embedding)
    

class MSABlock(nn.Module):
    
    """Multihead Self attention block of the decoder. Uses PyTorch's inbuild MHA layer for calculating the attention sub-blocks output"""

    def __init__(self, config) -> None:
        
        super().__init__()
        
        self.attn_block = nn.MultiheadAttention(
            embed_dim = config["d_model"],
            num_heads = config["num_heads"],
            batch_first = True,
            dropout=config['attn_dropout']
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=config["d_model"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x -> B, NUM_PATCHES+1, D_MODEL
        attn_output, _ = self.attn_block(x, x, x)
        return self.layer_norm(x + attn_output) 
    
class MLPBlock(nn.Module):
    
    """FFN block of the transformer architecture"""

    def __init__(self, config) -> None:
        
        super().__init__()
        d_model = config["d_model"]
        self.dense_net = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(p=config['mlp_dropout']),
            nn.Linear(d_model*4, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.layer_norm(x + self.dense_net(x))

    
class EncoderBlock(nn.Module):
    
    """Encoder block which combines both the MSA and MLP blocks"""

    def __init__(self, config) -> None:
        
        super().__init__()
        self.msa_block = MSABlock(config)
        self.mlp_block = MLPBlock(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.mlp_block(self.msa_block(x))
    

class Encoder(nn.Module):
    
    """The encoder backbone of the ViT architecture, made up of 'n' encoder blocks"""

    def __init__(self, config) -> None: 
        
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config["num_encoders"])])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for block in self.blocks:
            x = block(x)
        
        return x
    
class ViT(nn.Module):
    
    """The ViT class which puts together the embeddings and the encoder. Returns the representation of the image usign the [CLS] token"""

    def __init__(self, config) -> None:
        
        super().__init__()
        
        self.embedding_layer = ViTEmbedding(config)
        self.encoder = Encoder(config)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # B, C, H, W
        embeddings = self.embedding_layer(images) # B, NUM_PATCHES+1, D_MODEL
        encoded_vectors = self.encoder(embeddings) # B, NUM_PATCHES+1, D_MODEL
        return encoded_vectors[:, 0, :]



class GPTEmbedding(nn.Module):
    
    """Embedding class for the GPT decoder"""

    def __init__(self,config) -> None:
        
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings = config["vocab_size"],
            embedding_dim = config["d_model"]
        )
        
        self.positional_encoding = nn.Parameter(
            data = torch.randn(size=(1, config["context_length"], config["d_model"])),
            requires_grad = True
        )
        self.dropout = nn.Dropout(p=config['emb_dropout'])
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        
        # tokens -> B, CTX_LENGTH
        token_embeddings = self.token_embedding(tokens)
        return self.dropout(self.positional_encoding[:, :tokens.shape[1], :] + token_embeddings)


class CausalSelfAttnBlock(nn.Module):
    
    """Causal self attention class - does not use PyTorch's inbuild masked MHA, since it does not accomodate both causal mask and padding mask. 
       Safe softmax has been used instead of PyTorch's softmax, to bring numerical stability in the calculations.
    """

    def __init__(self, config) -> None:
        
        super().__init__()
        assert config["d_model"] % config["num_heads"] == 0, ValueError(f"{config['d_model']} d_model should be exactly divisible by {config['num_heads']} num_heads")
        
        self.d_model = config["d_model"]
        self.head_dim = config["d_model"] // config["num_heads"]
        self.num_heads = config["num_heads"]
        self.softmax_eps = config["softmax_eps"]
        
        self.projection_layer = nn.Linear(self.d_model, self.d_model*3)
        self.out_layer = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attn_dropout = nn.Dropout(p=config['attn_dropout'])

    def _safe_softmax(self, x: torch.Tensor) -> torch.Tensor:
        
        num = torch.exp(x)
        denom = torch.exp(x).sum(dim=-1, keepdims=True) + self.softmax_eps
        return num/denom
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        
        # x -> B, CTX_LENGTH, D_MODEL
        # attn_mask -> B, CTX_LENGTH, CTX_LENGTH
        B, CTX_LENGTH = x.shape[0], x.shape[1]
        q, k, v = self.projection_layer(x).split(self.d_model, dim=2) # B, CTX_LENGTH, D_MODEL
        q = q.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2) # B, NUM_HEADS, CTX_LENGTH, HEAD_DIM
        k = k.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_k_prod = (q @ k.transpose(2, 3)) + attn_mask.unsqueeze(1) # B, NUM_HEADS, CTX_LENGTH, CTX_LENGTH
        wts = self._safe_softmax(q_k_prod / math.sqrt(self.head_dim)) # B, NUM_HEADS, CTX_LENGTH, CTX_LENGTH
        wts = self.attn_dropout(wts)
        attn_outputs = wts @ v # B, NUM_HEADS, CTX_LENGTH, HEAD_DIM
        y = attn_outputs.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1)
        return self.layer_norm(x + self.out_layer(y))
        
        

class CrossAttnBlock(nn.Module):
    
    """Cross attention block - similar version of Causal self attention block, but with the key and value tensors being the encoding of the image from ViT"""
    
    def __init__(self, config) -> None:
        
        super().__init__()

        assert config["d_model"]%config["num_heads"]==0, ValueError(f"{config['d_model']} d_model must be divisible by {config['num_heads']} num_heads")

        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.head_dim = config['d_model'] // config['num_heads']
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attn_dropout = nn.Dropout(p=config['attn_dropout'])
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor) -> torch.Tensor:
        
        # image_encoding (key, value)-> B, 1, D_MODEL
        # x (query)-> B, CTX_LENGTH, D_MODEL 
        
        # q_k_prod -> query @ key.transpose(1, 2) -> B, CTX_LENGTH, 1
        # q_k_prod @ value -> B, CTX_LENGTH, D_MODEL

        B, CTX_LENGTH, _ = x.shape        

        q = self.q_proj(x).view(B, CTX_LENGTH, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, NUM_HEADS, CTX_LENGTH, HEAD_DIM
        k = self.k_proj(image_encoding).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, NUM_HEADS, 1, HEAD_DIM
        v = self.v_proj(image_encoding).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, NUM_HEADS, 1, HEAD_DIM

        wts = F.softmax((q @ k.transpose(2, 3)) / math.sqrt(self.head_dim), dim=-1) # B, NUM_HEADS, CTX_LENGTH, 1
        wts = self.attn_dropout(wts)
        y = wts @ v # B, NUM_HEADS, CTX_LENGTH, HEAD_DIM
        y = y.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1) # B, CTX_LENGTH, D_MODEL
        return self.layer_norm(x + self.projection_layer(y))
    
class GPTDecoderBlock(nn.Module):

    """The class which puts together Causal, Cross and FFN blocks together"""

    def __init__(self, config) -> None:
        
        super().__init__()
        self.csa_block = CausalSelfAttnBlock(config)
        self.cross_attn_block = CrossAttnBlock(config)
        self.mlp_block = MLPBlock(config)
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        
        csa_out = self.csa_block(x, attn_mask)
        cross_out = self.cross_attn_block(csa_out, image_encoding)
        mlp_out = self.mlp_block(cross_out)
        return mlp_out

class GPTDecoder(nn.Module):
    
    """The backbone of the caption generator, made up of 'n' decoder blocks"""

    def __init__(self, config) -> None:
        
        super().__init__()
        self.decoder_blocks = nn.ModuleList([GPTDecoderBlock(config) for _ in range(config["num_decoders"])])
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        
        for block in self.decoder_blocks:
            x = block(x, image_encoding, attn_mask)
        
        return x
    
class GPT(nn.Module):
    
    """The caption generator part of the system. Puts together everything"""

    def __init__(self, config) -> None:
        
        super().__init__()
        self.device = config["device"]
        self.context_length = config["context_length"]
        self.softmax_eps = config["softmax_eps"]
        self.embedding = GPTEmbedding(config)
        self.decoder = GPTDecoder(config)
        self.cls_head = nn.Linear(config["d_model"], config["vocab_size"])
        # self.cls_head.weight = self.embedding.token_embedding.weight
        # removed weight tying as it lead to slower convergence
        self.ignore_index = config["ignore_index"]
    
    def _create_mask(self, context_length: int, attn_mask: torch.Tensor) -> torch.Tensor:
        
        mask = torch.triu(
            input = torch.ones(size=(context_length, context_length), requires_grad = False)*float("-inf"),
            diagonal = 1
        ).unsqueeze(0).repeat(attn_mask.shape[0], 1, 1)
        mask = mask.to(self.device)
        for i in range(mask.shape[0]):
            mask[i, attn_mask[i].logical_not(), :] = float("-inf")
        return mask # B, CTX_LENGTH, CTX_LENGTH
        
    
    def forward(self, tokens: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor, targets: torch.Tensor=None) -> Tuple[torch.Tensor]:
        
        # tokens -> B, CTX_LENGTH
        # image_encoding -> B, 1, D_MODEL
        # attn_mask -> B, CTX_LENGTH
        
        embeddings = self.embedding(tokens) # B, CTX_LENGTH, D_MODEL
        mask = self._create_mask(tokens.shape[1], attn_mask)
        decoder_out = self.decoder(embeddings, image_encoding, mask) # B, CTX_LENGTH, D_MODEL
        logits = self.cls_head(decoder_out) # B, CTX_LENGTH, VOCAB_SIZE
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=self.ignore_index)
        
        return logits, loss


class ImageCaptionModel(nn.Module):
    
    """This class is the main class. Puts together both ViT and GPT. Lot more useful functions need to be added if someone uses them"""
    
    def __init__(self, config) -> None:
        
        super().__init__()
        
        self.device = config['device']
        self.is_vit_pretrained = False
        if config['vit_kwargs']["pretrained_model_name"] is not None:
            self.is_vit_pretrained = True
            self.vit = timm.create_model(
                model_name = config['vit_kwargs']["pretrained_model_name"],
                pretrained=True,
                num_classes = 0,
                global_pool = 'avg'
            )
            config["vit_kwargs"]["d_model"] = self.vit.embed_dim
        else:   
            self.vit = ViT(config['vit_kwargs'])
        self.gpt = GPT(config['gpt_kwargs'])
        self.dimension_mapping_layer = nn.Linear(config["vit_kwargs"]['d_model'], config["gpt_kwargs"]['d_model'])
        
    def forward(self, image: torch.Tensor, tokens: torch.Tensor, attn_mask: torch.Tensor, targets: torch.Tensor=None) -> Tuple[torch.Tensor]:
        
        image_encoding = self.vit(image)
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding[:, None, :])
        return self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, targets)

    
    @torch.inference_mode()
    def generate(self, 
                 image: torch.Tensor, 
                 sos_token: int,
                 eos_token: int,
                 max_len: int=40) -> List[int]:

        image_encoding: torch.Tensor = self.vit(image)
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding[:, None, :])

        tokens = torch.tensor(data=[[sos_token]], requires_grad=False).to(self.device)
        attn_mask = torch.tensor(data=[[1]], requires_grad=False).to(self.device)

        while tokens.shape[1]<max_len and tokens[0, -1]!=eos_token:
            logits, _ = self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, None) # 1, N+1, D_MODEL
            next_token = torch.argmax(logits[0, -1, :], dim=0).item()
            tokens = torch.cat(
                (tokens, torch.tensor([[next_token]], requires_grad=False)),
                dim = -1
            ).to(self.device)
            attn_mask = torch.cat(
                (attn_mask, torch.tensor([[1]], requires_grad=False)),
                dim = -1
            ).to(self.device)
        
        return list(tokens[0])
    
    @classmethod
    def from_pretrained(cls, checkpoint, device):

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"{checkpoint} does not exist")

        cp = torch.load(checkpoint, map_location=device)
        cp['model_config']['device'] = device
        cp['model_config']['vit_kwargs']['device'] = device
        cp['model_config']['gpt_kwargs']['device'] = device

        model = cls(cp['model_config'])
        model.load_state_dict(cp['model_state_dict'])
        model = model.to(device)
        return model