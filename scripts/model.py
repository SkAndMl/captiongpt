import torch
from torch import nn
import torch.nn.functional as F
from scripts.constants import *
from transformers import GPT2Tokenizer
from typing import Tuple
import math
from scripts.data import create_tokenizer

tokenizer = create_tokenizer(
    special_tokens_dict=special_tokens_dict
)

VOCAB_SIZE = tokenizer.vocab_size + len(special_tokens_dict)
IGNORE_INDEX = tokenizer.get_vocab()[tokenizer.pad_token]

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
            tensors = (self.class_token_embedding.repeat(patch_embeddings.shape[0], 1, 1), patch_embeddings), 
            dim = 1
        ) # B, NUM_PATCHES+1, D_MODEL
        return patch_embeddings_with_class_token + self.positional_embedding
    

class MSABlock(nn.Module):
    
    def __init__(self, 
                d_model: int = D_MODEL_VIT,
                num_heads: int = NUM_HEADS_VIT) -> None:
        
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
                num_heads: int = NUM_HEADS_VIT) -> None:
        
        super().__init__()
        self.msa_block = MSABlock(d_model=d_model, num_heads=num_heads)
        self.mlp_block = MLPBlock(d_model=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.mlp_block(self.msa_block(x))
    

class Encoder(nn.Module):
    
    def __init__(self, 
                num_encoders: int = NUM_ENCODERS_VIT,
                d_model: int = D_MODEL_VIT,
                num_heads: int = NUM_HEADS_VIT) -> None: 
        
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_encoders)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for block in self.blocks:
            x = block(x)
        
        return x
    
class ViT(nn.Module):
    
    def __init__(self, 
                num_encoders: int=NUM_ENCODERS_VIT,
                num_heads: int = NUM_HEADS_VIT,
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



class TextEmbedding(nn.Module):
    
    def __init__(self,
                vocab_size: int = VOCAB_SIZE,
                d_model: int = D_MODEL_GPT,
                context_length: int = CTX_LENGTH) -> None:
        
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = d_model
        )
        
        self.positional_encoding = nn.Parameter(
            data = torch.randn(size=(1, context_length, d_model)),
            requires_grad = True
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        
        # tokens -> B, CTX_LENGTH
        return self.positional_encoding + self.token_embedding(tokens)


class CausalSelfAttnBlock(nn.Module):
    
    def __init__(self,
                d_model: int=D_MODEL_GPT,
                num_heads: int = NUM_HEADS_GPT,
                softmax_eps: float = SOFTMAX_DENOM_EPS) -> None:
        
        super().__init__()
        assert d_model % num_heads == 0, ValueError(f"{d_model} d_model should be exactly divisible by {num_heads} num_heads")
        
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.softmax_eps = softmax_eps
        
        self.projection_layer = nn.Linear(d_model, d_model*3)
        self.out_layer = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        
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
        attn_outputs = wts @ v # B, NUM_HEADS, CTX_LENGTH, HEAD_DIM
        y = attn_outputs.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1)
        return self.layer_norm(x + self.out_layer(y))
        
        

class CrossAttnBlock(nn.Module):
    
    def __init__(self,
                d_model: int=D_MODEL_GPT,
                num_heads: int = NUM_HEADS_GPT) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.cross_attn_block = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor) -> torch.Tensor:
        
        # image_encoding (key, value)-> B, 1, D_MODEL
        # x (query)-> B, CTX_LENGTH, D_MODEL 
        
        # q_k_prod -> query @ key.transpose(1, 2) -> B, CTX_LENGTH, 1
        # q_k_prod @ value -> B, CTX_LENGTH, D_MODEL
        
        q_k_prod = (x @ image_encoding.transpose(1, 2)) / math.sqrt(self.d_model) # B, CTX_LENGTH, 1
        wts = F.softmax(q_k_prod, dim=-1) 
        out = wts @ image_encoding # B, CTX_LENGTH, D_MODEL
        return out
        
    
    
class GPTDecoderBlock(nn.Module):
    
    def __init__(self,
                d_model: int=D_MODEL_GPT,
                num_heads: int=NUM_HEADS_GPT,
                softmax_eps: float = SOFTMAX_DENOM_EPS) -> None:
        
        super().__init__()
        self.csa_block = CausalSelfAttnBlock(d_model, num_heads, softmax_eps)
        self.cross_attn_block = CrossAttnBlock(d_model, num_heads)
        self.mlp_block = MLPBlock(d_model)
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        
        csa_out = self.csa_block(x, attn_mask)
        cross_out = self.cross_attn_block(csa_out, image_encoding)
        mlp_out = self.mlp_block(cross_out)
        return mlp_out

class GPTDecoder(nn.Module):
    
    def __init__(self, 
                num_decoders: int = NUM_DECODERS_GPT,
                d_model: int = D_MODEL_GPT,
                num_heads: int = NUM_HEADS_GPT,
                softmax_eps: float = SOFTMAX_DENOM_EPS) -> None:
        
        super().__init__()
        self.decoder_blocks = nn.ModuleList([GPTDecoderBlock(d_model, num_heads, softmax_eps) for _ in range(num_decoders)])
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        
        for block in self.decoder_blocks:
            x = block(x, image_encoding, attn_mask)
        
        return x
    
class GPT(nn.Module):
    
    def __init__(self,
                vocab_size: int = VOCAB_SIZE,
                d_model: int = D_MODEL_GPT,
                context_length: int = CTX_LENGTH,
                num_decoders: int = NUM_DECODERS_GPT,
                num_heads: int = NUM_HEADS_GPT,
                softmax_eps: float = SOFTMAX_DENOM_EPS,
                ignore_index: int = IGNORE_INDEX,
                device: str = "cpu") -> None:
        
        super().__init__()
        self.device = device
        self.context_length = context_length
        self.softmax_eps = softmax_eps
        self.embedding = TextEmbedding(vocab_size, d_model, context_length)
        self.decoder = GPTDecoder(num_decoders, d_model, num_heads, softmax_eps)
        self.cls_head = nn.Linear(d_model, vocab_size)
        self.ignore_index = ignore_index
        
    
    def _safe_softmax(self, x: torch.Tensor) -> torch.Tensor:
        
        num = torch.exp(x)
        denom = torch.exp(x).sum(dim=-1, keepdims=True) + self.softmax_eps
        return num/denom
    
    
    def _create_mask(self, attn_mask: torch.Tensor) -> torch.Tensor:
        
        mask = torch.triu(
            input = torch.ones(size=(self.context_length, self.context_length), requires_grad = False)*float("-inf"),
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
        mask = self._create_mask(attn_mask)
        decoder_out = self.decoder(embeddings, image_encoding, mask) # B, CTX_LENGTH, D_MODEL
        logits = self.cls_head(decoder_out) # B, CTX_LENGTH, VOCAB_SIZE
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=self.ignore_index)
        
        return logits, loss


class ImageCaptionModel(nn.Module):
    
    def __init__(self, vit_kwargs, gpt_kwargs) -> None:
        
        super().__init__()
        
        self.vit = ViT(**vit_kwargs)
        self.gpt = GPT(**gpt_kwargs)
        self.dimension_mapping_layer = nn.Linear(vit_kwargs['d_model'], gpt_kwargs['d_model'])
        
    def forward(self, image: torch.Tensor, tokens: torch.Tensor, attn_mask: torch.Tensor, targets: torch.Tensor=None) -> Tuple[torch.Tensor]:
        
        image_encoding = self.vit(image)
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding)
        return self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, targets)