import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple
from transformers import GPT2Tokenizer
from .vit import MLPBlock

tokenizer= GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

D_MODEL_TEXT = 256
NUM_HEADS_DECODER = 4
NUM_DECODERS = 4
VOCAB_SIZE = tokenizer.vocab_size
CTX_LENGTH = 256
SOFTMAX_DENOM_EPS = 1e-9

class TextEmbedding(nn.Module):
    
    def __init__(self,
                vocab_size: int = VOCAB_SIZE,
                d_model: int = D_MODEL_TEXT,
                context_length: int = CTX_LENGTH) -> None:
        
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
        return self.positional_encoding + self.token_embedding


class CausalSelfAttnBlock(nn.Module):
    
    def __init__(self,
                d_model: int=D_MODEL_TEXT,
                num_heads: int = NUM_HEADS_DECODER,
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
                d_model: int=D_MODEL_TEXT,
                num_heads: int = NUM_HEADS_DECODER) -> None:
        
        super().__init__()
        self.cross_attn_block = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor) -> torch.Tensor:
        
        # image_encoding -> B, 1, D_MODEL
        # x -> B, CTX_LENGTH, D_MODEL
        
        if image_encoding.shape[1]==1:
            image_encoding = image_encoding.repeat(1, x.shape[1], 1)
        attn_output, _ = self.cross_attn_block(x, image_encoding, image_encoding)
        return self.layer_norm(x + attn_output)
    
    
class GPTDecoderBlock(nn.Module):
    
    def __init__(self,
                d_model: int=D_MODEL_TEXT,
                num_heads: int=NUM_HEADS_DECODER,
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
                num_decoders: int = NUM_DECODERS,
                d_model: int = D_MODEL_TEXT,
                num_heads: int = NUM_HEADS_DECODER,
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
                d_model: int = D_MODEL_TEXT,
                context_length: int = CTX_LENGTH,
                num_decoders: int = NUM_DECODERS,
                num_heads: int = NUM_HEADS_DECODER,
                softmax_eps: float = SOFTMAX_DENOM_EPS,
                ignore_index: int = -100) -> None:
        
        super().__init__()
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
            input = torch.ones(size=(self.context_length, self.context_length)*float("-inf"), requires_grad = False),
            diagonal = 1
        ).unsqueeze(0).repeat(attn_mask[0], 1, 1)
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
            loss = F.cross_entropy(input=logits, target=targets, ignore_index=self.ignore_index)
        
        return logits, loss