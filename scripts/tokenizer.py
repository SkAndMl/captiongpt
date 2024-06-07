from typing import List, Dict
import warnings
import torch

class Tokenizer:
    
    def __init__(self, texts: List[str], pad_token: str=None) -> None:
        
        if pad_token is None:
            warnings.warn(f"'pad_token' must be set if padding is to be done")
        
        self.texts = texts
        self.pad_token = pad_token
        self.pad_token_id = None
        self._create_mapping()
    
    def _create_mapping(self) -> None:
        
        complete_text: str = sorted(list(set(". ".join(self.texts))))
        self.itos = {i:ch for i, ch in enumerate(complete_text)}
        self.stoi = {ch: i for i, ch in enumerate(complete_text)}
        
        if self.pad_token is not None:
            if self.stoi.get(self.pad_token, None):
                raise ValueError(f"{self.pad_token} is present in the vocabulary")
            
            self.pad_token_id = len(self.itos)
            self.itos[self.pad_token_id] = self.pad_token
            self.stoi[self.pad_token] = self.pad_token_id
        
        self.vocab_size = len(self.itos)
    
    def encode(self, text: str, max_len: int, padding: bool=True) -> Dict[str, torch.Tensor]:
        
        tokens = [self.stoi[ch] for ch in text]
        attn_mask = [1 for _ in range(len(tokens))]
        if padding:
            if self.pad_token is None:
                raise ValueError("padding cannot be done when 'pad_token' is not set")
            if len(tokens) < max_len: 
                pad_len = max_len - len(tokens)
                tokens += [self.pad_token_id]*pad_len
                attn_mask += [0]*pad_len
        
        return {
            "input_ids": torch.tensor(tokens[:max_len], requires_grad=False),
            "attention_mask": torch.tensor(tokens[:max_len], requires_grad=False)
        }
    
    def decode(self, token_ids: List[int]) -> str:
        
        return ''.join([self.itos[token] for token in token_ids])
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)