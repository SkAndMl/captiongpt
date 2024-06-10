from typing import List, Dict
import transformers
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
    
    def get_vocab(self):
        return self.stoi

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


class TokenizerHF:

    def __init__(self, tokenizer_name, special_tokens_dict=None) -> None:

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        if special_tokens_dict is None:
           warnings.warn(f"'special_tokens_dict' has not been set, using default special_tokens_dict")
           self.tokenizer.add_special_tokens({
               "bos_token": "[BOS]",
               "eos_token": "[EOS]",
               "pad_token": "[PAD]"
           }) 
           self.vocab_size = self.tokenizer.vocab_size + 3
           self.pad_token = '[PAD]'
        else:
            assert 'pad_token' in special_tokens_dict, ValueError("'pad_token' key must be present in the 'special_tokens_dict' passed")                
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.vocab_size = self.tokenizer.vocab_size + len(special_tokens_dict)
            self.pad_token = special_tokens_dict['pad_token']

    def encode(self, text, max_len, padding=True) -> Dict[str, torch.Tensor]:
        return self.tokenizer(text, max_length=max_len, padding='max_length' if padding else True, 
                              return_tensors='pt')

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids)
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def get_vocab(self):
        return self.tokenizer.get_vocab()