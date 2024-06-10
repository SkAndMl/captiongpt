import torch

train_size = 0.9
bs = 16
img_size = 224
ctx_length = 256
num_encoders_vit = 8
num_heads_vit = 4
ps = 16
c = 3
d_model_vit = ps**2*c
num_patches = (img_size*img_size)//(ps*ps)
d_model_gpt = 512
num_decoders_gpt = 8
num_heads_gpt = 8
softmax_denom_eps = 1e-9
device = "cuda" if torch.cuda.is_available() else "cpu"
attn_dropout = 0
mlp_dropout = 0
emb_dropout = 0

vit_kwargs = {
    "num_encoders" : num_encoders_vit,
    "num_heads": num_heads_vit,
    "num_patches": num_patches,
    "patch_size": ps,
    "channels": c,
    "d_model": d_model_vit,
    "pretrained_model_name": None,
    "device": device,
    "emb_dropout": emb_dropout,
    "mlp_dropout": mlp_dropout,
    "attn_dropout": attn_dropout
}

gpt_kwargs = {
    "d_model": d_model_gpt,
    "context_length": ctx_length,
    "num_decoders": num_decoders_gpt,
    "softmax_eps": softmax_denom_eps,
    "num_heads": num_heads_gpt,
    "device": device,
    "emb_dropout": emb_dropout,
    "mlp_dropout": mlp_dropout,
    "attn_dropout": attn_dropout
# should add ignore_index and vocab_size before sending to the model
}

config = {
    "vit_kwargs": vit_kwargs,
    "gpt_kwargs": gpt_kwargs,
    "device": device
}