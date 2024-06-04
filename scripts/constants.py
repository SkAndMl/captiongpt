import torch

TRAIN_SIZE = 0.9
BS = 32
IMG_SIZE = 128
CTX_LENGTH = 128
BS = 32
NUM_ENCODERS_VIT = 6
NUM_HEADS_VIT = 4
PS = 8
C = 3
D_MODEL_VIT = PS**2*C
NUM_PATCHES = (IMG_SIZE*IMG_SIZE)//(PS*PS)
D_MODEL_GPT = 384
NUM_DECODERS_GPT = 6
NUM_HEADS_GPT = 6
SOFTMAX_DENOM_EPS = 1e-9
device = "cuda" if torch.cuda.is_available() else "cpu"

special_tokens_dict = {
    "bos_token": "[SOS]",
    "eos_token": "[EOS]",
    "pad_token": "[PAD]"
}

vit_kwargs = {
    "num_encoders" : NUM_ENCODERS_VIT,
    "num_heads": NUM_HEADS_VIT,
    "num_patches": NUM_PATCHES,
    "patch_size": PS,
    "channels": C,
    "d_model": D_MODEL_VIT
}

gpt_kwargs = {
    "d_model": D_MODEL_GPT,
    "context_length": CTX_LENGTH,
    "num_decoders": NUM_DECODERS_GPT,
    "softmax_eps": SOFTMAX_DENOM_EPS,
    "num_heads": NUM_HEADS_GPT,
    "device": device
# should add ignore_index and vocab_size before sending to the model
}