from scripts.model import ImageCaptionModel
from scripts.constants import *
from scripts.data import create_tokenizer
import torch
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
image_caption_model = ImageCaptionModel(vit_kwargs, gpt_kwargs)
image_caption_model.load_state_dict(
    state_dict=torch.load("checkpoints/image_caption_model.pt", map_location=device)
)

transform = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

tokenizer = create_tokenizer(special_tokens_dict=special_tokens_dict)

def caption_image(file_path: str, max_len: int=40) -> str:

    image_tensor = transform(Image.open(file_path)).unsqueeze(0)
    tokens = image_caption_model.generate(image_tensor, sos_token=tokenizer.bos_token_id,
                                          eos_token=tokenizer.eos_token_id,
                                          max_len=max_len)
    return tokenizer.decode(token_ids=tokens)

if __name__ == "__main__":
    print(caption_image(file_path="sample.jpeg"))