from scripts.model import ImageCaptionModel
from scripts.constants import *
from scripts.data import create_tokenizer
import torch
from torchvision import transforms
from PIL import Image
import argparse

transform = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

tokenizer = create_tokenizer(special_tokens_dict=special_tokens_dict)

def caption_image(file_path: str, checkpoint: str, device: str="cpu", max_len: int=40) -> str:

    image_tensor = transform(Image.open(file_path)).unsqueeze(0)
    image_caption_model = ImageCaptionModel(vit_kwargs, gpt_kwargs, device=device)
    image_caption_model.load_state_dict(
        state_dict=torch.load(checkpoint, map_location=device)
    )

    tokens = image_caption_model.generate(image_tensor, sos_token=tokenizer.bos_token_id,
                                          eos_token=tokenizer.eos_token_id,
                                          max_len=max_len)
    return tokenizer.decode(token_ids=tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencing the image caption model")
    parser.add_argument("--file_path", type=str, required=True, help="Image file path for captioning")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/image_caption_model.pt", help="File path for pt file")
    parser.add_argument("--max_len", type=int, default=40, help="Maximum length of the caption")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the inference")
    args = parser.parse_args()
    
    caption: str = caption_image(file_path=args.file_path, checkpoint=args.checkpoint,
                                 device=args.device, max_len=args.max_len)
    print(caption)