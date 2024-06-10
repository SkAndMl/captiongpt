from captiongpt.model import ImageCaptionModel
from captiongpt.params import *
from captiongpt.data import tokenizer
from torchvision import transforms
from PIL import Image
import argparse

transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

config['gpt_kwargs']['vocab_size'] = tokenizer.vocab_size
config['gpt_kwargs']['ignore_index'] = tokenizer.get_vocab()[tokenizer.pad_token]

def caption_image(file_path: str, checkpoint: str, device: str="cpu", max_len: int=40) -> str:
    
    image_tensor = transform(Image.open(file_path)).unsqueeze(0)
    image_caption_model = ImageCaptionModel.from_pretrained(checkpoint, device)
    tokens = image_caption_model.generate(image_tensor, 
                                          sos_token=tokenizer.get_vocab()['[BOS]'],
                                          eos_token=tokenizer.get_vocab()['[EOS]'],
                                          max_len=max_len)
    return tokenizer.decode(token_ids=[token.item() for token in tokens])


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