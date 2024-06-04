from scripts.model import ImageCaptionModel, tokenizer
from scripts.data import prepare_data
from scripts.constants import CTX_LENGTH, special_tokens_dict, vit_kwargs, gpt_kwargs, device
import torch
import os
from datetime import datetime
import argparse
import logging

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


gpt_kwargs['vocab_size'] = tokenizer.vocab_size + len(special_tokens_dict)
gpt_kwargs["ignore_index"] = tokenizer.get_vocab()[tokenizer.pad_token]

logger.info("Preparing data...")
train_dl, test_dl = prepare_data()
logger.info("Data preparation complete.")

def train_epoch():
    
    image_caption_model.train()
    total_loss = 0
    for image, text in train_dl:
        
        op = tokenizer(text, max_length=CTX_LENGTH+1, padding='max_length', truncation=True,
                      return_tensors='pt')
        tokens, attn_mask = op['input_ids'], op['attention_mask']
        input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
        attn_mask = attn_mask[:, :-1]
        
        image, input_tokens, target_tokens, attn_mask = image.to(device), input_tokens.to(device), \
                                                        target_tokens.to(device), attn_mask.to(device)
        
        
        _, loss = image_caption_model(image, input_tokens, attn_mask, target_tokens)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(train_dl)

@torch.inference_mode()
def eval_epoch():
    
    image_caption_model.eval()
    total_loss = 0
    for image, text in test_dl:
        op = tokenizer(text, max_length=CTX_LENGTH+1, padding='max_length', truncation=True,
                      return_tensors='pt')
        tokens, attn_mask = op['input_ids'], op['attention_mask']
        input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
        attn_mask = attn_mask[:, :-1]
        
        image, input_tokens, target_tokens, attn_mask = image.to(device), input_tokens.to(device), \
                                                        target_tokens.to(device), attn_mask.to(device)
        
        _, loss = image_caption_model(image, input_tokens, attn_mask, target_tokens)
        total_loss += loss.item()
        
    return total_loss / len(test_dl)


def train(epochs: int, model_file_name: str):
    

    image_caption_model.train()
    for epoch in range(epochs):
        train_loss = train_epoch()
        test_loss = eval_epoch()
        print(f"""
{epoch+1}/{epochs} train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}
        """)
    
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
        logger.info("Created checkpoint directory.")

# filename1 = datetime.now().strftime("%Y%m%d-%H%M%S")
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f"checkpoints/{model_file_name}_{date_time}.pt"
    torch.save(
        obj = image_caption_model.state_dict(),
        f = model_path
    )
    logger.info(f"Model saved as {model_path}")


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train the model.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer.')
    parser.add_argument('--model_file_name', type=str, default=None, help='Base name for saved model.')
    args = parser.parse_args()

    image_caption_model = ImageCaptionModel(gpt_kwargs=gpt_kwargs,
                                           vit_kwargs=vit_kwargs).to(device)
    optimizer = torch.optim.AdamW(
            params=image_caption_model.parameters(),
            lr=args.lr
    )
    logger.info("Starting the training process...")
    train(args.epochs, model_file_name=args.model_file_name)
    logger.info("Training process completed")