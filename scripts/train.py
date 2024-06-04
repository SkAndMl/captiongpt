from .model import ImageCaptionModel, tokenizer
from .data import prepare_data
from .constants import CTX_LENGTH, special_tokens_dict, vit_kwargs, gpt_kwargs, device
import torch


gpt_kwargs['vocab_size'] = tokenizer.vocab_size + len(special_tokens_dict)
gpt_kwargs["IGNORE_INDEX"] = tokenizer.get_vocab()[tokenizer.pad_token]

image_caption_model = ImageCaptionModel(gpt_kwargs=gpt_kwargs,
                                        vit_kwargs=vit_kwargs)
optimizer = torch.optim.AdamW(
    params = image_caption_model.parameters(),
    lr = 3e-4
)

train_dl, test_dl = prepare_data()



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


def train(epochs: int=2):
    
    image_caption_model.train()
    for epoch in range(epochs):
        train_loss = train_epoch()
        test_loss = eval_epoch()
        print(f"""
{epoch+1}/{epochs} train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}
        """)


if __name__ == "__main__":
    train(2)