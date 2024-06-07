from scripts.model import ImageCaptionModel
from scripts.data import prepare_data, tokenizer
from scripts.constants import config, device
import torch
import os
from datetime import datetime
import argparse
import logging

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


config.gpt_kwargs['vocab_size'] = tokenizer.vocab_size
config.gpt_kwargs["ignore_index"] = tokenizer.stoi[tokenizer.pad_token]

logger.info("Preparing data...")
train_dl, test_dl = prepare_data()
logger.info("Data preparation complete.")

class Trainer:

    def __init__(self, model_config, train_config, dls) -> None:

        self.model = ImageCaptionModel(model_config)
        self.train_config = train_config
        self.train_dl, self.test_dl = dls
    
    def fit(self):

        self.metrics = {
            "train_loss": [],
            "test_loss": []
        }

        # train_config has to atleast consist of epochs and freeze_epochs
        if self.model.is_vit_pretrained and self.train_config.freeze_epochs>0:
            for p in self.model.vit.parameters():
                p.requires_grad = False
        
        self.optimizer = torch.optim.Adam(
            params=filter(lambda p:p.requires_grad, self.model.parameters()),
            lr = self.train_config.lr
        )

        for _ in range(self.train_config.freeze_epochs):
            self.metrics['train_loss'].append(self._train())
            self.metrics['test_loss'].append(self._eval())
        
        if self.model.is_vit_pretrained and self.train_config.freeze_epochs>0:
            for p in self.model.vit.parameters():
                p.requires_grad = True
        
        self.optimizer = torch.optim.Adam(
            params=filter(lambda p:p.requires_grad, self.model.parameters()),
            lr = self.train_config.lr
        )

        for _ in range(self.train_config.freeze_epochs, self.train_config.epochs):
            self.metrics['train_loss'].append(self._train())
            self.metrics['test_loss'].append(self._eval()) 

        return self.metrics

    def _train(self):

        self.model.train()
        total_loss = 0
        for image, tokens, attn_mask in self.train_dl:
        
            input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
            attn_mask = attn_mask[:, :-1]
            image, input_tokens, target_tokens, attn_mask = image.to(device), input_tokens.to(device), \
                                                            target_tokens.to(device), attn_mask.to(device)
            _, loss = self.model(image, input_tokens, attn_mask, target_tokens)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(self.train_dl)

    
    def _eval(self):

        self.model.eval()
        total_loss = 0
        for image, tokens, attn_mask in self.test_dl:
        
            input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
            attn_mask = attn_mask[:, :-1]
            image, input_tokens, target_tokens, attn_mask = image.to(device), input_tokens.to(device), \
                                                            target_tokens.to(device), attn_mask.to(device)
            
            _, loss = self.model(image, input_tokens, attn_mask, target_tokens)
            total_loss += loss.item()
        
        return total_loss / len(self.test_dl)


    def save(self, file_path):

        torch.save(
            obj=self.model.state_dict(),
            f = file_path
        )


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model_file_name', type=str, default=None)
    parser.add_argument('--freeze_epochs', type=int, default=0)

    args = parser.parse_args()

    assert args.freeze_epochs <= args.epochs, ValueError(f"'freeze_epochs': {args.freeze_epochs} must be <= 'epochs': {args.epochs}")    

    train_config = {
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "lr": args.lr
    }

    trainer = Trainer(model_config=config,
                      train_config=train_config,
                      dls=(train_dl, test_dl))
    logger.info("Starting the training process...")
    trainer.fit()
    logger.info("Training process completed")