from captiongpt.model import ImageCaptionModel
from captiongpt.data import prepare_data, tokenizer
from captiongpt.params import *
from captiongpt.utils import clear_console
import torch
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from PIL import Image

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Preparing data...")
train_dl, test_dl = prepare_data(train_size, img_size, bs)
logger.info("Data preparation complete.")

config['gpt_kwargs']['vocab_size'] = tokenizer.vocab_size
config['gpt_kwargs']['ignore_index'] = tokenizer.get_vocab()[tokenizer.pad_token]

class Trainer:

    def __init__(self, model_config, train_config, dls, tokenizer) -> None:
        
        self.device = train_config['device']
        self.model = ImageCaptionModel(model_config).to(self.device)
        self.train_config = train_config
        self.model_config = model_config
        self.train_dl, self.test_dl = dls
        self.metrics = pd.DataFrame(columns=["epoch", "train_loss", "test_loss", "train_perplexity", "test_perplexity", "elapsed_time"])
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def fit(self):
        start_time = time.time()
        # train_config has to atleast consist of epochs and freeze_epochs
        if self.model.is_vit_pretrained and self.train_config["freeze_epochs"]>0:
            for p in self.model.vit.parameters():
                p.requires_grad = False
        
        self.optimizer = torch.optim.Adam([
                {"params": self.model.vit.parameters(), "lr": 0}, # setting to 0 will not update the frozen params
                {"params": self.model.dimension_mapping_layer.parameters(), "lr": self.train_config['lr']},
                {"params": self.model.gpt.parameters(), "lr": self.train_config['lr']}
            ],
            weight_decay=self.train_config['weight_decay']
        )

        for epoch in range(self.train_config["freeze_epochs"]):
            
            train_loss, train_perplexity = self._train()
            test_loss, test_perplexity = self._eval()
            elapsed_time = time.time() - start_time
            new_row = pd.DataFrame(data={
                "epoch": [epoch+1],
                "train_loss": [train_loss],
                "test_loss": [test_loss],
                "elapsed_time": [elapsed_time],
                "train_perplexity": [train_perplexity],
                "test_perplexity": [test_perplexity]
            })
            self.metrics = pd.concat([self.metrics, new_row], axis=0, ignore_index=True)

            clear_console()
            print(self.metrics.to_string(index=False))
            
        
        if self.model.is_vit_pretrained and self.train_config["freeze_epochs"]>0:
            for p in self.model.vit.parameters():
                p.requires_grad = True
        
        self.optimizer.param_groups[0]['lr'] = self.train_config['lr']  # unfreeze vit params

        for epoch in range(self.train_config["freeze_epochs"], self.train_config["epochs"]):
            train_loss = self._train()
            test_loss = self._eval()
            elapsed_time = time.time() - start_time
            new_row = pd.DataFrame(data={
                "epoch": [epoch+1],
                "train_loss": [train_loss],
                "test_loss": [test_loss],
                "elapsed_time": [elapsed_time],
                "train_perplexity": [train_perplexity],
                "test_perplexity": [test_perplexity]
            })
            self.metrics = pd.concat([self.metrics, new_row], axis=0, ignore_index=True)

            clear_console()
            print(self.metrics.to_string(index=False))
            
        return self.metrics

    def _train(self):

        self.model.train()
        total_loss = 0
        for image, tokens, attn_mask in self.train_dl:
        
            input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
            attn_mask = attn_mask[:, :-1]
            image, input_tokens, target_tokens, attn_mask = image.to(self.device), input_tokens.to(self.device), \
                                                            target_tokens.to(self.device), attn_mask.to(self.device)
            _, loss = self.model(image, input_tokens, attn_mask, target_tokens)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        avg_loss = total_loss / len(self.train_dl)
        train_perplexity = torch.exp(torch.tensor(avg_loss))
        return avg_loss, train_perplexity.item()

    
    def _eval(self):

        self.model.eval()
        total_loss = 0
        for image, tokens, attn_mask in self.test_dl:
        
            input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
            attn_mask = attn_mask[:, :-1]
            image, input_tokens, target_tokens, attn_mask = image.to(self.device), input_tokens.to(self.device), \
                                                            target_tokens.to(self.device), attn_mask.to(self.device)
            
            _, loss = self.model(image, input_tokens, attn_mask, target_tokens)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_dl)
        test_perplexity = torch.exp(torch.tensor(avg_loss))
        return avg_loss, test_perplexity.item()
    

    def inference(self, image_path, max_len) -> str:
        image_tensor = self.transform(Image.open(image_path)).unsqueeze(0)
        tokens = self.model.generate(image_tensor, 
                                    sos_token=self.tokenizer.get_vocab()['[BOS]'],
                                    eos_token=self.tokenizer.get_vocab()['[EOS]'],
                                    max_len=max_len)
        return self.tokenizer.decode(token_ids=[token.item() for token in tokens])


    def save(self, file_path):
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_config": self.train_config,
            "model_config": self.model_config
        }

        torch.save(checkpoint, file_path)

    def plot_metrics(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['epoch'], self.metrics['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['epoch'], self.metrics['train_perplexity'], label='Train Perplexity')
        plt.plot(self.metrics['epoch'], self.metrics['test_perplexity'], label='Test Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Training and Test Perplexity Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model_file_name', type=str, default=None)
    parser.add_argument('--freeze_epochs', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    assert args.freeze_epochs <= args.epochs, ValueError(f"'freeze_epochs': {args.freeze_epochs} must be <= 'epochs': {args.epochs}")    

    train_config = {
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "lr": args.lr,
        "device": args.device,
        "weight_decay": args.weight_decay
    }

    config['device'] = device
    config['gpt_kwargs']['device'] = device
    config['vit_kwargs']['device'] = device

    trainer = Trainer(model_config=config,
                      train_config=train_config,
                      dls=(train_dl, test_dl))
    logger.info("Starting the training process...")
    trainer.fit()
    logger.info("Training process completed")
