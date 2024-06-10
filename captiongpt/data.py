import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
from PIL import Image
from captiongpt.params import *
import random
from captiongpt.tokenizer import Tokenizer, TokenizerHF

image_folder = "data/flickr30k_images"
csv_file_path = "data/results.csv"

data = pd.read_csv(csv_file_path, delimiter="|")
data.drop_duplicates(subset=['image_name'], inplace=True)
data.drop(columns=' comment_number', axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)
data.rename({" comment": "comment"}, axis=1, inplace=True)
data.iloc[:, 0] = image_folder + "/" + data.iloc[:, 0]
data['comment'] = '[BOS] ' + data['comment'] + ' [EOS]'
# tokenizer = Tokenizer(texts=data['comment'].tolist(), pad_token='/')
tokenizer = TokenizerHF(tokenizer_name="gpt2", special_tokens_dict={"bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]"})


class ImageCaptionDataset(Dataset):
    
    def __init__(self, dataframe: pd.DataFrame, image_size: int, context_length: int) -> None:
        
        assert dataframe.columns[0] == 'image_name', ValueError("The first column should be the path to the image")
        assert dataframe.columns[1] == "comment", ValueError("The second column should be named 'comment'")
        
        self.context_length = context_length
        self.df = dataframe
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        
        image, text = Image.open(self.df.iloc[idx, 0]), self.df.iloc[idx, 1]
        image_tensor = self.transform(image)
        op = tokenizer(text, max_len=self.context_length+1)
        tokens, attention_mask = op['input_ids'].squeeze(), op['attention_mask'].squeeze()
        return image_tensor, tokens, attention_mask
    

def prepare_data(train_size: float, image_size: int, batch_size: int) -> Tuple[DataLoader]:
 
    idxs = set(range(data.shape[0]))
    train_idxs = random.sample(sorted(idxs), k=int(len(idxs)*train_size))
    test_idxs = list(idxs.difference(set(train_idxs)))

    train_data = data.copy(deep=True).iloc[train_idxs, :].reset_index(drop=True)
    test_data = data.copy(deep=True).iloc[test_idxs, :].reset_index(drop=True)   


    train_dataset = ImageCaptionDataset(
        dataframe = train_data,
        image_size = image_size,
        context_length = ctx_length
    )

    test_dataset = ImageCaptionDataset(
        dataframe = test_data,
        image_size = image_size,
        context_length = ctx_length
    )

    train_dl = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2
    )

    test_dl = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False
    )

    return train_dl, test_dl