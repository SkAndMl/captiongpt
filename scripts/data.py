import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
from PIL import Image
from scripts.constants import *
import random
from scripts.tokenizer import Tokenizer

image_folder = "data/flickr30k_images"
csv_file_path = "data/results.csv"

data = pd.read_csv(csv_file_path, delimiter="|")
data.drop_duplicates(subset=['image_name'], inplace=True)
data.drop(columns=' comment_number', axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)
data.rename({" comment": "comment"}, axis=1, inplace=True)
data.iloc[:, 0] = image_folder + "/" + data.iloc[:, 0]
tokenizer = Tokenizer(texts=data['comment'].tolist(), pad_token='/')


class ImageCaptionDataset(Dataset):
    
    def __init__(self, dataframe: pd.DataFrame, image_size: int, context_length: int) -> None:
        
        assert dataframe.columns[0] == 'image_name', ValueError("The first column should be the path to the image")
        assert dataframe.columns[1] == "comment", ValueError("The second column should be named 'comment'")
        
        self.context_length = context_length
        self.df = dataframe
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor()
        ])
            
    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        
        image, text = Image.open(self.df.iloc[idx, 0]), self.df.iloc[idx, 1]
        image_tensor = self.transform(image)
        op = tokenizer(text, max_len=self.context_length+1)
        tokens, attention_mask = op['input_ids'], op['attention_mask']
        return image_tensor, tokens, attention_mask
    

def prepare_data(train_size: float, image_size: int, batch_size: int) -> Tuple[DataLoader]:
 
    idxs = set(range(data.shape[0]))
    train_idxs = random.sample(sorted(idxs), k=int(len(idxs)*train_size))
    test_idxs = list(idxs.difference(set(train_idxs)))

    train_data = data.copy(deep=True).iloc[train_idxs, :].reset_index(drop=True)
    test_data = data.copy(deep=True).iloc[test_idxs, :].reset_index(drop=True)   


    train_dataset = ImageCaptionDataset(
        dataframe = train_data,
        image_size = image_size
    )

    test_dataset = ImageCaptionDataset(
        dataframe = test_data,
        image_size = image_size
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