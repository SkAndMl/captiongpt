import pandas as pd
import torch
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Union, Dict
from PIL import Image
from scripts.constants import IMG_SIZE, BS, TRAIN_SIZE
import random


image_folder = "data/flickr30k_images"
csv_file_path = "data/results.csv"

class ImageCaptionDataset(Dataset):
    
    def __init__(self, dataframe: pd.DataFrame, image_size: int=IMG_SIZE) -> None:
        
        assert dataframe.columns[0] == 'image_name', ValueError("The first column should be the path to the image")
        assert dataframe.columns[1] == "comment", ValueError("The second column should be named 'comment'")
        
     
        self.df = dataframe
        self.transform = transforms.Compose([
            transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
            
    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, str]]:
        
        image, text = Image.open(self.df.iloc[idx, 0]), self.df.iloc[idx, 1]
        image_tensor = self.transform(image)
        return image_tensor, text
    

def prepare_data() -> Tuple[DataLoader]:
 
    data = pd.read_csv(csv_file_path, delimiter="|")
    data.drop_duplicates(subset=['image_name'], inplace=True)
    data.drop(columns=' comment_number', axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.rename({" comment": "comment"}, axis=1, inplace=True)
    data.iloc[:, 1] = "[SOS] " + data.iloc[:, 1] + " [EOS]"
    data.iloc[:, 0] = image_folder + "/" + data.iloc[:, 0]


    idxs = set(range(data.shape[0]))
    train_idxs = random.sample(sorted(idxs), k=int(len(idxs)*TRAIN_SIZE))
    test_idxs = list(idxs.difference(set(train_idxs)))

    train_data = data.copy(deep=True).iloc[train_idxs, :].reset_index(drop=True)
    test_data = data.copy(deep=True).iloc[test_idxs, :].reset_index(drop=True)   


    train_dataset = ImageCaptionDataset(
        dataframe = train_data,
        image_size = IMG_SIZE
    )

    test_dataset = ImageCaptionDataset(
        dataframe = test_data,
        image_size = IMG_SIZE
    )

    train_dl = DataLoader(
        dataset = train_dataset,
        batch_size = BS,
        shuffle = True,
        num_workers = 2
    )

    test_dl = DataLoader(
        dataset = test_dataset,
        batch_size = BS,
        shuffle = False
    )

    return train_dl, test_dl


def create_tokenizer(tokenizer_name: str='gpt2', 
                    add_special_tokens: bool = True,
                    special_tokens_dict: Dict[str, str] = None) -> GPT2Tokenizer:
    
    if add_special_tokens and special_tokens_dict is None:
        raise ValueError("special_tokens_dict cannot be None when 'add_special_tokens' is set to True")
    
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer