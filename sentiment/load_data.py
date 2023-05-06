import torch
import pandas as pd


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokneizer):
        self.dataset = pd.read_csv(data_path)
        self.tokenizer = tokneizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        talk = self.dataset.iloc[idx]['talk']
        label = self.dataset.iloc[idx]['emotion']
        talk_encoding = self.tokenizer(talk, padding="max_length", max_length=128, truncation=True)

        result = {key: torch.LongTensor(val) for key, val in talk_encoding.items()}
        result['labels'] = torch.tensor(label)

        return result
