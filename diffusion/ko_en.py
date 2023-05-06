from pororo import Pororo
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from tqdm import tqdm


en_text = []
mt = Pororo(task='translation', lang='multi')
for i in tqdm(range(len(data_text))):
    en = mt(data_text[i], src='ko', tgt='en')
    en_text.append(en)
    
# print(ko_text)

# image = dataset['train']['image']

# df = {}
# df['image'] = image
# df['text'] = ko_text
# df = Dataset.from_pandas(pd.DataFrame(df))
# new_dataset = DatasetDict({'train' : df})

# print(new_dataset)
# print(new_dataset['train'])
# print(new_dataset['train']['text'][0])