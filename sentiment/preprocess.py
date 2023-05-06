# AI HUB의 감성 대화 말뭉치의 라벨링 데이터를 사용하였을 때, 전처리 결과

import pandas as pd

data = pd.read_json('../data/train.json')

emotion_list = []
talk_data = []

for i in range(len(data)):
    # 대분류 감정
    emotion_list.append(data['profile'][i]['emotion']['type'][:2])
    
    # 감정이 담긴 발화
    talk_data.append(data['talk'][i]['content']['HS01'])

new_data = pd.DataFrame({'emotion':emotion_list, 'talk':talk_data})
new_data.to_csv('../data/new_train.csv')