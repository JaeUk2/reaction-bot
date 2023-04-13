import torch
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from utils import set_seed
from load_data import SentimentDataset

## Reset the Memory
torch.cuda.empty_cache()

## device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

## seed 설정
set_seed(42)

## load tokenizer, model
tokenizer = PreTrainedTokenizerFast.from_pretrained('klue/roberta-small')
model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-small', num_labels=6)
model.to(device)

## load dataset
train_dataset = SentimentDataset('~/data/re_train.csv', tokenizer)
valid_dataset = SentimentDataset('~/data/re_valid.csv', tokenizer)


training_args = TrainingArguments(
    output_dir='./results',          # 출력 폴더
    num_train_epochs=10,              # 학습 에폭 수
    per_device_train_batch_size=16,  # GPU당 학습 배치 크기
    per_device_eval_batch_size=16,   # GPU당 평가 배치 크기기
    warmup_steps=500,                # 학습률 스케줄링을 위한 warm up 과정 스텝 수. 이동안은 학습률이 천천히 올라간다.
    weight_decay=0.01,               # 가중치 감쇠 (weight decay)
    logging_dir='./logs',            # 로그 기록을 위한 폴더
    logging_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    save_steps=100,
    evaluation_strategy='steps',
    eval_steps=100,
)

trainer = Trainer(
    model=model,                         # 학습할 모델
    args=training_args,                  # 학습 인자
    train_dataset=train_dataset,         # 학습 데이터 셋
    eval_dataset=valid_dataset             # 평가 데이터 셋
)

trainer.train()

model.save_pretrained('.../model_result/test')