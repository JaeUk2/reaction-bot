import torch
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from utils import set_seed, compute_metrics
from load_data import SentimentDataset
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import wandb

def train(cfg):

    ## wandb
    wandb.init(entity='sympathy', project=cfg.wandb.project_name, name=cfg.wandb.exp_name)

    ## device 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ## load tokenizer, model
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.model_name, num_labels=7)
    model.to(device)

    ## load dataset
    train_dataset = SentimentDataset(cfg.data.train_data, tokenizer)
    valid_dataset = SentimentDataset(cfg.data.valid_data, tokenizer)

    # Scheduler = ["CosineAnnealing", "CyclicLR", "LambdaLR"]

    if cfg.train.scheduler == "CosineAnnealing":
        optimizer = optim.AdamW(model.parameters(),lr = cfg.train.lr, weight_decay=cfg.train.weight_decay, eps = cfg.train.eps)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.T_0, T_mult=cfg.train.T_mult, eta_min=cfg.train.eta_min)
        optimizers = (optimizer,scheduler)

    elif cfg.train.scheduler == "CyclicLR":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=1000, step_size_down=2000, mode='triangular')
        optimizers = (optimizer,scheduler)

    else:
        optimizer = optim.AdamW(model.parameters(),lr = cfg.train.lr, weight_decay=cfg.train.weight_decay, eps = cfg.train.eps)
        scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** cfg.train.epoch)
        optimizers = (optimizer,scheduler)


    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir,          # 출력 폴더
        num_train_epochs=cfg.train.epoch,                   # 학습 에폭 수
        per_device_train_batch_size=cfg.train.batch_size,   # GPU당 학습 배치 크기
        per_device_eval_batch_size=cfg.train.batch_size,    # GPU당 평가 배치 크기기
        warmup_steps=cfg.train.warmup_steps,                # 학습률 스케줄링을 위한 warm up 과정 스텝 수. 이동안은 학습률이 천천히 올라간다.
        weight_decay=cfg.train.weight_decay,                # 가중치 감쇠 (weight decay)
        logging_dir='./logs',                               # 로그 기록을 위한 폴더
        learning_rate= cfg.train.lr,
        logging_steps=cfg.train.logging_step,
        save_total_limit=3,
        load_best_model_at_end=True,
        save_steps=1000,
        evaluation_strategy='steps',
        eval_steps=1000,
        report_to=["wandb"]
    )

    trainer = Trainer(
        model=model,                         # 학습할 모델
        args=training_args,                  # 학습 인자
        train_dataset=train_dataset,         # 학습 데이터 셋
        eval_dataset=valid_dataset,          # 평가 데이터 셋
        compute_metrics=compute_metrics,
        optimizers = optimizers
    )

    trainer.train()

    model.save_pretrained(cfg.model.HF_model) 
    torch.save(model.state_dict(), cfg.model.saved_model)

    ## wandb finish
    wandb.finish()