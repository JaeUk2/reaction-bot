import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from utils import set_seed, compute_metrics, num_to_label
from load_data import SentimentDataset
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler,CosineAnnealingWarmRestarts
import torch.nn.functional as F
import numpy as np
import time

def inference(model, token_dialog, tokenizer, device):
    inputs = tokenizer(token_dialog, padding="max_length", max_length=128, truncation=True, return_tensors='pt')
    result = {key: torch.LongTensor(val).to(device) for key, val in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(input_ids = result["input_ids"], token_type_ids = result["token_type_ids"], attention_mask = result["attention_mask"])
        logits = outputs['logits']
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
  
    return result

def test(cfg):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    model = torch.load(cfg.model.saved_model)
    # model = AutoModelForSequenceClassification.from_pretrained("/opt/ml/reaction-bot/sentiment/results/checkpoint-35000", num_labels=7)
    # torch.save 사용 후 이 코드는 사용안함 torch.load 사용
    
    model.parameters
    model.to(device)
    
    dialog = input()
    
    start_time = time.time()
    result = inference(model, dialog, tokenizer, device)
    
    print(f"inference time : {time.time() - start_time}")
    
    print(num_to_label(result))
    # 5.2초 , 0.48초 

### 하나의 input(dialog)를 넣었을 때 inference 결과