import json
import logging
import sys
import os

import torch
import torch.nn as nn
from transformers import BertTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

BERT_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
VERSION = 'demo'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        self.bert_path = BERT_MODEL_NAME
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop=nn.Dropout(0.3)
        self.out = nn.Linear(768, 2)
        # self.out = nn.Linear(1024, 2)

    def forward(self, ids, mask):
        outputs = self.bert(input_ids=ids, attention_mask=mask)
        rh=self.bert_drop(outputs.pooler_output)
        return self.out(rh)

# defining model and loading weights to it
def model_fn(model_dir):
    print("Model loading")
    model = BertClassificationModel().to(device)
    with open(os.path.join(model_dir, 'model_epoch_2.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    print("Model loaded")
    model.eval()

    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    print("Preprocessing")
    
    text = json.loads(request_body)["inputs"]
    print(text)
    
    data = tokenizer.encode_plus(
        text,
        max_length=512,
        pad_to_max_length=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_overflowing_tokens=False,
        truncation=True,
        return_tensors='pt'
    )

    return data


# inference
def predict_fn(input_object, model):
    print("Inference")
    
    with torch.no_grad():
        ids = input_object['input_ids'].to(device, dtype=torch.long)
        mask = input_object['attention_mask'].to(device, dtype=torch.long)

        otps = model(ids=ids, mask=mask)
        pred_proba = torch.softmax(otps, dim=1)

    print(pred_proba)
    return pred_proba


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    print("Output")

    res = {
        "positive": f"{predictions[0][1]:0.4f}",
        "negative": f"{predictions[0][0]:0.4f}",
        "version": f"{VERSION}"
    }
    
    print(json.dumps(res))
    return json.dumps(res)
