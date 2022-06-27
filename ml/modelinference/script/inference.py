import json
import logging
import sys
import os

import torch
from transformers import BertTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

BERT_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
VERSION = 'demo'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


# defining model and loading weights to it
def model_fn(model_dir):
    print("Model loading")
    checkpoints = []
    
    for file in os.listdir(model_dir):
        if file.endswith(".pt"):
            print(file)
            checkpoints.append(file)
    model = torch.load(f"{model_dir}/{checkpoints[0]}")
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
