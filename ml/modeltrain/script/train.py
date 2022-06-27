import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import transformers
from torch.utils.data import DataLoader, RandomSampler, Dataset
from transformers import AdamW, BertTokenizer
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.FATAL)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 514  # this is the max length of the sentence
BERT_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

logger.info("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


class BertDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_len=None
    ):
        self.tokenizer = tokenizer
        self.data_raw = data
        self.max_len = max_len
        self.data, self.label = self.process_data()

    def process_data(self):
        df = self.data_raw.copy()
        train = df.copy()

        train = train.reindex(np.random.permutation(train.index))
        return train['comments'].values, train['labels'].values

    def __getitem__(self, idx):

        text = str(self.data[idx])
        target = int(self.label[idx])

        data = self.tokenizer.encode_plus(
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

        return({
            'ids': data['input_ids'].long(),
            'mask': data['attention_mask'].int(),
            'targets': torch.tensor([target], dtype=torch.int),
            'texto': text.strip(),
        })

    def __len__(self):
        return self.label.shape[0]


class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        self.bert_path = BERT_MODEL_NAME
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 2)

    def forward(self, ids, mask):
        outputs = self.bert(input_ids=ids, attention_mask=mask)
        rh = self.bert_drop(outputs.pooler_output)
        return self.out(rh)


def my_collate(batches):
    return [{key: value for key, value in batch.items()} for batch in batches]


def loss_fun(outputs, targets):
    loss = nn.CrossEntropyLoss()
    return loss(outputs, targets)


def _get_train_data_loader(batch_size, training_dir, is_distributed):
    logger.info("Get train data loader")

    dataset = pd.read_csv(os.path.join(training_dir, "feedbacks_train.csv"))

    train_data = BertDataset(
        data=dataset,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
    else:
        train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=my_collate
    )

    return train_dataloader


def _get_test_data_loader(batch_size, training_dir):
    logger.info("Get test data loader")

    dataset = pd.read_csv(os.path.join(training_dir, "feedbacks_test.csv"))

    test_data = BertDataset(
        data=dataset,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(
        test_data,
        sampler=test_sampler,
        batch_size=batch_size,
        collate_fn=my_collate
    )

    return test_dataloader


def train_loop(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses = []

    for batch_idx, batch in enumerate(data_loader):
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        targets = [data["targets"] for data in batch]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        targets = torch.cat(targets)

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)
        loss = loss_fun(outputs, targets)
        loss.backward()
        model.float()
        optimizer.step()

        if scheduler:
            scheduler.step()
        losses.append(loss.item())

        if batch_idx % 100 == 0:
            print(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} seconds ___")
            t0 = time.time()
    return losses


def eval_loop(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []

    for batch_idx, batch in enumerate(data_loader):
        ids = [data["ids"] for data in batch]
        mask = [data["mask"] for data in batch]
        targets = [data["targets"] for data in batch]

        ids = torch.cat(ids)
        mask = torch.cat(mask)
        targets = torch.cat(targets)

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(ids=ids, mask=mask)
            loss = loss_fun(outputs, targets)
            losses.append(loss.item())

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(
            outputs, dim=1).cpu().detach().numpy())

    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses


def evaluate(target, predicted):

    true_label_mask = [1 if (np.argmax(x)-target[i]) ==
                       0 else 0 for i, x in enumerate(predicted)]
    nb_prediction = len(true_label_mask)
    true_prediction = sum(true_label_mask)
    false_prediction = nb_prediction-true_prediction
    accuracy = true_prediction/nb_prediction

    roc = roc_auc_score(target, predicted[:, 1])

    return{
        "accuracy": accuracy,
        "nb examples": len(target),
        "true_prediction": true_prediction,
        "false_prediction": false_prediction,
        "roc_auc": roc
    }


def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - %s", is_distributed)
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - %d", args.num_gpus)
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend,
                                rank=host_rank, world_size=world_size)
        print(
            "Initialized the distributed environment: '%s' backend on %d nodes. "
            "Current host rank is %d. Number of gpus: %d",
            args.backend, dist.get_world_size(),
            dist.get_rank(), args.num_gpus
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_data_loader = _get_train_data_loader(
        args.batch_size, args.data_dir, is_distributed)
    test_data_loader = _get_test_data_loader(args.test_batch_size, args.test)

    logger.info("Starting BertClassificationModel\n")
    model = BertClassificationModel().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )

    val_losses = []
    batches_losses = []
    val_acc = []
    batches_probs = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(
            f"\n=============== EPOCH {epoch} / {args.epochs} ===============\n")
        batches_losses_tmp = train_loop(
            train_data_loader, model, optimizer, device)
        epoch_loss = np.mean(batches_losses_tmp)
        print(
            f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
        t1 = time.time()
        output, target, val_losses_tmp = eval_loop(
            test_data_loader, model, device)
        print(
            f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
        tmp_evaluate = evaluate(target.reshape(-1), output)
        print(f"=====>\t{tmp_evaluate}")
        val_acc.append(tmp_evaluate['accuracy'])
        val_losses.append(np.mean(val_losses_tmp))
        batches_losses.append(np.mean(batches_losses_tmp))
        batches_probs.append(output)

    print("Saving model...")
    torch.save(
        model,
        f"{args.model_dir}/model_bert_{datetime.now().strftime('%H-%M-%S')}.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--num_labels", type=int, default=2, metavar="N", help="number of labels to classify"
    )

    parser.add_argument(
        "--batch-size", type=int, default=4, metavar="N", help="input batch size for training (default: 4)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=4, metavar="N", help="input batch size for testing (default: 4)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N",
                        help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        metavar="LR", help="learning rate (default: 0.00002)")
    parser.add_argument("--momentum", type=float, default=0.5,
                        metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        metavar="S", help="random seed (default: 42)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list,
                        default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str,
                        default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str,
                        default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str,
                        default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str,
                        default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int,
                        default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
