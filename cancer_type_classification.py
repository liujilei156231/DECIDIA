import argparse
parser = argparse.ArgumentParser(description='OPT Pretraining')
parser.add_argument('--input_dir', type=str, help='input directory')
parser.add_argument('--num_hidden_layers', type=int, default=1, help='num_hidden_layers [1]')
parser.add_argument('--train_file', type=str, help='training file')
parser.add_argument('--val_file', type=str, help='validation file')
parser.add_argument('--device', type=str, help='device', default='cuda:1')
parser.add_argument('--num_classes', type=int, help='num_classes [32]', default=32)
parser.add_argument('--diseases', type=str, default=None, help='diseases included, e.g "LUAD,LUSC"')
parser.add_argument('--weight_decay', type=float, help='weight_decay [1e-5]', default=1e-5)
parser.add_argument('--modeling_context', action='store_true', help='whether use OPT to model context dependency')
parser.add_argument("--lr_scheduler_type", type=str,
                    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                    default="constant", help="The scheduler type to use.")
parser.add_argument('--pretrained_weight', type=str, help='pretrained weight')
parser.add_argument('--pretrained_cls_token', type=str, help='pretrained cls token')
parser.add_argument('--epochs', type=int, default=100, help='epochs (default: 100)')
parser.add_argument('--num_sequences', type=int, default=None, help='num of sequences to sample from training set')
parser.add_argument('--max_length', type=int, default=None, help='num of sequences to sample from training set')
parser.add_argument('--num_train_patients', type=int, default=None, help='num of patients data to sample from training set')

args = parser.parse_args()

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import glob
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW, Adam, SGD, Adagrad
from sklearn.utils import resample
from transformers import get_scheduler
import numpy as np
import pandas as pd
import random
import time
from transformers import PreTrainedTokenizerFast, OPTForCausalLM, BertForMaskedLM

from model.model_toad import TOAD_fc_mtl_concat


torch.set_num_threads(4)

device = args.device

random.seed(123)

tokenizer = PreTrainedTokenizerFast.from_pretrained('./model/opt-alphabet')
net = OPTForCausalLM.from_pretrained('./model/clm-2m')

net = net.to(device)
net.eval()

feature_dim = net.config.hidden_size

trn_df = pd.read_csv(f'{args.input_dir}/trn.csv.gz')
reads_per_patient = trn_df.patient.value_counts().unique()
assert len(reads_per_patient) == 1
reads_per_patient = reads_per_patient[0]
if args.num_sequences < reads_per_patient:
    trn_df = pd.concat([df.sample(args.num_sequences, random_state=123) for patient, df in trn_df.groupby('patient')])

num_train_samples = len(trn_df.patient.unique())
if args.num_train_patients is None:
    args.num_train_patients = num_train_samples
if args.num_train_patients < num_train_samples:
    trn_df = trn_df[trn_df.patient.isin(random.sample(trn_df.patient.unique().tolist(), args.num_train_patients))]
    
trn_x = torch.zeros(args.num_train_patients, args.num_sequences, feature_dim)
trn_y = torch.as_tensor([-1] * args.num_train_patients)

test_df = pd.read_csv(f'{args.input_dir}/test.csv.gz')
num_test_samples = len(test_df.patient.unique())
test_x = torch.zeros(num_test_samples, reads_per_patient, feature_dim)
test_y = torch.as_tensor([-1] * num_test_samples)
test_patients = []

val_df = pd.read_csv(f'{args.input_dir}/val.csv.gz')
num_val_samples = len(val_df.patient.unique())
val_x = torch.zeros(num_val_samples, reads_per_patient, feature_dim)
val_y = torch.as_tensor([-1] * num_val_samples)
val_patients = []


pad_token_id = net.config.pad_token_id


for i, (patient, e) in tqdm(enumerate(trn_df.groupby('patient')), total=args.num_train_patients):
    a = [' '.join(list(s)) for s in e.seq]
    inputs = tokenizer(a, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt', return_token_type_ids=False)
    for k, v in inputs.items():inputs[k] = v.to(device)
    with torch.inference_mode():
        out = net.model(**inputs)
    #features = out.last_hidden_state[:,0,:].cpu()
    features = out.last_hidden_state.mean(1).cpu()
    trn_x[i] = features
    trn_y[i] = e.label.tolist()[0]


for i, (patient, e) in tqdm(enumerate(test_df.groupby('patient')), total=num_test_samples):
    a = [' '.join(list(s)) for s in e.seq]
    inputs = tokenizer(a, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt', return_token_type_ids=False)
    for k, v in inputs.items():inputs[k] = v.to(device)
    with torch.inference_mode():
        out = net.model(**inputs)
    #features = out.last_hidden_state[:,0,:].cpu()
    features = out.last_hidden_state.mean(1).cpu()
    test_x[i] = features
    test_y[i] = e.label.tolist()[0]
    test_patients.append(patient)

for i, (patient, e) in tqdm(enumerate(val_df.groupby('patient')), total=num_val_samples):
    a = [' '.join(list(s)) for s in e.seq]
    inputs = tokenizer(a, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt', return_token_type_ids=False)
    for k, v in inputs.items():inputs[k] = v.to(device)
    with torch.inference_mode():
        out = net.model(**inputs)
    #features = out.last_hidden_state[:,0,:].cpu()
    features = out.last_hidden_state.mean(1).cpu()
    val_x[i] = features
    val_y[i] = e.label.tolist()[0]
    val_patients.append(patient)



fout = open(f'{args.input_dir}/log-reads-{args.num_sequences}-patients-trn{args.num_train_patients}-val{num_val_samples}-test{num_test_samples}-tiny.txt', 'w')
print("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\teval_loss\teval_acc", file=fout)

model = TOAD_fc_mtl_concat(input_dim=feature_dim, n_classes=args.num_classes, size_arg='big')
#model = CLAM_SB(input_dim=DIM, n_classes=2)


if args.pretrained_weight:
    state_dict = torch.load(args.pretrained_weight, map_location='cpu')
    if state_dict['classifier.weight'].size(0) != args.num_classes:
        del state_dict['classifier.weight']
        del state_dict['classifier.bias']

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)#, file=fout)

model = model.to(device)

print(model)#, file=fout)


criterion = nn.CrossEntropyLoss()
#opt = Adam(model.parameters(), lr=2e-5, weight_decay=1e-5)
#opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 1e-5,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
opt = AdamW(optimizer_grouped_parameters, lr=2e-5)


num_update_steps_per_epoch = len(trn_df)
max_train_steps = args.epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=opt, num_warmup_steps=num_update_steps_per_epoch*1, num_training_steps=max_train_steps)


best_eval_acc = 0.0
best_eval_loss = 100000.0
best_val_loss = 100000.0
for epoch in range(args.epochs):
    model.train()
    total_loss, total_batch, total_num, correct_k = 0, 0, 0, 0
    idxs = random.sample(range(len(trn_y)), len(trn_y))
    for idx in idxs:
        x = trn_x[idx]
        y = trn_y[idx].unsqueeze(0)
        x = x.to(device)
        y = y.to(device)

        logit = model(x)
        loss = criterion(logit, y) 

        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_scheduler.step()

        total_loss += loss.item()
        total_batch += 1
        total_num += len(y)
        correct_k += logit.argmax(1).eq(y).sum()

    train_acc = correct_k / total_num
    train_loss = total_loss / total_batch

    #######Evalutate on test set ################
    model.eval()
    total_loss, total_batch, total_num, correct_k = 0, 0, 0, 0
    eval_probs = []
    for x, y, pid in zip(test_x, test_y, test_patients):
        y = y.unsqueeze(0).to(device)
        x = x.to(device)

        with torch.inference_mode():
            logit = model(x)
        loss = criterion(logit, y) 

        eval_probs.append(logit.flatten().softmax(0).tolist())

        total_loss += loss.item()
        total_batch += 1
        total_num += len(y)
        correct_k += logit.argmax(1).eq(y).sum()

    eval_acc = correct_k / total_num
    eval_loss = total_loss / total_batch

    #######Evalutate on val set ################
    model.eval()
    total_loss, total_batch, total_num, correct_k = 0, 0, 0, 0
    val_probs = []
    for x, y, pid in zip(val_x, val_y, val_patients):
        y = y.unsqueeze(0).to(device)
        x = x.to(device)

        with torch.inference_mode():
            logit = model(x)
        loss = criterion(logit, y) 

        val_probs.append(logit.flatten().softmax(0).tolist())

        total_loss += loss.item()
        total_batch += 1
        total_num += len(y)
        correct_k += logit.argmax(1).eq(y).sum()

    val_acc = correct_k / total_num
    val_loss = total_loss / total_batch


    #print(f"Epoch: {epoch + 1}; train loss: {train_loss:.5f}, acc: {train_acc:.5f}; eval loss: {eval_loss:.5f}, acc: {eval_acc:.5f}; ", file=fout)
    print(f"{epoch+1}\t{train_loss}\t{train_acc}\t{val_loss}\t{val_acc}\t{eval_loss}\t{eval_acc}", file=fout)
    fout.flush()

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), f'{args.input_dir}/model-reads-{args.num_sequences}-patients-trn{args.num_train_patients}-val{num_val_samples}-test{num_test_samples}-tiny.pt')
        best_val_loss = val_loss

        eval_probs = pd.DataFrame(eval_probs, columns=['p_crc', 'p_hcc', 'p_lung'])
        info = pd.DataFrame({'patient':test_patients, 'label':test_y.tolist()})
        info = pd.concat([info, eval_probs], axis=1)
        info.to_csv(f'{args.input_dir}/test_prediction-reads-{args.num_sequences}-patients-trn{args.num_train_patients}-val{num_val_samples}-test{num_test_samples}-tiny.csv', index=False)

        val_probs = pd.DataFrame(val_probs, columns=['p_crc', 'p_hcc', 'p_lung'])
        info = pd.DataFrame({'patient':val_patients, 'label':val_y.tolist()})
        info = pd.concat([info, val_probs], axis=1)
        info.to_csv(f'{args.input_dir}/val_prediction-reads-{args.num_sequences}-patients-trn{args.num_train_patients}-val{num_val_samples}-test{num_test_samples}-tiny.csv', index=False)

fout.close()



