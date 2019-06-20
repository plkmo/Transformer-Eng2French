# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:41:08 2019

@author: WT
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data import BucketIterator
from models import Transformer, create_masks
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
    
def dum_tokenizer(sent):
    return sent.split()

### Loads model and optimizer states
def load(net, optimizer, model_no=0, load_best=True):
    base_path = "./data/"
    if load_best == False:
        checkpoint = torch.load(os.path.join(base_path,"test_checkpoint_%d.pth.tar" % model_no))
    else:
        checkpoint = torch.load(os.path.join(base_path,"test_model_best_%d.pth.tar" % model_no))
    start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_acc']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, best_pred

def evaluate(output, labels_e):
    ### ignore index 1 (padding) when calculating accuracy
    idxs = (labels_e != 1).nonzero().squeeze()
    labels = torch.softmax(output, dim=1).max(1)[1]
    return sum(labels_e[idxs] == labels[idxs]).item()/len(idxs)

def evaluate_results(net, data_loader, cuda):
    net.eval(); acc = 0
    print("Evaluating...")
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        trg_input = data.FR[:,:-1]
        labels = data.FR[:,1:].contiguous().view(-1)
        src_mask, trg_mask = create_masks(data.EN, trg_input)
        if cuda:
            data.EN = data.EN.cuda(); trg_input = trg_input.cuda(); labels = labels.cuda()
            src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
        outputs = net(data.EN, trg_input, src_mask, trg_mask)
        outputs = outputs.view(-1, outputs.size(-1))
        acc += evaluate(outputs, labels)
    return acc/(i + 1)

if __name__=="__main__":
    FR = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>",\
                              batch_first=True)
    EN = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, batch_first=True)
    train = torchtext.data.TabularDataset(os.path.join("./data/", "df.csv"), format="csv", \
                                             fields=[("EN", EN), ("FR", FR)])
    FR.build_vocab(train)
    EN.build_vocab(train)
    src_vocab = len(EN.vocab)
    trg_vocab = len(FR.vocab)
    
    batch_size = 32
    model_no = 0
    cuda = torch.cuda.is_available()
    net = Transformer(src_vocab=src_vocab, trg_vocab=trg_vocab, d_model=512, num=6, n_heads=8)
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = optim.Adam(net.parameters(), lr=0.00027, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,100,200], gamma=0.9)
    if cuda:
        net.cuda()
        
    try:
        start_epoch, acc = load(net, optimizer, model_no, load_best=False)
    except:
        start_epoch = 0; acc = 0
    stop_epoch = 200; end_epoch = 200
    
    try:
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
    except:
        losses_per_epoch = []; accuracy_per_epoch = []
        
    train_iter = BucketIterator(train, batch_size=batch_size, repeat=False, sort_key=lambda x: (len(x["EN"]), len(x["FR"])),\
                                shuffle=True, train=True)
    
    for e in range(start_epoch, end_epoch):
        scheduler.step()
        net.train()
        losses_per_batch = []; total_loss = 0.0
        for i, data in enumerate(train_iter):
            #data.EN = data.EN.transpose(0,1)
            #data.FR = data.FR.transpose(0,1)
            trg_input = data.FR[:,:-1]
            labels = data.FR[:,1:].contiguous().view(-1)
            src_mask, trg_mask = create_masks(data.EN, trg_input)
            if cuda:
                data.EN = data.EN.cuda(); trg_input = trg_input.cuda(); labels = labels.cuda()
                src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
            optimizer.zero_grad()
            outputs = net(data.EN, trg_input, src_mask, trg_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 99: # print every 100 mini-batches of size = batch_size
                losses_per_batch.append(total_loss/100)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.7f' %
                      (e, (i + 1)*batch_size, len(train), total_loss/100))
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(evaluate_results(net, train_iter, cuda))
        print("Losses at Epoch %d: %.7f" % (e, losses_per_epoch[-1]))
        print("Accuracy at Epoch %d: %.7f" % (e, accuracy_per_epoch[-1]))
        if accuracy_per_epoch[-1] > acc:
            acc = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': acc,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/" ,\
                    "test_model_best_%d.pth.tar" % model_no))
        if (e % 2) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % model_no, accuracy_per_epoch)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/",\
                    "test_checkpoint_%d.pth.tar" % model_no))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(losses_per_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_title("Loss vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/",\
                             "test_loss_vs_epoch_%d.png" % model_no))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.set_title("Accuracy vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/",\
                             "test_Accuracy_vs_epoch_%d.png" % model_no))
    