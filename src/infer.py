# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:01:47 2019

@author: WT
"""
import os
import pickle
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchtext
from models import Transformer, create_masks, create_trg_mask
from train import load_state
from process_data import tokener

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

if __name__ == "__main__":
    ### Load model and vocab
    FR = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>",\
                              batch_first=True)
    EN = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, batch_first=True)
    train = torchtext.data.TabularDataset(os.path.join("./data/", "df.csv"), format="csv", \
                                             fields=[("EN", EN), ("FR", FR)])
    FR.build_vocab(train)
    EN.build_vocab(train)
    src_vocab = len(EN.vocab)
    trg_vocab = len(FR.vocab)
    tokenizer_en = tokener("en")
    
    batch_size = 32
    model_no = 0
    cuda = torch.cuda.is_available()
    net = Transformer(src_vocab=src_vocab, trg_vocab=trg_vocab, d_model=512, num=6, n_heads=8)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    if cuda:
        net.cuda()
    start_epoch, acc = load_state(net, optimizer, model_no=0, load_best=False)
    net.eval()
    trg_init = FR.vocab.stoi["<sos>"]
    trg_init = Variable(torch.LongTensor([trg_init])).unsqueeze(0)
    
    while True:
    ### process user input sentence
        sent = input("Enter English sentence:\n")
        sent = tokenizer_en.tokenize(sent).split()
        sent = [EN.vocab.stoi[tok] for tok in sent]
        sent = Variable(torch.LongTensor(sent)).unsqueeze(0)
        
        trg = trg_init
        src_mask, _ = create_masks(sent, trg_init)
        if cuda:
            sent = sent.cuda(); src_mask = src_mask.cuda()
            trg = trg.cuda()
        e_out = net.encoder(sent, src_mask) # encoder output for english sentence
        translated_word = []; translated_word_idxs = []
        for i in range(2, 128):
            trg_mask = create_trg_mask(trg, cuda=cuda)
            if cuda:
                trg = trg.cuda(); trg_mask = trg_mask.cuda()
            outputs = net.fc1(net.decoder(trg, e_out, src_mask, trg_mask))
            out_idxs = torch.softmax(outputs, dim=2).max(2)[1]
            trg = torch.cat((trg, out_idxs[:,-1:]), dim=1)
            if cuda:
                out_idxs = out_idxs.cpu().numpy()
            else:
                out_idxs = out_idxs.numpy()
            translated_word_idxs.append(out_idxs.tolist()[0][-1])
            if translated_word_idxs[-1] == FR.vocab.stoi["<eos>"]:
                break
            translated_word.append(FR.vocab.itos[translated_word_idxs[-1]])
            
        print(" ".join(translated_word))
        print(" ".join(FR.vocab.itos[i] for i in out_idxs[0][:-1]))