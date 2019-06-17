# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:23:19 2019

@author: WT
"""

import pandas as pd
import os
import dill
import re
import spacy
import torchtext
from torchtext.data import BucketIterator

class tokener(object):
    def __init__(self, lang):
        d = {"en":"en_core_web_sm", "fr":"fr_core_news_sm"}
        self.ob = spacy.load(d[lang])
    
    def tokenize(self, sent):
        sent = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sent))
        sent = re.sub(r"\!+", "!", sent)
        sent = re.sub(r"\,+", ",", sent)
        sent = re.sub(r"\?+", "?", sent)
        sent = re.sub(r"[ ]+", " ", sent)
        sent = sent.lower()
        sent = [token.text for token in self.ob.tokenizer(sent) if token.text != " "]
        sent = " ".join(sent)
        return sent

def dum_tokenizer(sent):
    return sent.split()

    
if __name__=="__main__":
    df = pd.read_csv(os.path.join("./data/", "english.txt"), names=["English"])
    df["French"] = pd.read_csv(os.path.join("./data/", "french.txt"), names=["French"])["French"]    
    tokenizer_fr = tokener("fr")
    tokenizer_en = tokener("en")
    df["English"] = df["English"].apply(lambda x: tokenizer_en.tokenize(x))
    df["French"] = df["French"].apply(lambda x: tokenizer_fr.tokenize(x))
    df.to_csv(os.path.join("./data/", "df.csv"), index=False)
    
    FR = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>",\
                              batch_first=True)
    EN = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, batch_first=True)
    train = torchtext.data.TabularDataset(os.path.join("./data/", "df.csv"), format="csv", \
                                             fields=[("EN", EN), ("FR", FR)])
    FR.build_vocab(train)
    EN.build_vocab(train)
    train_iter = BucketIterator(train, batch_size=25, repeat=False, sort_key=lambda x: (len(x["EN"]), len(x["FR"])),\
                                shuffle=True, train=True)
    