# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:39:44 2017

@author: e3acadgrp
"""

#Word dictionary
import os,sys,json, re, string
from time import time
import pandas as pd   
import pickle as pkl
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def my_tokenizer(s):
    tokens = []
    for l in s.split('\n'):
        l = l.strip()
        l = l.lower()
        tokens.extend(l.split(' '))
    tokens = [re.sub('[^a-zA-Z]',',', t) for t in tokens]

    wn = WordNetLemmatizer()
    tokens = [wn.lemmatize(t) for t in tokens]
    tokens = [re.sub('[\W_]+', '', t) for t in tokens]
    tokens = [w for w in tokens if len(w)<20]
    return tokens

def get_maps (fnames_list):
    vectizer = CountVectorizer('filename',
                               stop_words='english',
                               max_df= 0.99,
                               min_df = 0.0011,
                               tokenizer=my_tokenizer,decode_error='ignore')
    Vec = vectizer.fit(fnames_list)
    word2idx = vectizer.vocabulary_
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word, vectizer, Vec

def main ():

    tgt_folder = 'D:/fake/test/'
    dataset = pd.read_csv('D:/nlp_class/nlp/Testfile.csv')
    posneg_files = [os.path.join(tgt_folder, f) for f in dataset.Filename]

    all_files = posneg_files
    word2idx, idx2word, vectizer, Vec = get_maps (all_files)

    with open('word2idx_full.pkl', "wb") as f:
        pkl.dump(word2idx, f, protocol=2)
  
    print( 'word2index map dict is saved in word2idx.json')
    print(  'vocab size: ', len(word2idx.keys()))
    

if __name__ == '__main__':
   main ()

