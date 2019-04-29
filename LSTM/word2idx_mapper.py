import os,sys,json, re, string
from time import time
import pandas as pd   
import pickle as pkl
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
#from joblib import Parallel,delayed
#from random import shuffle
import numpy as np

def my_tokenizer(s):
    tokens = []
    for l in s.split('\n'):
        l = l.strip()
        l = l.lower()
        tokens.extend(l.split(' '))
    tokens = [re.sub('[^a-zA-Z]',',', t) for t in tokens    #ps = PorterStemmer()
    #tokens = [ps.stem(t) for t in tokens]
    #wn = WordNetLemmatizer()
    #tokens = [wn.lemmatize(t) for t in tokens]
    #tokens = [re.sub('[\W_]+', '', t) for t in tokens]
    #tokens = [w for w in tokens if len(w)<20]
    #print(tokens)
    return tokens

def get_maps (fnames_list):
    vectizer = CountVectorizer('filename',
                               stop_words='english',
                               max_df= 0.99,
                               min_df = 0.0011,
                               tokenizer=my_tokenizer,decode_error='ignore')
    d = vectizer.fit(fnames_list)
    word2idx = vectizer.vocabulary_
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word, vectizer, d

def transform_textfile_to_index (fname, word2idx):
    try:
        tokens = my_tokenizer(open(fname).read())
    except: UnicodeDecodeError
    pass
    
    op_idx_tokens = []
    for t in tokens:
        try:
            idx = word2idx[t]
            op_idx_tokens.append(idx)
        except:
            # op_idx_tokens.append(-1)
            pass
    return op_idx_tokens 


def main ():

    tgt_folder = 'C://Users//Ray//Downloads//temp//'
    all_files = [os.path.join(tgt_folder,f) for f in os.listdir(tgt_folder)]
    #dataset = pd.read_csv('../data/file_emotion_sentiment.csv')
    #posneg_files = [os.path.join(tgt_folder, str(file) + ".txt") for file in dataset.Filename]
    #tgt_folder = 'C://Users//Ray//Downloads//temp//'
    #dataset = pd.read_csv('D:/nlp_class/nlp/Data4530.csv')
    #posneg_files = [os.path.join(tgt_folder, f) for f in dataset.Filename]#gdater))
    #posneg_labels = list(dataset.Sentiment)


    tgt_folder = 'C:/Users/Ray/Downloads/superclean/superclean/mann-k/sent/'
    all_files = [os.path.join(tgt_folder, f) for f in os.listdir(tgt_folder)]
  
    # neutral_files = list(set(all_files) - set(posneg_files))
    # all_files = posneg_files + neutral_files
    # all_labels = [0 if i==-1 else i for i in posneg_labels] + [2 for _ in neutral_files]
    #
    # t = zip (all_files,all_labels)
    # shuffle (t)
    # all_files, all_labels = zip(*t)

    posneg_files  = all_files
    
    word2idx, idx2word, vectizer, d = get_maps (all_files)

  #  with open ('word2idx.json','w') as fh:
  #      json.dump(word2idx,fh,indent=4)
 
    with open('word2idx_4530.pkl', "wb") as f:
        pkl.dump(word2idx, f, protocol=2)
  
    print( 'word2index map dict is saved in word2idx.json')
    print(  'vocab size: ', len(word2idx.keys()))

    op_tuples = []
    for i,f in enumerate(all_files):
        idxes = transform_textfile_to_index (f, word2idx)
        op_tuples.append(idxes)

   # with open ('index_label_tuples.json','w') as fh:
    #    json.dump(op_tuples,fh,indent=4)
        
    with open('mann.pkl', "wb") as f:
        pkl.dump(op_tuples, f, protocol=2)
  

    len_of_lists = np.array([len(l) for l,label in op_tuples])
    print( 'mean of len of lists: ', len_of_lists.mean())
    print( 'std of len of lists: ', len_of_lists.std())
    print( 'max limit for doc length (mean + 3* std): ',len_of_lists.mean() + \
                                                       (3*(len_of_lists.std())))

if __name__ == '__main__':
   main ()


#reduce the list length
#word2vec
#