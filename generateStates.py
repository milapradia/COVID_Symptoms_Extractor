################################################
# This code processes your messages and saves the embeddings of tokens in a message as separate files. These embeddings of tokens # is later used during the iterative search process to find tokens similar to the context embedding. This code has 2 parts
# 1. Generates embeddings of each message individually by running it through the BERT model. This results in a single saved file 
# for each message
# 2. Collates individual files (typically 10000) from the first step into bigger files. This is done to speed up the iterative 
# search process
################################################


import pandas as pd
import numpy as np

import os

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 

from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM, BertModel

from simpletransformers.language_modeling import LanguageModelingModel

from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

from tqdm import tqdm
import torch

import networkx as nx

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from functools import partial

import pickle

from collections import deque

stop_words = set(stopwords.words('english')) 

import time

from utils import *
from plotting import *

import marshal
import sys

class config:
  MODEL = 'digitalepidemiologylab/covid-twitter-bert-v2'
  EMBEDDING_SIZE = 1024
  MAXLENGTH = 80
  '''
  MODEL = 'CovidBertModel/checkpoint-757-epoch-1'
  MODEL = 'bert-base-uncased'
  EMBEDDING_SIZE = 768
  '''

batchNum = int(sys.argv[1])
covidData = 'Batch/'
df = pd.read_csv(os.path.join(covidData, 'batch_' + str(batchNum) + '.tsv'), sep = '\t')
df = df[['message_id','user_id','message']]

model = BertModel.from_pretrained(config.MODEL, output_hidden_states= True)
tokenizer = BertTokenizer.from_pretrained(config.MODEL)

outputFolder = 'DataStore/'
embeddingType = 'last4sum'


###########################
######### PART 1 ##########
###########################


# device = 'cuda:1'

# model = model.to(device)

# startTime = time.time()

# for i in tqdm(range(850000,1000000)):
            
#     if os.path.exists(os.path.join(outputFolder, str(i)+".msh")):
#         continue


#     tokens = tokenizer.encode(df.iloc[i]['message'].lower())
#     decoded = tokenizer.decode(tokens).split(" ")
#     logits, hidden_states = model(torch.Tensor(tokens).to(device).unsqueeze(0).long())

#     hidden_states = torch.stack(hidden_states).squeeze(1).permute(1,0,2)


#     if embeddingType == 'last4sum':
#         embedding = torch.sum(hidden_states[:,9:13,:],1)
#     elif embeddingType =='last4concat':
#         embedding = hidden_states[tokenIndex,9:13,:].reshape(-1)
#     elif embeddingType == 'secondlast':
#         embedding = hidden_states[tokenIndex,-2,:]
#     else:
#         embedding = hidden_states[tokenIndex,-1,:]


#     embedding = embedding.detach().cpu().numpy()

#     marshal.dump(embedding.tolist(), open(os.path.join(outputFolder, str(i)+".msh"), 'wb'))
    

# print(f"Time taken : {time.time() - startTime}")

device = 'cuda:0'

model = model.to(device)

startTime = time.time()

for i in tqdm(range(0, len(df))):
    #break    
    if os.path.exists(os.path.join(outputFolder, str(i)+".msh")):
        continue

    text = df.iloc[i]['message'].lower()
    marked_text = "[CLS] " + text + " [SEP]"


    #tokenized_text = tokenizer.tokenize(marked_text, padding=True, truncation=True)

    #indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    indexed_tokens = tokenizer.encode(marked_text, padding=True, truncation=True, max_length = 80)

    tokenized_text = tokenizer.convert_ids_to_tokens(indexed_tokens)

    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    segments_tensors = torch.tensor([segments_ids]).to(device)

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)

    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs_sum = [[0]]

    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec.detach().cpu().numpy())

    
    '''
    if embeddingType == 'last4sum':
        embedding = torch.sum(hidden_states[:,9:13,:],1)
    elif embeddingType =='last4concat':
        embedding = hidden_states[tokenIndex,9:13,:].reshape(-1)
    elif embeddingType == 'secondlast':
        embedding = hidden_states[tokenIndex,-2,:]
    else:
        embedding = hidden_states[tokenIndex,-1,:]
    '''
    #print(len(token_vecs_sum))
    #print(len(token_vecs_sum[1]))

    #embedding = np.concatenate(token_vecs_sum[1:], axis=0)
    embedding = np.array(token_vecs_sum[1:])
    #print(embedding.shape)
    #break
    marshal.dump(embedding.tolist(), open(os.path.join(outputFolder, str(i)+".msh"), 'wb'))


print(f"Time taken : {time.time() - startTime}")

###########################
######### PART 2 ##########
###########################

def aggFiles(index, numComp, df, tokenizer, inputFolder, outputFolder):
    
    filename = os.path.join(outputFolder, f"{index}.pkl")
    print(filename)
    
    if os.path.exists(filename):
        print("Skipping file since it exists")
        return

    IDList = []
    tokenList = []
    embList = []

    for i in range(index*numComp, (index+1)*numComp):
        text = df.iloc[i]['message'].lower()

        marked_text = "[CLS] " + text + " [SEP]"

        indexed_tokens = tokenizer.encode(marked_text, padding=True, truncation=True, max_length = config.MAXLENGTH)

        tokens = tokenizer.convert_ids_to_tokens(indexed_tokens)

        #tokens = tokenizer.tokenize(marked_text, padding=True, truncation=True, max_length = 50)
        
        emb_file_name = os.path.join(inputFolder, f"{i}.msh")
        #print(emb_file_name)
        
        try:

            emb_file = open(emb_file_name, 'rb')
            #print(emb_file)
            emb_array = marshal.load(emb_file)
            #print(emb_array)
            emb = np.array(emb_array)
        except EOFError as e:
            print(emb_file_name)
            emb = np.zeros((len(tokens), model.EMBEDDING_SIZE))
            #continue
        IDList += [i]*len(tokens)
        tokenList += tokens

        embList.append(emb)

    IDList = np.array(IDList)
    tokenList = np.array(tokenList)
    embList = np.concatenate(embList,axis=0)

    subDict = {'id':IDList, 'token':tokenList,'emb':embList}
    print(filename)    
    pickle.dump(subDict, open(filename,'wb'))
    
    
    
numComp = 1000
inputFolder = 'DataStore/'
outputFolder = 'DataStore_10000/'
numBatch = df.shape[0] // numComp

for i in tqdm(range(0, numBatch)):
    
    aggFiles(i, numComp, df, tokenizer, inputFolder, outputFolder)
        
 
################################################################################# 
    
    
    

    
    
    
