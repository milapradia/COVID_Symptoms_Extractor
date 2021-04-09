import pandas as pd
import numpy as np

import os

import nltk
nltk.download('wordnet')


from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 

from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM

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
import time
from collections import deque

stop_words = set(stopwords.words('english')) 



from utils import *
# from plotting import *

import marshal

from ADRModel_v5 import *

class config:
  MODEL = 'digitalepidemiologylab/covid-twitter-bert-v2'
  EMBEDDING_SIZE = 1024


model = BertForSequenceClassification.from_pretrained(config.MODEL, output_hidden_states= True)

tokenizer = BertTokenizer.from_pretrained(config.MODEL)

import sys

batchNum = str(sys.argv[1])
covidData = 'data/'
df = pd.read_csv(os.path.join(covidData, 'batch_' + batchNum + '.csv'), nrows = 10000)

df = df[['message_id','user_id','message']]



### The following code will be used if you want to instantiate a new starting seed word with its corresponding embedding
### The getSymptomEmbedding is a function that finds instances matching the seedword and the user can then manually select
### appropriate instances to build the seed word embedding
# embList, msgList = getSymptomEmbedding(model,tokenizer, df, 'cough',0)
# indexList = [0,2,3,12,13,25]
# meanEmb = np.array(embList)[indexList,:]
# meanEmb = np.mean(meanEmb,0)



file = 'cough_Emb.npy'
embList = np.load(os.path.join('EmbDir/',file))
meanEmb = np.mean(embList,0)


### Setting up the graph with the seed word and its corresponding embedding
graph = Graph()
graph.addNode('cough', 0, 0)
graph['cough'].vector = meanEmb

### To speed up the search process, embeddings of messages are combined into chunks of 10000 messages. This helps as the key 
### bottleneck was the read time for reading embedding files of messages. If your dataset is small enough to be saved in-memory, ### definitely go for that. outputFolder contains the messages individually while combinedOutputFolder contains them in chunks of ### 1000
outputFolder = 'DataStore/'
combinedOutputFolder = 'DataStore_10000/'
modelFolder = './ModelFolder/Covid_cough/'

q = deque()
q.append(('cough',0))

ADR = ADRModel(df, model, tokenizer, graph, outputFolder, combinedOutputFolder, modelOutputFolder = modelFolder, 
               queue = q,  useMasterEmb=True, masterContrib=0.4, numThreshold=1000000, saveEveryDepth = True)


startTime = time.time()

print("Training started")

ADR.trainModel(maxDepth=3,topk=10) 
ADR.df.to_csv('Predicted/pred_batch_' + str(batchNum) + '.tsv', sep='\t'))
print("Training finished")
print(time.time() - startTime)


# ADR.model = None
# ADR.tokenizer = None
# ADR.df = None
      
# pickle.dump(ADR, open('Covid_fever_3_5_0.4.pkl','wb'))

print("Model saved")
