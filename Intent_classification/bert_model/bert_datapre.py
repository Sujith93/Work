
import os
os.chdir(r"C:/User/soppa/Document/intent_classification")

import pandas as pd

df= pd.read_csv("data/train_trail.csv")
df.columns

df = df.drop_duplicates(keep='first')
df['sentence'] = df['sentence'].fillna("Fillna")

seq_len = 512
num_samples = len(df)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

tokens = tokenizer(df['sentence'].tolist(),max_length=seq_len,
			trancation = True,padding='max_length',
			add_special_tokens = True,
			return_tensors='np')

tokens.keys()
tokens['input_ids']
tokens['attention_mask']

import numpy as np
with open('sent_inputids.npy','wb') as f:
	np.save(f,tokens['input_ids'])
with open('sent_attmask.npy','wb') as f:
	np.save(f,tokens['attention_mask'])


labels = df[['BookRestaurent','GetWeather','PlayMusic','RateBook']].to_numpy() 
# Multi class labels
# for each label we have a column
with open('sent_labels.npy','wb') as f:
	np.save(f,labels)

