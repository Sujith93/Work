import os
os.chdir(r"C:/User/soppa/Document/intent_classification")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Dropout,Activation,Flatten,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizer import SGD 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import AUC

from sklearn.metrics import f1_score

df_org = pd.read_excel("data/DFCX_intent_Training_corpus.xlsx")

##excluding 1 occurance intents
df_count = df_org.Intents.value_counts()
# total intents : 868
less_1 = df_count[df_count==1].index.tolist()
# intents equal to 1(only 1 tp exist) : 42
df_less1 = df_org[~df_org['Intents'].isin(less_1)]


## excluding 1 occurance intents and checking head_intent
less_5 = df_count[(df_count<11) & (df_count>1)].index.tolist()
# intents less than 5 : 81
# less_5_head = [ l for l in less_5 if "head_intent" not in l]
# intents less than 5 : 80
# df_less5 = df_less1[~df_less1['Intents'].isin(less_5)]
df_less5 = df_less1[~df_less1['Intents'].isin(less_5)]

## checking the intents after excluding
df_count_after = df_less5.Intents.value_counts()
#868-42-80=746 intents used for training
df = df_less5.copy()
# df.head()

# from nltk.tokenize import word_tokenize
# sent_token = [ len(word_tokenize(phrase) for phrase in df['Training Phrases'])]
# sent_token

# sent_token.index(max(sent_token))
# plt.hist(sent_token,bins=5)
# np.median(sent_token)

X_df = df['Training Phrases'].values
y_df = pd.get_dummies(df['Intents'])
classes = y_df.columns
y_df = y_df.values
X_train,X_val,y_train,y_val = train_test_split(X_df,y_df,test_size=0.10,
												stratify=y_df,
												random_state=1)
# X_val,X_test,y_val,y_test = train_test_split(X_int,y_int,test_size=0.50,
# 												stratify=y_int,
# 												random_state=2)

X_train.shape
y_train.shape

X_val.shape
y_val.shape

labels_no = y_train.shape[1]

vocab_size = 50000
emebdding_dim = 100
max_length = 50
padding_type = 'post'
trunc_type = 'post'
oov_tok = '<OOV'

#text tokenizer
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_text(X_train)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences,maxlen=max_length,
				padding=padding_type,truncating=trunc_type)
val_sequences = tokenizer.texts_to_sequences(X_val)
val_padded = pad_sequences(val_sequences,maxlen=max_length,
				padding=padding_type,truncating=trunc_type)

model_name = 'base_model_with_accuracy_v2_trainvalid_incfilt'

if not os.path.exists("models/"+model_name):
	os.mkdir("models/"+model_name)

def f1_measure(y_true,y_pred):
	true_args = np.argmax(y_true,axis=1)
	true_classes = classes[true_args]
	pred_args = np.argmax(y_pred.numpy(),axis=1)
	test_classes = classes[pred_args]
	return f1_score(true_classes,test_classes,average='macro')

model = Sequential([
	Embedding(vocab_size,emebdding_dim,input_length=max_length),
	Bidirectional(LSTM(128,return_sequences=True)),
	Bidirectional(LSTM(64)),
	# Flatten(),
	Dense(64,activation='relu')
	Dense(200,activation='relu')
	Dense(400,activation='relu')
	Dense(labels_no,activation='softmax')
	])
model.complie(loss='categorical_crossentropy',optimizer='adam',
		metrics=['accuracy',Precision(),Recall(),AUC(),f1_measure],
		run_eagerly=True)

es = EarlyStopping(monitor='val_f1_measure',mode='max',verbose=1,patience=10)
# es = EarlyStopping(monitor='val_accuracy',mode='max',verbose=1,patience=10)
# es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)

model.summary()

num_epochs=100
num_batch=64
history = model.fit(train_padded,y_train,epochs=num_epochs,
					batch_size=num_batch,validation_data=(val_padded,y_val),verbose=1,
					callbacks=[es])

def plot_graphs(history, string,model_name):
	plt.plot(history.history[string])
	plt.plot(history.history['val_'+string])
	plt.xlabel("Epochs")
	plt.ylabel(string)
	plt.legend([string,'val_'+string])
	plt.savefig(f"models/{model_name}/"+string+"_"+model_name+"_"+".png",format='png')
	plt.show()

plot_graphs(history,"accuracy",model_name)
plot_graphs(history,"loss",model_name)
plot_graphs(history,"precision",model_name)
plot_graphs(history,"recall",model_name)
plot_graphs(history,"auc",model_name)
plot_graphs(history,"f1_measure",model_name)

# save the model
model.save(f"models/{model_name}/intent_classification_{model_name}.h5")

# model evaluation
#model.evaluate(train_padded,y_train)
#model.evaluate(val_padded,y_val)
#model.evaluate(test_padded,y_test)

with open(f"models/{model_name}/evaluation.txt","a+") as f:
	i="train"
	loss,accuracy,precision,recall,auc,f1_measure = model.evaluate(train_padded,y_train)
	f.write(f"{i}_accuracy = " + str(accuracy))
	f.write(f"{i}_precision = " + str(precision))
	f.write(f"{i}_recall = " + str(recall))
	f.write(f"{i}_f1_measure = " + str(f1_measure))
	f.write(f"{i}_auc = " + str(auc))
	f.write(f"{i}_loss = " + str(loss))
	f.write("\n")
	i="val"
	loss,accuracy,precision,recall,auc,f1_measure = model.evaluate(val_padded,y_val)
	f.write(f"{i}_accuracy = " + str(accuracy))
	f.write(f"{i}_precision = " + str(precision))
	f.write(f"{i}_recall = " + str(recall))
	f.write(f"{i}_f1_measure = " + str(f1_measure))
	f.write(f"{i}_auc = " + str(auc))
	f.write(f"{i}_loss = " + str(loss))
	f.write("\n")


import pickle
dependency = [tokenizer,pad_sequences,max_length,padding_type,trunc_type,classes]

with open(f"models/{model_name}/dependency.pkl",'wb') as f:
	pickle.dump(dependency,f)


############### Prediction ################
import pickle
from tensorflow.keras.models import load_model


model_name = 'base_model_with_accuracy_v2_trainvalid'

dep_path = f"models/{model_name}/dependency.pkl"
model_path = f"models/{model_name}/intent_classification_{model_name}.h5"

with open(dep_path,'rb') as f:
	dependency = pickle.load(f)


tokenizer,pad_sequences,max_length,padding_type,trunc_type,classes = dependency

def f1_measure(y_true,y_pred):
	true_args = np.argmax(y_true,axis=1)
	true_classes = classes[true_args]
	pred_args = np.argmax(y_pred.numpy(),axis=1)
	test_classes = classes[pred_args]
	return f1_score(true_classes,test_classes,average='macro')

custom_objects = {
	'f1_measure' : f1_measure
}

saved_model = load_model(model_path,custom_objects=custom_objects)

def predict_score(sentence):
	tokens = tokenizer.texts_to_sequences([sentence])
	tokens = pad_sequences(tokens,maxlen = max_length,
				padding=padding_type,truncating = trunc_type)
	prediction = saved_model.predict(tokens)
	pred_arr = prediction.ravel()
	pred_args = pred_arr.argsort()[::-1][:5]

	prob = pred_arr[pred_args]
	prob_li = [round(pro,4) for pro in prob]
	pred_class = classes[pred_args.tolist()].tolist()
	return str(dict(zip(pred_class,prob_li)))

# sentence = " I want to delete my jackpack"

predict_score(sentence)


