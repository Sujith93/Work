import os
os.chdir(r"C:/User/soppa/Document/intent_classification")

#------ Build dataset

import numpy as np
with open('sent_inputids.npy','rb') as f:
	Xids = np.load(f,allow_pickle=True)
with open('sent_attmask.npy','rb') as f:
	Xmask = np.load(f,allow_pickle=True)
with open('sent_labels.npy','rb') as f:
	labels = np.load(f,allow_pickle=True)	

Xids.shape

import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices((Xids,Xmask,labels))
dataset.take(1)

def map_func(input_ids,masks,labels):
	return {'input_ids':input_ids,
			'attention_mask':masks},labels

dataset = dataset.map(map_func)
dataset.take(1)


#-------- dataset shuffle, Batch, split, save

batch_size = 16
dataset = dataset.shuffle(1000).batch(batch_size,drop_remainder=True)
dataset.take(1)

split = 0.9
size = int((Xids.shape[0]/batch_size*split))
train_ds = dataset.take(size)
val_ds = dataset.skip(size)

tf.dataset.experimental.save(train_ds,'train')
tf.dataset.experimental.save(val_ds,'val')

train_ds.elements_spec
val_ds.elements_spec

train_ds.elements_spec == val_ds.elements_spec

ds = tf.data.experimental.load('train',elements_spec=train_ds.elements_spec)
ds.take(1)

#------- Build and save
from transformers import TFAutoModel
bert = TFAutoModel.from_pretrained("bert-base-cased")
bert.summary()

import tensorflow as tf
# two inputs
input_ids = tf.keras.layers.Input(shape(512,),
							name='input_ids',dtype='int32')
mask = tf.keras.layers.Input(shape(512,),
							name='attention_mask',dtype='int32')
# Transformer
embeddings = bert.bert(input_ids,attention_mask=mask)[1]
# classifier head
x = tf.keras.layers.Dense(1024,activation='relu')(embeddings)
y = tf.keras.layers.Dense(4,activation='softmax',name='outputs')(x)

model = tf.keras.Model(inputs=[input_ids,mask],outputs=y)
model.summary()

model.layers[2].trainable = False
model.summary()

optimizer = tf.keras.optimizers.Adam(lr=5e-5,decay=1e-6)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.complie(optimizer=optimizer,loss=loss,metrics=[acc])

history = model.fit(train_ds,
					validation_data=val_ds,
					epochs=10)
model.save('sent_class_model')


#------- Loading and Prediction

import tensorflow as tf
model = tf.keras.model.load_model('sent_class_model')

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def prep_data(text):
	tokens = tokenizer(text,max_length=512,truncation=True,
				padding='max_length',add_special_tokens=True,
				return_tensors='tf')

prep_data('hello world')
model.predict(prep_data('hello world'))

probs = model.predict(prep_data('hello world'))[0]
probs

import numpy as np
np.argmax(probs)










