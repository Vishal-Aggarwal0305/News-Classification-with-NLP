# -*- coding: utf-8 -*-
"""Classification of news.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JInL_zeXDx7RtJuONCkYfGiejA4gG4oV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Embedding,LSTM,Conv1D,MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score



"""
***Exporing fake news***"""

fake=pd.read_csv('https://raw.githubusercontent.com/Vishal-Aggarwal0305/fake-real-news-dataset/main/data/Fake.csv')
fake.head()

fake.columns

fake['subject'].value_counts()

plt.figure(figsize=(10,6))
sns.countplot(x='subject',data=fake)



"""## wordcloud """

text=' '.join(fake['text'].tolist())

wordcloud=WordCloud(width=1920,height=1080).generate(text)
fig=plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()



"""## Explore Real news

"""

real=pd.read_csv('https://raw.githubusercontent.com/Vishal-Aggarwal0305/fake-real-news-dataset/main/data/True.csv')

text=' '.join(real['text'].tolist())

wordcloud=WordCloud(width=1920,height=1080).generate(text)
fig=plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

real.sample(5)

unknown_publishers=[]
for index,row in enumerate(real.text.values):
  try:
    record=row.split('-',maxsplit=1)
    record[1]
    assert(len(record[0])<120)
  except:
    unknown_publishers.append(index)

len(unknown_publishers)

real.iloc[unknown_publishers].text

real.iloc[8970]

real=real.drop(8970,axis=0)

publisher=[]
 tmp_text=[]
 for index,row in enumerate(real.text.values):
   if index in unknown_publishers:
     tmp_text.append(row)
     publisher.append('Unknown')
   else:
     record=row.split('-',maxsplit=1)
     publisher.append(record[0].strip())
     tmp_text.append(record[1].strip())

real['publisher']=publisher
real['text']=tmp_text

real.head()

real.shape

empty_fake_index=[index for index,text in enumerate(fake.text.tolist()) if str(text).strip()==""]

fake.iloc[empty_fake_index]

real['text']=real['title']+ " " + real['text']
fake['text']=fake['title']+ " " + fake['text']

real['text']=real['text'].apply(lambda x: str(x).lower())
fake['text']=fake['text'].apply(lambda x: str(x).lower())





"""## Preprocessing Text"""

real['class']=1
fake['class']=0

real=real[['text','class']]

fake=fake[['text','class']]

data=real.append(fake,ignore_index=True)

data.sample(5)

"""######https://github.com/laxmimerit/preprocess_kgptalkie/tree/master/preprocess_kgptalkie"""

!pip install spacy==2.2.3
!python -m spacy download en_core_web_sm
!pip install beautifulsoup4==4.9.1
!pip install textblob==0.15.3
!pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall

import preprocess_kgptalkie as ps

data['text']=data['text'].apply(lambda x: ps.remove_special_chars(x) )

ps.remove_special_chars('this ,. @## is gre#t')



"""## Vectorisation

"""

import gensim

y=data['class'].values

X=[d.split() for d in data['text'].tolist()]

type(X[0])

print(X[0])

DIM=100
w2v_model=gensim.models.Word2Vec(sentences=X,size=DIM,window=5,min_count=1)

len(w2v_model.wv.vocab)

w2v_model.wv.most_similar('friend')

tokenizer=Tokenizer()
tokenizer.fit_on_texts(X)

X=tokenizer.texts_to_sequences(X)

#tokenizer.word_index



plt.hist([len(x) for x in X],bins=700)
plt.show()

nos=np.array([len(x) for x in X])
len(nos[nos>1000])

maxlen=1000
 X=pad_sequences(X,maxlen=maxlen)

len(X[101])

vocab_size=len(tokenizer.word_index)+1
vocab=tokenizer.word_index

def get_weight_matrix(model):
  weight_matrix=np.zeros((vocab_size,DIM))

  for word,i in vocab.items():
    weight_matrix[i]=model.wv[word]

  return weight_matrix

embedding_vectors=get_weight_matrix(w2v_model)

embedding_vectors.shape

model = Sequential()
model.add(Embedding(vocab_size,output_dim=DIM,weights=[embedding_vectors],input_length=maxlen,trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model.summary()



X_train,X_test,y_train,y_test=train_test_split(X,y)

model.fit(X_train,y_train,validation_split=0.3,epochs=6)

y_pred=(model.predict(X_test)>0.5).astype(int)

accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))



x=['This is a news']
x=tokenizer.texts_to_sequences(x)
x=pad_sequences(x,maxlen=maxlen)
a=(model.predict(x)>=0.5).astype(int)
if(a==0):
  print("News is Fake")
else:
  print("News is genuine")

