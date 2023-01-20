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

fake=pd.read_csv('https://raw.githubusercontent.com/Vishal-Aggarwal0305/fake-real-news-dataset/main/data/Fake.csv')
fake.head()
fake.columns
fake['subject'].value_counts()
plt.figure(figsize=(10,6))
sns.countplot(x='subject',data=fake)