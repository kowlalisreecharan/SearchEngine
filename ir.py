
# coding: utf-8

# In[1]:

import pandas as pd
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math as mt
import numpy as np
from numpy import linalg as LA
import csv


# In[2]:

def text_process(text):
    stemmer = PorterStemmer()
    lmtzr = WordNetLemmatizer()
    sw = stopwords.words('english')
    text = re.sub('[^a-z ]+', '', text.lower())
    text = word_tokenize(text)
    keys = [stemmer.stem(lmtzr.lemmatize(key)) for key in text if key not in sw]
    return keys      


# data = pd.read_csv("C:/Users/sricharan/Desktop/ir_assignment/songdata.csv")

# data = data.sample(n = 5000)
# data.to_csv('C:/Users/sricharan/Desktop/ir_assignment/sample.csv',index=False)

# lyrics = data['text'].values
# array_of_words = []

# for s in lyrics:
#     array_of_words.append(text_process(s))

# wordset = array_of_words[0]
# for words in array_of_words:
#     wordset = set(wordset).union(words)

# dframe = pd.DataFrame()
# for doc_no in range(len(array_of_words)):
#     print(counter)
#     dic = dict.fromkeys(wordset,0)
#     for word in array_of_words[doc_no]:
#         dic[word] += 1
#     temp = pd.DataFrame(dic,[1])
#     dframe = dframe.append(temp,ignore_index=True)
#     counter += 1

# dframe.to_csv('C:/Users/sricharan/Desktop/ir_assignment/sample_freq.csv',index=False)

# df = dict.fromkeys(wordset,0)
# counter = 0
# for w in wordset:
#     print(counter)
#     total = dframe[w].astype(bool).sum(axis=0)
#     df[w] = total
#     counter += 1

# Df = pd.DataFrame(df,[1])
# Df.to_csv('C:/Users/sricharan/Desktop/ir_assignment/sample_Df.csv',index=False)

# tf = pd.read_csv("C:/Users/sricharan/Desktop/ir_assignment/sample_freq.csv")
# df = pd.read_csv("C:/Users/sricharan/Desktop/ir_assignment/sample_df.csv")
# names = df.columns

# tf_idf = []
# norm = []
# df = df.values[0]

# for index,row in tf.iterrows():
#     temp = []
#     row = row.values
#     for i in range(len(df)):
#         if row[i] > 0:
#             temp.append(float(1+mt.log(row[i]))*mt.log(5000/df[i]))
#         else:
#             temp.append(0)
#     tf_idf.append(temp)
#     norm.append(LA.norm(temp))

# with open('C:/Users/sricharan/Desktop/ir_assignment/norm.txt', 'w') as csvfile:
#     matrixwriter = csv.writer(csvfile, delimiter=',')
#     matrixwriter.writerow(norm)

# with open('C:/Users/sricharan/Desktop/ir_assignment/tf_idf.txt', 'w') as csvfile:
#     matrixwriter = csv.writer(csvfile, delimiter=',')
#     for row in tf_idf:
#         matrixwriter.writerow(row)

# with open('C:/Users/sricharan/Desktop/ir_assignment/names.txt', 'w') as csvfile:
#     matrixwriter = csv.writer(csvfile, delimiter=',')
#     matrixwriter.writerow(names)

# In[3]:

names = pd.read_csv("C:/Users/sricharan/Desktop/ir_assignment/names.txt", sep=',').columns.values
tf_idf = pd.read_csv("C:/Users/sricharan/Desktop/ir_assignment/tf_idf.txt", sep=',', names=names)
norm = pd.read_csv("C:/Users/sricharan/Desktop/ir_assignment/norm.txt", sep=',', names=range(5000)).values[0]


# In[4]:

df = pd.read_csv("C:/Users/sricharan/Desktop/ir_assignment/sample.csv")
del df['text']
artist = df['artist'].values
title = df['song'].values
df = df.values


# In[65]:

query = "I said maybe, you're gonna be the one that saves me"
k = text_process(query)
fract = mt.sqrt(len(k))


# In[66]:

res = []
for index, row in tf_idf.iterrows():
    jacc_title = len(list(set(text_process(title[index]))&set(k)))/len(list(set(text_process(title[index]))|set(k)))
    jacc_artist = len(list(set(text_process(artist[index]))&set(k)))/len(list(set(text_process(artist[index]))|set(k)))
    val = 0
    for key in k:
        if key in names:
            val += float(row[key]/norm[index])
    val = (val/fract)*0.5 + jacc_title*0.2 + jacc_artist*0.3 
    res.append([val,index])


# In[67]:

res.sort(key=lambda x: x[0],reverse=True)


# In[68]:

for r in range(20):
    if res[r][0] == 0:
        print("\t\t\t\t**END**")
        break
    print("Artist: ",df[int(res[r][1])][0])
    print("Song: ",df[int(res[r][1])][1])
    print("Link: ",df[int(res[r][1])][2])
    print("___________________________________________________________________________________")


# In[ ]:



