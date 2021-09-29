import sklearn
import pandas
import csv
import string
import nltk
from nltk.corpus import stopwords
from nltk.wsd import lesk
from pprint import pprint
from nltk import tokenize
from nltk.corpus import sentiwordnet
from nltk.tokenize import word_tokenize
#from sets import Set
import numpy as np
import re
texts=[]
set_of_terms = []
results=[]
'''with open('#anger.csv', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        texts.append(row[' tweet_text'])'''
#texts=["I am not good happy and joy"]
#texts = ["Since she isn't happy, she will not take an umbrella"]
#text = "Since she isn't happy, she will not take an umbrella"
a = "Although he is very good, he is very bad. I am very happy."
texts = tokenize.sent_tokenize(a)
#text = "Although he is very good, he is very bad."
term_temp = []
index_although = -1
index_b = -1
#list_words = word_tokenize(text)
for i in range(0,len(texts)):

    index_b = texts[i].find(" but ")
    if (index_b != -1):
        texts[i] = texts[i][index_b+5:len(texts[i])]
    index_although = texts[i].find("Although ")
    if(index_although != -1 ):
        for k in range(index_although,len(texts[i])):
            if texts[i][k] == "." or texts[i][k] == ";" or texts[i][k] == "!" or texts[i][k] == ",":
                break
        index_p = k
        print(index_p)
        texts[i] = texts[i][index_p + 1:len(texts[i])]

    print(texts[i])
#
# for word in list_words:
#     pprint(word)
#     if word[-1] == "t" and word[-2] =="\'" and word[-3] == "n":
#         set_of_terms.append(word)
#     else:
#         continue
# pprint(set_of_terms)
"""
set_of_terms = np.unique(set_of_terms)
pprint (set_of_terms)
"""
# for term in set_of_terms:
#     term_temp.append(term.translate(term.maketrans("","", string.punctuation)))
term_temp = ["not","none","non","nothing","nobody","never","barely","least","no",]
# term_temp = term_temp + neg_words
# pprint(term_temp)
# #file=pandas.read_csv("C:/Users/Aritra/Downloads/nrc2.csv")
# c=np.array([0,0,0,0,0,0,0,0,0,0])
sense=None
c=0
pprint(texts)
for sent in texts:
    num=0
    out = sent.translate(sent.maketrans("","", string.punctuation))
    #print(file[['Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']])
    d=nltk.word_tokenize(out)

    for k in d:
        stop = set(stopwords.words('english'))
        sense = None
        if k not in stop:
            # newword=wordnet_lemmatizer.lemmatize(word)
            # print(newword)
            # print(text)
            sense = lesk(sent, k)
        if sense is not None:
            score = sentiwordnet.senti_synset(str(sense)[8:-2])
            print(score)
                #temp = d[num-1]
            if score is not None:
                if d[num-1] in term_temp:
                #if d[num-1]=="not":
                    c=c-score
                else:
                    c=c+score
                if d[num-1]=="occasionally" or d[num-1]=="less" or d[num-1]=="only" or d[num-1]=="just":
                    score=score*0.5
                    if d[num-2]=="not":
                        c=c-score
                    else:
                        c=c+score
                if d[num-1]=="very" or d[num-1]=="so" or d[num-1]=="too":
                    c=c*1.5
                    print(score)
                    print(c)
                    if d[num-2]=="not":
                        c=-c
        num=num+1
        print(c)
    result=c/len(d)
    pprint(result)
    if result[0]<0:
        result[1]=result[1]-result[0]
        result[0]=0
    for i in range (2,len(result)):
        if(result[i]<0):
            result[i]=0
    pprint(result)
    results.append(result)
pprint(results)