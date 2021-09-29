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
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas
from sqlalchemy import create_engine
#from sets import Set
import numpy as np
import re
data=pandas.read_csv("train.csv")
#a="I like apples. I lovingly slapped him."
nrc=pandas.read_csv("nrc_emotion.csv")
engine = create_engine('sqlite://', echo=False)
nrc.to_sql('nrc', con=engine)
#k=engine.execute("SELECT * FROM nrc").fetchall()
#print(k)

term_negation = ["not","none","non","nothing","nobody","never","barely","least","no"]
term_diminisher=["occasionally","less","only","just"]
term_booster=["very","so","too"]
for a in data["text"]:
    #print(a)
    wordnet_lemmatizer = WordNetLemmatizer()

    list= tokenize.sent_tokenize(a)
    for i in range(0,len(list)):
        index_b = list[i].find(" but ")
        if (index_b != -1):
            list[i] = list[i][index_b + 5:len(list[i])]
        index_although = list[i].find("Although ")
        if (index_although != -1):
            for k in range(index_although, len(list[i])):
                if list[i][k] == "." or list[i][k] == ";" or list[i][k] == "!" or list[i][k] == ",":
                    break
            index_p = k
            #print(index_p)
            list[i] = list[i][index_p + 1:len(list[i])]
    #print(list)
    score=0
    size=0
    for sent in list:
        word_list=word_tokenize(sent.translate(sent.maketrans("","", string.punctuation)))
        #print(word_list)
        for i in range(0,len(word_list)):
            f=wordnet_lemmatizer.lemmatize(word_list[i], pos="v")
            #print(f)
            k=engine.execute("SELECT anger FROM nrc where text= :x", x=f).fetchone()
            if k is not None:
                if word_list[i-1] in term_negation:
                    score-=k[0]
                elif word_list[i-1] in term_diminisher:
                    score+=0.5*k[0]
                elif word_list[i-1] in term_booster:
                    score+=1.5*k[0]
            size=size+1
    print(score/size)