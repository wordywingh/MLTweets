from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("twitter.csv")

data["labels"] = data["class"].map({0: "Hateful", 1: "Offensive", 2: "Neither hateful nor offensive"})

# creating a new dataframe containing only the columns needed for the modelling
data_v1 = data[["tweet", "labels"]]

import re
import nltk
from nltk.stem.porter import *

stemmer = PorterStemmer()
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words("english")
#extending the stopwords to include other words used in twitter such as retweet(rt) etc.
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

def preprocess(tweet):  
    
    # removal of extra spaces
    regex_pat = re.compile(r'\s+')
    tweet_space = tweet.str.replace(regex_pat, ' ')

    # removal of @name[mention]
    regex_pat = re.compile(r'@[\w\-]+')
    tweet_name = tweet_space.str.replace(regex_pat, '')

    # removal of links[https://abc.com]
    giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweets = tweet_name.str.replace(giant_url_regex, '')
    
    # removal of punctuations and numbers
    punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
    # remove whitespace with a single space
    newtweet=punc_remove.str.replace(r'\s+', ' ')
    # remove leading and trailing whitespace
    newtweet=newtweet.str.replace(r'^\s+|\s+?$','')
    # replace normal numbers with numbr
    newtweet=newtweet.str.replace(r'\d+(\.\d+)?','numbr')
    # removal of capitalization
    tweet_lower = newtweet.str.lower()
    
    # tokenizing
    tokenized_tweet = tweet_lower.apply(lambda x: x.split())
    
    # removal of stopwords
    tokenized_tweet=  tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])
    
    # stemming of the tweets
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
    
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
        tweets_p= tokenized_tweet
    
    return tweets_p

# creating a col to store cleaned up tweets
data_v1['clean_tweets'] = preprocess(data_v1.tweet)

# applying TF-IDF on the cleaned up tweet
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)

# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(data_v1['clean_tweets'] )

# input data
X = tfidf

# output data
y = data_v1['labels']

# splitting the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14, test_size=0.2)

#fitting the model to train data
model_LR = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter = 500)
model_LR.fit(X_train, y_train)
# model_LR.score(X_test, y_test)

def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = tfidf_vectorizer.transform([sample]).toarray()
        a = model_LR.predict(data)
        st.title(a)
        
hate_speech_detection()