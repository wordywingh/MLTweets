# import statements
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import warnings
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

# read in the data
data = pd.read_csv("twitter.csv")

# creating a new column to store the classification labels of tweets
data["labels"] = data["class"].map({0: "Hateful", 1: "Offensive", 2: "Neither hateful nor offensive"})

# creating a new dataframe containing only the columns needed for the modeling
data_v1 = data[["tweet", "labels"]]

# download and set stopwords to be used with nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words("english")

# extending the stopwords to include other words used in twitter such as retweet(rt) etc.
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()

# text preprocessing function declaration
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

# collecting only the tweets from the csv file into a variable name tweet
tweet=data_v1.tweet
processed_tweets = preprocess(tweet)   
data_v1['processed_tweets'] = processed_tweets

# applying TF-IDF on the processed tweet
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)
tfidf = tfidf_vectorizer.fit_transform(data_v1['processed_tweets'])

# input data
X = tfidf

# output data
y = data_v1['labels']

# splitting the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14, test_size=0.2)

# ignore warnings setting on
warnings.simplefilter(action='ignore', category=FutureWarning)

# apply SMOTE oversampling to improve detection of hateful speech
smt = SMOTE( sampling_strategy={'Hateful':15367,'Offensive':15367,'Neither hateful nor offensive':15367}, k_neighbors=2, random_state=1)
X_train_smt, y_train_smt = smt.fit_resample(X_train,y_train)

# create final model after applying hypertuning parameter
final_model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l2',C=1,max_iter = 1000)
final_model.fit(X_train_smt,y_train_smt)

def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Please enter the Tweet to be analyzed:")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = tfidf_vectorizer.transform([sample]).toarray()
        a = final_model.predict(data)
        st.subheader("This Tweet is: "+ str(a))
        
hate_speech_detection()