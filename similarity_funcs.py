# Import libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
import string

# Read in Data (retrieved from webscraper) and set column display options

df = pd.read_csv("data/scraped_data.tsv", sep='\t')

pd.options.display.max_columns = 999

mlb = MultiLabelBinarizer()

stop_words = stopwords.words("english")

sid = SentimentIntensityAnalyzer()

mm_scaler = MinMaxScaler()

stemmer = PorterStemmer()

# Build cosine similarity function

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]

# Build function for genre similarity

def genre_similarity(input_movie, input_df):

    temp_df = input_df[['title', 'genre']]

    scores = []

    for i in temp_df['title']:

        sim = cosine_sim(str(temp_df.loc[temp_df['title'] == input_movie, 'genre']), str(temp_df.loc[temp_df['title'] == i, 'genre']))

        scores.append(sim)

    ret_df = pd.DataFrame({'title': temp_df['title'],
                           'genre_sims': scores})

    ret_df['genre_sims'] = 1 - ret_df['genre_sims']

    return ret_df

# Build function for summary similarity

def summ_similarity(input_movie, input_df):

    temp_df = input_df[['title', 'genre']]

    scores = []

    for i in temp_df['title']:

        sim = cosine_sim(str(temp_df.loc[temp_df['title'] == input_movie, 'genre']), str(temp_df.loc[temp_df['title'] == i, 'genre']))

        scores.append(sim)

    ret_df = pd.DataFrame({'title': temp_df['title'],
                           'summ_sims': scores})

    ret_df['summ_sims'] = 1 - ret_df['summ_sims']

    return ret_df

# Build function for cleaning sentiment

def sent_clean(sen_df):
    """
    Clean up the input dataframe and return a dataframe with sentiments of the summary
    :param df:
    :return:
    """

    # split the dataframe
    txt_df = sen_df[['title', 'summary']]

    # tokenize the words
    txt_df['summary'] = txt_df['summary'].apply(lambda row: word_tokenize(row))

    # remove stopwords
    txt_df['summary'] = txt_df['summary'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))

    # get them sentiments

    txt_df['sid'] = txt_df['summary'].apply(lambda x: sid.polarity_scores(x))

    # bring everything together

    ret_df = pd.concat([txt_df.drop(columns=['sid', 'summary']), txt_df['sid'].apply(pd.Series)], axis=1)

    # get just the compound column which is an aggregation of the rest

    ret_df = ret_df[['title', 'compound']]

    return ret_df


# Euclidean Distance on the sentiment scores

def sent_dist(input_movie, sen_df):

    """
    Return the euclidean distance between the two sentiment scores
    :param input_movie:
    :param sen_df:
    :return:
    """

    inp_sentiment = sen_df.loc[sen_df['title'] == input_movie, 'compound']

    dist_list = []

    for i in range(sen_df.shape[0]):

        other_sentiment = sen_df.iloc[i, 1]

        dist = math.sqrt((inp_sentiment - other_sentiment)**2)

        dist_list.append(dist)

    ret_df = pd.DataFrame({'title': sen_df['title'],
                           'sent_dist': dist_list})

    return ret_df

# Work with the ratings

def clean_ratings(inp_df):

    """
    The goal of this function is to scale the ratings between 0 and 1
    then calculate 1 minus the new scaled rating and store those as new variables
    in a new df.  This does not need similarity because we don't necessarily want similar
    scores, but better scores.  We will get one final rating by multiplying the two

    We want lower the better to add together to the rest of the metrics
    :param inp_df:
    :return:
    """

    temp_df = inp_df[['imdb_rating', 'meta_score']]

    temp_imdb = list(mm_scaler.fit_transform(temp_df[['imdb_rating']]))
    temp_meta = list(mm_scaler.fit_transform(temp_df[['meta_score']]))

    temp_df = pd.DataFrame({'title': inp_df['title'],
                            'scaled_imdb': temp_imdb,
                            'scaled_meta': temp_meta})

    temp_df['scaled_imdb'] = 1 - (temp_df['scaled_imdb'].str[0])
    temp_df['scaled_meta'] = 1 - (temp_df['scaled_meta'].str[0])

    temp_df['final_score'] = temp_df['scaled_imdb'] * temp_df['scaled_meta']

    return temp_df[['title', 'final_score']]


