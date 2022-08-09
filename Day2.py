import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS,WordCloud
from gensim import corpora
import statistics
import string
import os
import re
from textblob import TextBlob

# The data
data = pd.read_csv('cleaned_fintech_data.csv')
print(data.head())
df = data.copy()

# Infos on the variables
print(df.info())
print("The number of missing value(s): {}".format(df.isnull().sum().sum()))

print(df.columns)

# Shape analysis of the data
print(df.shape)
plt.figure()
df.dtypes.value_counts().plot.pie()
plt.savefig('data_type.png')
# Look at the missing values
plt.figure(figsize=(30, 10))
sns.heatmap(df.isna(), cbar=False)
plt.savefig('Missing_data.png')


# Percentage of missing data for variable with missing values
df_missing = df[['possibly_sensitive', 'hashtags', 'place', 'place_coord_boundaries']]

print((df_missing.isna().sum()/df.shape[0]).sort_values(ascending=False))

# Data Preparation
place = df['place']
sensitivity = df['possibly_sensitive']
hashtags = df['hashtags']
place_coor = df['place_coord_boundaries']

# Drop the 4 columns with missing data
df = df.drop(['possibly_sensitive', 'hashtags', 'place', 'place_coord_boundaries'], axis = 1)

# Univarite anmlysis using language
# Tweets by language
tweets = pd.DataFrame(columns=['text','lang'])
tweets['text'] = df['clean_text'].to_list()
tweets['lang'] = df['lang'].to_list()
tweets_by_lang = tweets['lang'].value_counts()

# Plotting tweets by language
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=10)
ax.set_ylabel('Number of tweets' , fontsize=10)
ax.set_title('Top 5 languages', fontsize=10)
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='orange')
plt.savefig('tweets_by_lang.png')
## Visualisation for texts

# retain only english tweets
English_tweets = df.loc[df['lang'] =="en"]

#text Preprocessing
English_tweets['clean_text'] = English_tweets['clean_text'].astype(str)
English_tweets['clean_text'] = English_tweets['clean_text'].apply(lambda x: x.lower())
English_tweets['clean_text'] = English_tweets['clean_text'].apply(lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))

# Plot
plt.figure(figsize=(20, 10))
plt.imshow(WordCloud(width=1000, height=600, stopwords=STOPWORDS).generate(' '.join(English_tweets.clean_text.values)))
plt.axis('off')
plt.title('Most Frequent Words In Our Tweets',fontsize=16)
plt.savefig('Wordcloud_text.png')

# New data frame
cleanTweet = pd.DataFrame(columns=['text','polarity'])
cleanTweet['text'] = df['clean_text'].to_list()
cleanTweet['polarity'] = df['polarity'].to_list()

# Write a function text_category that takes a value p and returns, depending on the value of p, a string 'positive', 'negative' or 'neutral'


def text_category(p):
    if p < 0:
        category = 'negative'
    elif p == 0:
        category = 'neutral'
    else:
        category = 'positive'
    return category


# Apply this function (text_category) on the polarity column of cleanTweet in 1 above to form a new column called scores  in cleanTweet
polarities = [TextBlob(t).sentiment.polarity for t in df['original_text'].to_list()]

scores = [text_category(p) for p in polarities]

cleanTweet['scores'] = scores

plt.figure()
cleanTweet['scores'].value_counts().plot.pie()
plt.savefig('scores_plot1.png')

plt.figure()
cleanTweet['scores'].value_counts().plot(kind='bar')
plt.savefig('scores_plot2.png')
