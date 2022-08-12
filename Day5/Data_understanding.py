"""Needed packages"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from textblob import TextBlob
from wordcloud import STOPWORDS, WordCloud
from scipy.stats import chi2_contingency

pd.set_option('display.max_column', 20)

"""Importing the data"""
data = pd.read_csv('processed_tweet_data.csv')
df = data.copy()

"""Looking at the different type of data"""
plt.figure()
df.dtypes.value_counts().plot.pie()
plt.savefig('My_data_type.png')

"""Look at the missing values"""
plt.figure(figsize=(20, 10))
sns.heatmap(df.isna(), cbar=False)
plt.savefig('Missing_values_mydata.png')

""" Droping place_coord_boundaries"""
df = df.drop(['place_coord_boundaries'], axis=1)

"""Percentage of missing values in sensitivity and place"""
df_missing = df[['possibly_sensitive', 'place']]
(df_missing.isna().sum() / df.shape[0]).sort_values(ascending=False)

"""Form a new data frame (named cleanTweet), containing columns cleaned_text and polarity"""
cleanTweet = pd.DataFrame(columns=['text', 'polarity'])
cleanTweet['text'] = df['cleaned_text'].to_list()
cleanTweet['polarity'] = df['polarity'].to_list()

"""Write a function text_category that takes a value p and returns, depending on the value of p, a string 'positive',
 'negative' or 'neutral'"""


def text_category(p):
    if p < 0:
        category = 'negative'
    elif p == 0:
        category = 'neutral'
    else:
        category = 'positive'
    return category


"""Apply this function (text_category) on the polarity column of cleanTweet in 1 above to form a new column called
 scores in cleanTweet"""
polarities = [TextBlob(t).sentiment.polarity for t in df['original_text'].to_list()]
scores = [text_category(p) for p in polarities]
cleanTweet['scores'] = scores

"""Construct a column scoremap Use the mapping {'positive':1, 'negative':0} on the score column"""
scoremap = {'positive': 1, 'negative': 0}
cleanTweet['scoremap'] = cleanTweet['scores'].map(scoremap)

"""Add the scoremap to the general dataset"""
df['scores'] = cleanTweet['scoremap'].to_list()

"""Remove rows from cleanTweet where score = nan (i.e where 𝐬𝐜𝐨𝐫𝐞 = Neutral) and reset the frame index."""
df = df.loc[df['scores'].isna() == False]
df['scores'] = df['scores'].astype('int')

"""Split the data in function of dtype"""
df_float = df.select_dtypes('float')
df_int = df.select_dtypes('int')
df_obj = df.select_dtypes('object')

"""Some explanatory Data Aalysis"""

""" Using bbject data"""
df_obj.head()

"""Text univariate analysis"""

"""Text univariate analysis"""

"""Text Preprocessing"""
df_text = df_obj.copy()
df_text['cleaned_text'] = df_text['cleaned_text'].astype(str)
df_text['cleaned_text'] = df_text['cleaned_text'].apply(
    lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))

"""Wordcloud"""
plt.figure(figsize=(20, 10))
plt.imshow(WordCloud(width=1000, height=600, stopwords=STOPWORDS).generate(' '.join(df_text.cleaned_text.values)))
plt.axis('off')
plt.title('Most Frequent Words In Our Tweets', fontsize=16)
plt.savefig('Wordcloud_text.png')

"""Original authors: univariate analysis"""
by_authors = df_obj['original_author'].value_counts()

"""Looking for top 5 authors of the tweets"""
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Original authors', fontsize=10)
ax.set_ylabel('Number of tweets', fontsize=10)
ax.set_title('Top 5 tweets authors', fontsize=10)
by_authors[:5].plot(ax=ax, kind='bar', color='orange')
plt.savefig('texts_by_authors.png')

"""Hashtags: univariate analysis"""

"""Method to find hashtags from texts"""


def find_hashtags(text):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', text)


"""update Hashtage columon with hashtages extracted from Orignal_text columon"""
df_obj['hashtags'] = df_obj.original_text.apply(find_hashtags)

"""take the rows from the hashtag columns where there are actually hashtags"""
hashtags_list_df = df_obj.loc[
    df_obj.hashtags.apply(
        lambda hashtags_list: hashtags_list != []
    ), ['hashtags']]

"""create dataframe where each use of hashtag gets its own row"""
flattened_hashtags_df = pd.DataFrame(
    [hashtag for hashtags_list in hashtags_list_df.hashtags
     for hashtag in hashtags_list],
    columns=['hashtag'])

"""add flatten_hashtags to the dataset"""
df_obj["flattened_hashtags"] = flattened_hashtags_df

"""Plot top 20 Hashtags"""
plt.figure()
df_obj['flattened_hashtags'].value_counts()[:20].plot(kind='bar')
plt.savefig('hashtags_plot.png')

"""Places"""
plt.figure()
df_obj['place'].value_counts()[:5].plot(kind='bar')
plt.savefig('places_plot.png')

"""Places and score (sentiments)"""
positive_data = df[df_float['scores'] == 1]
negative_data = df[df_float['scores'] == 0]

"""Plot place by sentiment"""
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
positive_data['place'].value_counts()[:5].plot(kind='bar')
plt.title("Positive sentiments")
plt.subplot(1, 2, 2)
negative_data['place'].value_counts()[:5].plot(kind='bar')
plt.title("Negative sentiments")
plt.savefig("places_vs_scores.png")

"""Float (Continuous) data analysis"""

"""Histogram of polarity and subjectivity"""
for col in df_float:
    plt.figure()
    sns.distplot(df[col])
    plt.savefig(f'{col}_plot.png')

"""Polarity, Subjectivity vs Sentiments scores"""
df_float['scores'] = df['scores']

"""Creation of two subsets of data: one for positives scores and negatives scores"""
positive_df = df_float[df_float['scores'] == 1]
negative_df = df_float[df_float['scores'] == 0]

for col in df_float.select_dtypes('float'):
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.savefig(f'{col}_vs_scores_plot.png')
    plt.legend()

"""Varaible of type integer: brief exploration"""
df_int = df_int.drop(['scores'], axis=1)
df_int['polarity'] = df_float['polarity']

"""Visualization"""
"""Correlation between the variables and with polarity"""
plt.figure()
sns.clustermap(df_int.corr())
plt.savefig('Correlation between counts vars and with polarity.png')

"""Cross tab sensitivity and score"""
plt.figure()
sns.heatmap(pd.crosstab(df['scores'], df['possibly_sensitive']), annot=True, fmt='d')
plt.savefig('crosstab_sensitive_score.png')

chi2_test = pd.crosstab(df['scores'], df['possibly_sensitive'])
stat, p, dof, expected = chi2_contingency(chi2_test)

print(p)

### Comment: Looking at the p-value, sentiments and sensitivity are dependant.
