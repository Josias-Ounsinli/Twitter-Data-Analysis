import pandas as pd

import streamlit as st

import string
import os
import re
from wordcloud import STOPWORDS,WordCloud
import matplotlib.pyplot as plt


st.title('The data in a DataFrame')

df = pd.read_csv('data_dash.csv')

st.dataframe(df)

df['cleaned_text'] = df['cleaned_text'].astype(str)
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))

# Plot
plt.figure(figsize=(20, 10))
plt.imshow(WordCloud(width=1000, height=600, stopwords=STOPWORDS).generate(' '.join(df.cleaned_text.values)))
plt.axis('off')
plt.title('Most Frequent Words In Our Tweets',fontsize=16)
plt.show()

st.pyplot()
