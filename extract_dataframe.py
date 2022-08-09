import json
import pandas as pd
from textblob import TextBlob
import re
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords

def read_json(json_file: str)->list:
    """
    json file reader to open and read json files into a list
    Args:
    -----
    json_file: str - path of a json file
    
    Returns
    -------
    length of the json file and a list of json
    """
    
    tweets_data = []
    for tweets in open(json_file,'r'):
        tweets_data.append(json.loads(tweets))
    
    
    return len(tweets_data), tweets_data

class TweetDfExtractor:
    """
    this function will parse tweets json into a pandas dataframe
    
    Return
    ------
    dataframe
    """
    def __init__(self, tweets_list):
        
        self.tweets_list = tweets_list

    # an example function
    def find_statuses_count(self)->list:
        statuses_count = [tweet['user']['statuses_count'] for tweet in self.tweets_list]
        return statuses_count

    def find_full_text(self)->list:
        text = [tweet['full_text'] for tweet in self.tweets_list]
        return text

    def clean_text(self, text)->list:
        clean_text = []
        for t in text:
            new = t.lower()
            new = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", new)
            stop = stopwords.words('english')
            new = " ".join([word for word in new.split() if word not in (stop)])
            clean_text.append(new)
        return clean_text

    def find_sentiment(self, text)->list:
        sentiment = [TextBlob(t).sentiment for t in text]
        return sentiment

    def find_sentiments(self, text)->list:
        polarity = [TextBlob(t).sentiment.polarity for t in text]
        self.subjectivity = [TextBlob(t).sentiment.subjectivity for t in text]
        return polarity, self.subjectivity

    def find_created_time(self)->list:
        created_at = [tweet['created_at'] for tweet in self.tweets_list]
        return created_at

    def find_source(self)->list:
        source = [tweet['source'] for tweet in self.tweets_list]
        return source

    def find_screen_name(self)->list:
        screen_name = [tweet['user']['screen_name'] for tweet in self.tweets_list]
        return screen_name

    def find_screen_count(self)->list:
        screen_count = [tweet['user']['listed_count'] for tweet in self.tweets_list]
        return screen_count

    def find_followers_count(self)->list:
        followers_count = [tweet['user']['followers_count'] for tweet in self.tweets_list]
        return followers_count

    def find_friends_count(self)->list:
        friends_count = [tweet['user']['friends_count'] for tweet in self.tweets_list]
        return friends_count

    def is_sensitive(self)->list:
        is_sensitive = []
        for tweet in self.tweets_list:
            try:
                is_sensitive.append(tweet["retweeted_status"]["possibly_sensitive"])
            except KeyError:
                try:
                    is_sensitive.append(tweet["possibly_sensitive"])
                except KeyError:
                    is_sensitive.append(None)
        return is_sensitive

    def find_favourite_count(self)->list:
        favourite_count = [tweet['user']['favourites_count'] for tweet in self.tweets_list]
        return favourite_count
    
    def find_retweet_count(self)->list:
        retweet_count = [tweet['retweet_count'] for tweet in self.tweets_list]
        return retweet_count

    def find_hashtags(self)->list:
        hashtags = [tweet['entities']['hashtags'] for tweet in self.tweets_list]
        return hashtags

    def find_lang(self)->list:
        lang = [tweet['lang'] for tweet in self.tweets_list]
        return lang

    def find_mentions(self)->list:
        mentions = [tweet['entities']['user_mentions'] for tweet in self.tweets_list]
        return mentions

    def find_location(self)->list:
        location = []
        for tweet in self.tweets_list:
            try:
                location.append(tweet['user']['location'])
            except TypeError:
                location.append('')
        return location

    def find_coordinates(self)->list:
        coordinates = [tweet['coordinates'] for tweet in self.tweets_list]
        return coordinates
        
    def get_tweet_df(self, save=False)->pd.DataFrame:
        """required column to be generated you should be creative and add more features"""
        
        columns = ['created_at', 'source', 'original_text', 'cleaned_text', 'sentiment', 'polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count',
            'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries', 'statuses_count']

        created_at = self.find_created_time()
        source = self.find_source()
        text = self.find_full_text()
        clean_text = self.clean_text(text)
        sentiment = self.find_sentiment(text)
        polarity, subjectivity = self.find_sentiments(text)
        lang = self.find_lang()
        fav_count = self.find_favourite_count()
        retweet_count = self.find_retweet_count()
        screen_name = self.find_screen_name()
        screen_count = self.find_screen_count()
        follower_count = self.find_followers_count()
        friends_count = self.find_friends_count()
        sensitivity = self.is_sensitive()
        hashtags = self.find_hashtags()
        mentions = self.find_mentions()
        location = self.find_location()
        place_coord = self.find_coordinates()
        statuses_count = self.find_statuses_count()

        data = zip(created_at, source, text, clean_text, sentiment, polarity, subjectivity, lang, fav_count, retweet_count, screen_name, screen_count, follower_count, friends_count, sensitivity, hashtags, mentions, location, place_coord, statuses_count)
        df = pd.DataFrame(data=data, columns=columns)

        if save:
            df.to_csv('processed_tweet_data.csv', index=False)
            print('File Successfully Saved.!!!')
        
        return df

                
if __name__ == "__main__":
    # required column to be generated you should be creative and add more features
    columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count',
    'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries','statuses_count']
    _, tweet_list = read_json("/home/jds98/10 Academy/Pycharm/global_twitter_data.json")
    tweet = TweetDfExtractor(tweet_list)
    tweet_df = tweet.get_tweet_df(save=True)

    # use all defined functions to generate a dataframe with the specified columns above
