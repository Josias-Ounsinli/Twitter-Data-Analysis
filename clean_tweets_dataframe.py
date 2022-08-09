import pandas as pd


class Clean_Tweets:
    """
    The PEP8 Standard AMAZING!!!
    """
    def __init__(self, df:pd.DataFrame):
        self.df = df
        print('Automation in Action...!!!')
        
    def drop_unwanted_column(self, df:pd.DataFrame)->pd.DataFrame:
        """
        remove rows that has column names. This error originated from
        the data collection stage.  
        """
        unwanted_rows = df[df['retweet_count'] == 'retweet_count' ].index
        df.drop(unwanted_rows , inplace=True)
        columns = ['created_at', 'source', 'original_text', 'cleaned_text', 'sentiment', 'polarity', 'subjectivity',
                   'lang', 'favorite_count', 'retweet_count',
                   'original_author', 'screen_count', 'followers_count', 'friends_count', 'possibly_sensitive',
                   'hashtags', 'user_mentions', 'place', 'place_coord_boundaries', 'statuses_count']
        for col in columns:
            df = df[df[col] != col]
        return df

    def drop_duplicate(self, df:pd.DataFrame)->pd.DataFrame:
        """
        drop duplicate rows
        """
        df = df.drop_duplicates()
        return df

    def convert_to_datetime(self, df:pd.DataFrame)->pd.DataFrame:
        """
        convert column to datetime
        """
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df[df['created_at'] >= '2020-12-31']
        return df
    
    def convert_to_numbers(self, df:pd.DataFrame)->pd.DataFrame:
        """
        convert columns like polarity, subjectivity, retweet_count
        favorite_count etc to numbers
        """
        columns_to_convert = ['polarity','subjectivity, retweet_count','favorite_count', 'screen_count', 'followers_count','friends_count','statuses_count']
        df[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')
        return df
    
    def remove_non_english_tweets(self, df:pd.DataFrame)->pd.DataFrame:
        """
        remove non english tweets from lang
        """
        df = df[df['lang'] == 'en']
        return df