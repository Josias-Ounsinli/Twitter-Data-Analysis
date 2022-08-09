from extract_dataframe import *

if __name__ == "__main__":
    # required column to be generated you should be creative and add more features
    columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count',
    'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries','statuses_count']
    _, tweet_list = read_json("/home/jds98/10 Academy/Pycharm/global_twitter_data.json")
    tweet = TweetDfExtractor(tweet_list)
    tweet_df = tweet.get_tweet_df()

true_statuses_count = list(tweet_df['statuses_count'][:5])
true_full_text = list(tweet_df['original_text'][:5])
true_polarity = list(tweet_df['polarity'][:5])
true_subjectivity = list(tweet_df['subjectivity'][:5])
true_screen_name = list(tweet_df['original_author'][:5])
true_followers_count = list(tweet_df['followers_count'][:5])
true_friends_count = list(tweet_df['friends_count'][:5])
true_sensitivity = list(tweet_df['possibly_sensitive'][:5])
true_hashtags = list(tweet_df['hashtags'][:5])
true_mentions = list(tweet_df['user_mentions'][:5])
