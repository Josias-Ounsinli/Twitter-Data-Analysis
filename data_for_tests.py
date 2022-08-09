import json
from textblob import TextBlob

data = []
for t in open("/home/jds98/10 Academy/Pycharm/africa_twitter_data.json",'r'):
    data.append(json.loads(t))

true_statuses_count = [x['user']['statuses_count'] for x in data[:5]]
true_full_text = [x['full_text'] for x in data[:5]]
true_polarity = [TextBlob(x).sentiment.polarity for x in true_full_text]
true_subjectivity = [TextBlob(x).sentiment.subjectivity for x in true_full_text]
true_screen_name = [x['user']['screen_name'] for x in data[:5]]
true_followers_count = [x['user']['followers_count'] for x in data[:5]]
true_friends_count = [x['user']['friends_count'] for x in data[:5]]
true_sensitivity = []
for x in data[:5]:
    try:
        true_sensitivity.append(x['retweeted_status']['possibly_sensitive'])
    except KeyError:
        try:
            true_sensitivity.append(x["possibly_sensitive"])
        except KeyError:
            true_sensitivity.append(None)
true_hashtags = [x['entities']['hashtags'] for x in data[:5]]
true_mentions = [x['entities']['user_mentions'] for x in data[:5]]
