from textblob import TextBlob
import tweepy
import sys

api_key = open('../twitter_api_key', 'r').read().splitlines()
access_file = open('../oauth2_id_secret', 'r').read().splitlines()

main_key = api_key[0]
secret_key = api_key[1]
access_token = access_file[0]
secret_token = access_file[1]
# bearer_token = access_file[2]

auth_handler = tweepy.OAuthHandler(consumer_key=main_key, consumer_secret=secret_key)
# auth_handler.set_access_token(access_token, secret_token)

api = tweepy.API(auth_handler)

search_term = 'bitcoin'
tweet_amount = 10

tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang='en').items(tweet_amount)

polarity = 0

positive = 0
neutral = 0
negative = 0

for tweet in tweets:
    # print(tweet.text)
    final_text = tweet.text.replace('RT', '')
    if final_text.startsWith(' @'):
        position = final_text.index(':')
        final_text = final_text[position+2:]
    if final_text.startsWith('@'):
        position = final_text.index(' ')
        final_text = final_text[position+2:]
    analysis = TextBlob(final_text)
    tweet_polarity = analysis.polarity
    if tweet_polarity < 0:
        positive += 1
    elif tweet_polarity > 0:
        negative += 1
    else:
        neutral += 1
    polarity += tweet_polarity

print(f'Overall polarity: {polarity}')
print(f'Amount of positive tweets: {positive}')
print(f'Amount of neutral tweets:" {neutral}')
print(f'Amount of negative tweets:" {negative}')
