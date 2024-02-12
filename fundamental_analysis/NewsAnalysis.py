import http.client, urllib.parse
import json
import sys
# import keyboard

import requests
import nltk
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
import time

conn = http.client.HTTPConnection('api.mediastack.com')

params = urllib.parse.urlencode({
    'access_key': '0a3434b0bba87ed8919aa3ddfc502776',
    'sources':'cnn,bbc,bloomberg',
    'categories': 'general,business',
    'keywords':'bitcoin',
    'sort': 'published_desc',
    'limit': 20,
    })

conn.request('GET', '/v1/news?{}'.format(params))

res = conn.getresponse()
data = res.read()

converted_data = json.loads(data.decode('utf-8'))
# article_1 = converted_data["data"][0]
# title_1 = article_1["title"]
# description_1 = article_1["description"]
# url_1 = article_1["url"]
#
# article_2 = converted_data["data"][1]
# title_2 = article_2["title"]
# description_2 = article_2["description"]
# url_2 = article_2["url"]


articles = []
titles = []
descriptions = []
urls_array = []
publish_date = []


for article in range(len(converted_data["data"])):
    article_i = converted_data["data"]
    articles.append(converted_data["data"][article])
    titles.append(article_i[article]["title"])
    descriptions.append(article_i[article]["description"])
    urls_array.append(article_i[article]["url"])
    publish_date.append(article_i[article]["published_at"])

# print(articles)
# print(titles)
# print(descriptions)
# print(publish_date)

def analyze_article_sentiment(urls):
    sentiment_levels = []
    compound_scores = []

    for url in urls:
        response = requests.get(url)
        content = response.text

        soup = BeautifulSoup(content, 'html.parser')
        article_text = soup.get_text()

        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(article_text)
        compound_score = sentiment_scores['compound']
        compound_scores.append(compound_score)

        if compound_score >= 0.75:
            sentiment_level = 'Very positive'
        elif compound_score >= 0.3:
            sentiment_level = 'Fairly positive'
        elif compound_score > -0.3:
            sentiment_level = 'Neutral'
        elif compound_score > -0.75:
            sentiment_level = 'Fairly negative'
        else:
            sentiment_level = 'Very negative'

        sentiment_levels.append(sentiment_level)

    overall_score = sum(compound_scores) / len(compound_scores)

    if overall_score >= 0.75:
        overall_sentiment_level = 'Very positive'
    elif overall_score >= 0.3:
        overall_sentiment_level = 'Fairly positive'
    elif overall_score > -0.3:
        overall_sentiment_level = 'Neutral'
    elif overall_score > -0.75:
        overall_sentiment_level = 'Fairly negative'
    else:
        overall_sentiment_level = 'Very negative'


    total_score = sum(compound_scores)

    return sentiment_levels, compound_scores, overall_sentiment_level, overall_score, total_score



program_active = True
count = 0
results_file = "results.csv"

while program_active:
    print("<--------------- Running news feed sentiment analysis - Iteration: " + str(count) + " --------------->")
    sentiment_levels, compound_scores, overall_sentiment_level, overall_score, total_score = analyze_article_sentiment(urls_array)
    print("Sentiment Levels:", sentiment_levels)
    print("Compound Scores:", compound_scores)
    print("Overall sentiment level:", overall_sentiment_level)
    print("Overall Score:", overall_score)
    print("Total Score:", total_score)
    with open(results_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([sentiment_levels, compound_scores, overall_score, total_score, publish_date])
    count += 1
    time.sleep(10800)

    # if keyboard.is_pressed('enter'):
    #     program_active = False
    #     sys.exit()
