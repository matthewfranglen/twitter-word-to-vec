Twitter Word2Vec
================

This is just an implementation of [this blog post](http://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html).

The point is not to do this (although fun). Instead it is to create a shareable version of the outcome.

Set Up
------

The [data set](https://drive.google.com/uc?id=0B04GJPshIjmPRnZManQwWEdTZjg&export=download) should be placed in data/

I had to add `"Sentiment","ItemID","Date","SentimentSource","SentimentAuthor","SentimentText"` as the first row.

Then you need a python virtual environment:

```bash
pyenv install 3.6.1
pyenv virtualenv 3.6.1 twitter-word2vec
pyenv activate twitter-word2vec
pip install -r requirements.txt
```
