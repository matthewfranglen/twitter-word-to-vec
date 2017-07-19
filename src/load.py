import pandas
from nltk.tokenize import TweetTokenizer

pandas.options.mode.chained_assignment = None

TOKENIZER = TweetTokenizer()

def load(filename, count):
    raw_data = ingest(filename, count)
    return postprocess(raw_data)

def ingest(filename, count):
    data = pandas.read_csv(filename, encoding='latin1')

    if count:
        data = data.head(count)

    data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data = data[data['SentimentText'].isnull() == False]
    data['Sentiment'] = data['Sentiment'].map(int)

    return data

def postprocess(data):
    data['tokens'] = data['SentimentText'].map(tokenize)
    data = data[data.tokens != 'NC']

    return data

def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = TOKENIZER.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return list(tokens)
    except:
        return 'NC'
