import numpy
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import LabeledSentence
from gensim.models.word2vec import Word2Vec

def train(tweets, dimensions):
    x_train, x_test, y_train, y_test = train_test_split(
        numpy.array(tweets.tokens),
        numpy.array(tweets.Sentiment),
        test_size=0.2
    )

    x_train = label_tweets(x_train, 'TRAIN')
    x_test = label_tweets(x_test, 'TEST')
    words = [x.words for x in x_train]

    tweet_w2v = Word2Vec(size=dimensions, min_count=10)
    tweet_w2v.build_vocab(words)
    tweet_w2v.train(words, total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

    return (tweet_w2v, x_train, x_test, y_train, y_test)

def label_tweets(tweets, label):
    return [
        LabeledSentence(tweet, [f'{label}_{index}'])
        for index, tweet in enumerate(tweets)
    ]
