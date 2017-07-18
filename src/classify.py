import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense

def classify(tweet_w2v, x_training_set, x_testing_set, y_training_set, y_testing_set):
    tfidf, _ = make_tfidf(x_training_set)
    train_vecs_w2v = make_vectors(tweet_w2v, tfidf, x_training_set)
    test_vecs_w2v = make_vectors(tweet_w2v, tfidf, x_testing_set)
    model = make_model(tweet_w2v, train_vecs_w2v, y_training_set)

    return model.evaluate(test_vecs_w2v, y_testing_set, batch_size=128, verbose=2)

def make_tfidf(tweets):
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform(tweet.words for tweet in tweets)
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    return (tfidf, matrix)

def make_vectors(tweet_w2v, tfidf, tweets):
    vectors = numpy.concatenate([
        tweet_vector(tweet_w2v, tfidf, tweet.words)
        for tweet in tweets
    ])
    return scale(vectors)

def tweet_vector(tweet_w2v, tfidf, tokens):
    size = tweet_w2v.layer1_size
    vec = numpy.zeros(size).reshape((1, size))
    words = [
        token
        for token in tokens
        if token in tweet_w2v and token in tfidf
    ]

    if not words:
        return vec

    for word in words:
        vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
    vec /= len(words)

    return vec

def make_model(tweet_w2v, training_vectors, other_training_set):
    size = tweet_w2v.layer1_size
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=size))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(training_vectors, other_training_set, epochs=9, batch_size=32, verbose=2)

    return model
