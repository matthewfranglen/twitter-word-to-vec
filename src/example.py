# import pandas as pd # provide sql-like data manipulation tools. very handy.
# import numpy as np # high dimensional vector computing library.
# from copy import deepcopy
# from string import punctuation
# from random import shuffle
# import gensim
# from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
# from tqdm import tqdm
# from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# # importing bokeh library for interactive dataviz
# import bokeh.plotting as bp
# from bokeh.models import HoverTool, BoxSelectTool
# from bokeh.plotting import figure, show, output_notebook
# 
# pd.options.mode.chained_assignment = None
# LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below
# tqdm.pandas(desc="progress-bar")
# 
# tokenizer = TweetTokenizer()
# 
# n = 250
# n_dim = 200
# 
# data = ingest()
# data = postprocess(data, n)
# 
# x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
#                                                     np.array(data.head(n).Sentiment), test_size=0.2)
# 
# def labelizeTweets(tweets, label_type):
#     labelized = []
#     for i,v in tqdm(enumerate(tweets)):
#         label = '%s_%s'%(label_type,i)
#         labelized.append(LabeledSentence(v, [label]))
#     return labelized
# 
# x_train = labelizeTweets(x_train, 'TRAIN')
# x_test = labelizeTweets(x_test, 'TEST')
# 
# tweet_w2v = Word2Vec(size=n_dim, min_count=10)
# tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
# tweet_w2v.train([x.words for x in tqdm(x_train)])
# 
# # defining the chart
# output_notebook()
# plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
#     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
#     x_axis_type=None, y_axis_type=None, min_border=1)
# 
# # getting a list of word vectors. limit to 10000. each is of 200 dimensions
# word_vectors = [tweet_w2v[w] for w in tweet_w2v.wv.vocab.keys()[:5000]]
# 
# # dimensionality reduction. converting the vectors to 2d vectors
# from sklearn.manifold import TSNE
# tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
# tsne_w2v = tsne_model.fit_transform(word_vectors)
# 
# # putting everything in a dataframe
# tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
# tsne_df['words'] = tweet_w2v.wv.vocab.keys()[:5000]
# 
# # plotting. the corresponding word appears when you hover on the data point.
# plot_tfidf.scatter(x='x', y='y', source=tsne_df)
# hover = plot_tfidf.select(dict(type=HoverTool))
# hover.tooltips={"word": "@words"}
# show(plot_tfidf)
# 
# print('building tf-idf matrix ...')
# vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
# matrix = vectorizer.fit_transform([x.words for x in x_train])
# tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
# print('vocab size :', len(tfidf))
# 
# def buildWordVector(tokens, size):
#     vec = np.zeros(size).reshape((1, size))
#     count = 0.
#     for word in tokens:
#         try:
#             vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
#             count += 1.
#         except KeyError: # handling the case where the token is not
#                          # in the corpus. useful for testing.
#             continue
#     if count != 0:
#         vec /= count
#     return vec
# 
# from sklearn.preprocessing import scale
# train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
# train_vecs_w2v = scale(train_vecs_w2v)
# 
# test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
# test_vecs_w2v = scale(test_vecs_w2v)
# 
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=200))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# 
# model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)
# 
# score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
# 
# print(score)
