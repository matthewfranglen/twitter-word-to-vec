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

Running
-------

This is a terrible score. It made a pretty plot though.

```
➜ python main.py --input data/training.1600000.processed.noemoticon.csv --render index.html
Using TensorFlow backend.
[t-SNE] Computing pairwise distances...
[t-SNE] Computing 91 nearest neighbors...
[t-SNE] Computed conditional probabilities for sample 1000 / 5000
[t-SNE] Computed conditional probabilities for sample 2000 / 5000
[t-SNE] Computed conditional probabilities for sample 3000 / 5000
[t-SNE] Computed conditional probabilities for sample 4000 / 5000
[t-SNE] Computed conditional probabilities for sample 5000 / 5000
[t-SNE] Mean sigma: 0.656117
[t-SNE] KL divergence after 100 iterations with early exaggeration: 0.962263
[t-SNE] Error after 300 iterations: 0.962263
Epoch 1/9
2017-07-19 01:05:24.456008: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-19 01:05:24.456033: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-19 01:05:24.456039: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-07-19 01:05:24.456043: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-07-19 01:05:24.456047: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
26s - loss: -1.8033e+01 - acc: 0.1395
Epoch 2/9
26s - loss: -1.8215e+01 - acc: 0.1438
Epoch 3/9
26s - loss: -1.8217e+01 - acc: 0.1403
Epoch 4/9
26s - loss: -1.8207e+01 - acc: 0.1375
Epoch 5/9
26s - loss: -1.8197e+01 - acc: 0.1362
Epoch 6/9
26s - loss: -1.8187e+01 - acc: 0.1348
Epoch 7/9
26s - loss: -1.8181e+01 - acc: 0.1340
Epoch 8/9
26s - loss: -1.8170e+01 - acc: 0.1330
Epoch 9/9
27s - loss: -1.8164e+01 - acc: 0.1325
Classification completed with score of 0.143940625
```
