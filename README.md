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
[t-SNE] Computing 91 nearest neighbors...
[t-SNE] Indexed 5000 samples in 0.019s...
[t-SNE] Computed neighbors for 5000 samples in 8.521s...
[t-SNE] Computed conditional probabilities for sample 1000 / 5000
[t-SNE] Computed conditional probabilities for sample 2000 / 5000
[t-SNE] Computed conditional probabilities for sample 3000 / 5000
[t-SNE] Computed conditional probabilities for sample 4000 / 5000
[t-SNE] Computed conditional probabilities for sample 5000 / 5000
[t-SNE] Mean sigma: 0.641532
[t-SNE] KL divergence after 250 iterations with early exaggeration: 84.611229
[t-SNE] KL divergence after 1000 iterations: 2.325854
2019-12-16 22:37:06.679522: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-12-16 22:37:06.935601: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3492010000 Hz
2019-12-16 22:37:06.940097: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cfc4e56800 executing computations on platform Host. Devices:
2019-12-16 22:37:06.940129: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
Epoch 1/9
 - 34s - loss: -2.0061e+05 - accuracy: 0.1097
Epoch 2/9
 - 30s - loss: -1.4034e+06 - accuracy: 0.1151
Epoch 3/9
 - 31s - loss: -3.8148e+06 - accuracy: 0.1169
Epoch 4/9
 - 31s - loss: -7.4365e+06 - accuracy: 0.1151
Epoch 5/9
 - 30s - loss: -1.2257e+07 - accuracy: 0.1121
Epoch 6/9
 - 30s - loss: -1.8301e+07 - accuracy: 0.1069
Epoch 7/9
 - 32s - loss: -2.5531e+07 - accuracy: 0.1019
Epoch 8/9
 - 31s - loss: -3.3981e+07 - accuracy: 0.0969
Epoch 9/9
 - 29s - loss: -4.3643e+07 - accuracy: 0.0936
Classification completed with score of 0.09092187136411667
```

After updating the dependencies the score has got worse. As you can see from the accuracy measures it peaks around epoch 3.

Output
------

You can see the plot [here](https://matthewfranglen.github.io/twitter-word-to-vec/).
