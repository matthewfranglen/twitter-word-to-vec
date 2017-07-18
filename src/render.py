import pandas
from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_file
from sklearn.manifold import TSNE

def render(tweet_w2v, filename, word_count=5000):
    words = list(tweet_w2v.wv.vocab.keys())[:word_count]
    word_vectors = [tweet_w2v[word] for word in words]

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_w2v = tsne_model.fit_transform(word_vectors)

    tsne_df = pandas.DataFrame(tsne_w2v, columns=['x', 'y'])
    tsne_df['words'] = words

    output_file(filename)
    plot_tfidf = figure(
        plot_width=700,
        plot_height=600,
        title="A map of 10000 word vectors",
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
        x_axis_type=None,
        y_axis_type=None,
        min_border=1
    )

    plot_tfidf.scatter(x='x', y='y', source=tsne_df)
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips = {"word": "@words"}
    show(plot_tfidf)
