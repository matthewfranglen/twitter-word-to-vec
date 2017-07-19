import argparse

def main():
    args = parse_args()

    from src import load, train, render, classify

    data = load(args.input, args.limit)
    tweet_w2v, x_train, x_test, y_train, y_test = train(data, args.dimensions)
    if args.render:
        render(tweet_w2v, args.render)

    score = classify(tweet_w2v, x_train, x_test, y_train, y_test)

    print(f'Classification completed with score of {score[1]}')

def parse_args():
    parser = argparse.ArgumentParser(description='Do random data science!')
    parser.add_argument(
        '--input',
        required=True,
        help='CSV file containing tweets'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Max count to process'
    )
    parser.add_argument(
        '--dimensions',
        type=int,
        default=200,
        help='Word vector dimension count'
    )
    parser.add_argument(
        '--render',
        help='File to write the word cloud to'
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()
