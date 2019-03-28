"""Module to explore data.

Contains functions to help study, visualize and understand datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
# from sklearn.utils.multiclass import unique_labels

from w266_common import utils


def get_counts(target):
    """
    Checks the distribution of values in the input list
    """
    unique, counts = np.unique(target, return_counts=True)
    M = np.concatenate((unique.astype(int), counts.astype(int))).reshape((2, unique.shape[0])).T
    utils.pretty_print_matrix(M, cols=['Category', 'Count'], dtype=int)


def get_counts_by_category(target, category):

    target = np.array(target)
    category = np.array(category)
    categories = np.unique(category)

    for c in categories:
        print("{}: \t {}".format(c, target[category == c].sum()))


def random_speech(data, ids, target, descr, value=1):
    """
    Prints a random speech for a specific target value

    Args:
        data: list of speeches
        ids: list of speech ids
        target: list of 0,1
        value: int, 0 or 1
        descr: dictionary with descriptive infor
    """
    r = np.random.choice(np.array(ids)[np.array(target) == value])
    print("Speaker information")
    for i in descr[r]:
        print(i, descr[r][i])
    print()
    print(data[ids.index(r)])


def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def get_ngrams(data, ngram_range=(3, 3), top_n=10):
    """"
    # Arguments
    samples_texts: list, sample texts.
    ngram_range: tuple (min, mplt), The range of n-gram values to consider.
        Min and mplt are the lower and upper bound values for the range.
    num_ngrams: int, number of n-grams to plot.
        Top `num_ngrams` frequent n-grams will be plotted.
    """

    # Create args required for vectorizing.
    kwargs = {
        'ngram_range': ngram_range,
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',  # Split text into word tokens.
    }

    vectorizer = CountVectorizer(**kwargs)
    data_vec = vectorizer.fit_transform(data)

    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(top_n, len(all_ngrams))
    all_counts = data_vec.sum(axis=0).tolist()[0]
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])

    return list(all_ngrams)[:num_ngrams], list(all_counts)[:num_ngrams]


def plot_frequency_distribution_of_ngrams(data,
                                          ngram_range=(1, 3),
                                          num_ngrams=50):
    """
    Plots the frequency distribution of n-grams.
    """

    ngrams, counts = get_ngrams(data,
                                ngram_range=ngram_range,
                                top_n=num_ngrams)

    print(ngrams)

    idx = np.arange(num_ngrams)
    plt.figure(figsize=(20, 10))
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=90)
    plt.show()


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    speech_len = [len(s.split(' ')) for s in sample_texts]
    pct = pd.DataFrame([np.percentile(speech_len, p) for p in range(0, 101, 10)]).T
    pct.columns = list(range(0, 101, 10))
    print("Speech length percentiles")
    print(pct)

    plt.figure(figsize=(20, 10))
    plt.hist(speech_len, 50)
    plt.xlabel('Length of a speech')
    plt.ylabel('Number of speeches')
    plt.title('Speech length distribution')
    plt.show()


def ngrams_by_category(data, ids, descr, category, category_name,
                       p=False,
                       ngram_range=(3, 3),
                       top_n=10):

    np.random.seed(444)

    if p:
        sample = np.random.choice([0, 1], size=len(data), p=[1 - p, p])
    else:
        sample = np.ones(len(data), dtype=int)

    for cat in category:
        data_cat = []
        for i in range(len(ids)):
            if descr[ids[i]][category_name] == cat:
                if sample[i] == 1:
                    data_cat.append(data[i])

        if data_cat:
            ngrams = get_ngrams(data_cat,
                                ngram_range=ngram_range,
                                top_n=top_n)[0]

            print("\nFor {} {} top {} {} ngrams are:".format(category_name, cat, top_n, ngram_range))
            for n in ngrams:
                print(n)
            print("-" * 20)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    print('Confusion matrix, without normalization')
    print(cm)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_calibration_curve(y_true, y_probs, target_name, n_bins=10):
    """
    Source: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
    """

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_true, y_probs, n_bins=n_bins)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=target_name)
    ax2.hist(y_probs, range=(0, 1), bins=n_bins, histtype="step", lw=2, label=target_name)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()


def summarize_df(df, index):

    colnames = [
        'Gender_F',
        'Ethinicity_NW',
        'AvgAge',
        'Party_D',
        'Chamber_H',
        'AvgWordCount'
    ]

    # d = {k: v for k, v in descr.items() if k in set(ids)}
    # df = pd.DataFrame.from_dict(d, orient='index')
    vals = [
        np.mean(df.Female == '1.0'),
        np.mean(df.NonWhite == '1.0'),
        np.mean(pd.to_numeric(df.Age)),
        np.mean(df.Party == 'D'),
        np.mean(df.Chamber == 'H'),
        np.mean(pd.to_numeric(df.word_count))
    ]

    return pd.DataFrame(vals, columns=[index], index=colnames).T


def check_bin_probs_distr(y_probs, ids, descr, bins=[0.0, 0.4, 0.6, 1.0]):

    main_df = summarize_df(descr_df, 'base')
    print("Validation sample means:")
    print(main_df)

    y_binned = np.digitize(y_probs, bins)

    df = pd.DataFrame()

    for i in range(1, len(bins)):
        ids_bin = np.array(ids)[y_binned.flatten() == i]
        ids_df = summarize_df(descr_df.loc[np.asarray(ids_bin, dtype=int)], bins[i])
        df = df.append(ids_df)

    main_df = main_df.append([main_df] * (df.shape[0] - 1), ignore_index=True)
    main_df.index = df.index
    df = round(df.divide(main_df), 2)

    return df


def ngrams_by_bin(data, y_probs,
                  bins=[0.0, 0.4, 0.6, 1.0],
                  ngram_range=(3, 4),
                  top_n=10):

    y_binned = np.digitize(y_probs, bins)
    for i in range(1, len(bins)):
        data_bin = np.array(data)[y_binned.flatten() == i]
        ngrams = get_ngrams(data_bin,
                            ngram_range=ngram_range,
                            top_n=top_n)[0]
        print("\nIn {} bin top {} ngrams:\n".format(bins[i], top_n))
        for i in range(top_n):
            print(ngrams[i])
        print("-" * 20)


def compare_ngrams(data, y_probs,
                   bins=[0.0, 0.4, 0.6, 1.0],
                   ngram_range=(3, 4),
                   top_k=10):

    y_binned = np.digitize(y_probs, bins).flatten()

    kwargs = {
        'ngram_range': ngram_range,
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',  # Split text into word tokens.
    }

    vectorizer = CountVectorizer(**kwargs)
    data_vec = vectorizer.fit_transform(data)

    selector = SelectKBest(
        f_classif,
        k=min(top_k, data_vec.shape[1]))

    selector.fit(data_vec, y_binned)

    data_vec = selector.transform(data_vec).astype(dtype=np.float32)

    all_ngrams = np.array(vectorizer.get_feature_names())
    top_ngrams = all_ngrams[np.argsort(selector.scores_)[::-1]][:top_k]
    top_scores = selector.scores_[np.argsort(selector.scores_)[::-1]][:top_k]

    cols = selector.get_support(indices=True)
    data_vec = pd.DataFrame(data_vec.toarray())
    data_vec = pd.concat([data_vec, pd.Series(y_binned)], axis=1)
    data_vec.columns = list(all_ngrams[cols]) + ['bin']
    data_vec = data_vec.groupby('bin').sum().T
    data_vec.columns = bins[1:]

    print("\nTop {} ngrams by differentiating score:".format(top_k))
    for i in range(len(top_ngrams)):
        print(top_ngrams[i], "\t", round(top_scores[i], 1))

    return data_vec


def get_mispredictions(y_true, y_probs, data, ids, true, prob):

    if prob > 0.5:
        indices = (np.array(y_true) == true) & (y_probs.flatten() > prob)
    else:
        indices = (np.array(y_true) == true) & (y_probs.flatten() < prob)

    speeches = np.array(data)[indices]
    probs = y_probs[indices]
    ids = np.array(ids)[indices]
    ind = np.random.choice(len(speeches))
    return speeches[ind], probs[ind][0], ids[ind]


def print_mispredictions(y_true, y_probs, data, ids, descr):

    preds = get_mispredictions(y_true, y_probs, data, ids, 1, 0.9)
    print("\nTrue positive (Predicted prob: {0:.2f}):\n".format(preds[1]))
    for i in descr[preds[2]]:
        print(i, descr[preds[2]][i])
    print("\n", preds[0])
    print("-" * 20)

    preds = get_mispredictions(y_true, y_probs, data, ids, 0, 0.1)
    print("\nTrue negative (Predicted prob: {0:.2f}):\n".format(preds[1]))
    for i in descr[preds[2]]:
        print(i, descr[preds[2]][i])
    print("\n", preds[0])
    print("-" * 20)

    preds = get_mispredictions(y_true, y_probs, data, ids, 0, 0.9)
    print("\nFalse positive (Predicted prob: {0:.2f}):\n".format(preds[1]))
    for i in descr[preds[2]]:
        print(i, descr[preds[2]][i])
    print("\n", preds[0])
    print("-" * 20)

    preds = get_mispredictions(y_true, y_probs, data, ids, 1, 0.1)
    print("\nFalse negative (Predicted prob: {0:.2f}):\n".format(preds[1]))
    for i in descr[preds[2]]:
        print(i, descr[preds[2]][i])
    print("\n", preds[0])
    print("-" * 20)