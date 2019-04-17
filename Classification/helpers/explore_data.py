#!/usr/bin/env python

"""
Module to explore data.

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
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve

from w266_common import utils


def get_counts(target):
    """
    Prints the distribution of values in the input list

    Args:
        target: list of target values
    """
    unique, counts = np.unique(target, return_counts=True)
    M = np.concatenate((unique.astype(int), counts.astype(int))).reshape((2, unique.shape[0])).T
    utils.pretty_print_matrix(M, cols=['Category', 'Count'], dtype=int)


def get_counts_by_category(target, category):
    """
    Prints the distribution of values in the input list by category

    Args:
        target: list of target values
        category: list of category values (same length as target)
    """

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
        descr: dictionary with descriptive information

    Kwargs:
        value: int, 0 or 1
    """
    r = np.random.choice(np.array(ids)[np.array(target) == value])
    print("Speaker information")
    for i in descr[r]:
        print(i, descr[r][i])
    print()
    print(data[ids.index(r)])


def get_num_words_per_sample(data):
    """
    Gets the median number of words per speech

    Args:
        data: list of speeches

    Returns:
        int, median number of words per speech
    """
    num_words = [len(s.split()) for s in data]
    return np.median(num_words)


def get_ngrams(data, ngram_range=(3, 3), top_n=10):
    """"
    Creates ngrams from input and outputs only top_n of them

    Args:
        data: list of speeches

    Kwargs:
        ngram_range: tuple (min, max), range of ngram values to consider
        top_n: int, n most frequent ngrams

    Returns:
        tuple of lists (ngrams, counts)
    """

    # vectorization parameters
    kwargs = {
        'ngram_range': ngram_range,
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',
        'min_df': 5,
        'max_df': 0.7
    }

    vectorizer = CountVectorizer(**kwargs)
    data_vec = vectorizer.fit_transform(data)

    # get ngrams values (words, phrases)
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(top_n, len(all_ngrams))

    # get total counts for each ngram
    all_counts = data_vec.sum(axis=0).tolist()[0]

    # combine ngram names and counts
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])

    # return only top_n ngrams with counts
    return list(all_ngrams)[:num_ngrams], list(all_counts)[:num_ngrams]


def plot_frequency_distribution_of_ngrams(data,
                                          ngram_range=(1, 3),
                                          num_ngrams=50):
    """
    Plots the frequency distribution of ngrams

    Args:
        data: list of speeches

    Kwargs:
        ngram_range: tuple (min, max), range of ngram values to consider
        num_ngrams: int, number of most frequent ngrams to plot
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


def plot_sample_length_distribution(data):
    """
    Plots speech length distribution

    Args:
        data: list of speeches
    """

    # get length for each speech
    speech_len = [len(s.split(' ')) for s in data]
    # calculate percentiles for speech length
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
    """
    Prints top_n ngrams by category

    Args:
        data: list of speeches
        ids: speech ids (same length as data)
        descr: dictionary with descriptive information
        category: list of category values (same length as data)
        catgory_name: string, name of the category for the chart

    Kwargs:
        p: float between 0 and 1; if not False, take sample of data
        ngram_range: tuple (min, max), range of ngram values to consider
        top_n: int, n most frequent ngrams
    """

    # random seed in case random sample is requested
    np.random.seed(444)

    # create list for sampling
    if p:
        sample = np.random.choice([0, 1], size=len(data), p=[1 - p, p])
    else:
        sample = np.ones(len(data), dtype=int)

    # create data sub-list for each category value
    for cat in category:
        data_cat = []
        for i in range(len(ids)):
            if descr[ids[i]][category_name] == cat:
                if sample[i] == 1:
                    data_cat.append(data[i])

        # get ngrams for each sub-list
        if data_cat:
            ngrams = get_ngrams(data_cat,
                                ngram_range=ngram_range,
                                top_n=top_n)[0]

            # print ngrams for sub-list
            print("\nFor {} {} top {} {} ngrams are:".format(category_name, cat, top_n, ngram_range))
            for n in ngrams:
                print(n)
            print("-" * 20)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    Prints and plots the confusion matrix.

    Args:
        y_true: list of true target labels
        y_pred: list of predicted target labels
        classes: tuple of class labels in 0, 1 order

    Kwargs:
        normalize: bool, normalize confusion matrix or not
        cmap: color map
    """

    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion matrix, without normalization')
    print(cm)

    print(classification_report(y_true, y_pred, target_names=classes))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # loop over data dimensions and create text annotations.
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

    Plots calibration curves

    Args:
        y_true: list of true target labels
        y_probs: list of predicted probabilities for each speech
        target_name: str, name of the target category

    Kwargs:
        n_bins: int, number of bins for the curve
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


def plot_compare_calibration_curves(y_true, model_types, target_name, n_bins=10):
    """
    Plots calibration curves for multiple models

    Args:
        y_true: list of true target labels
        model_types: list of tuples (list of predicted probabilities, model type name)
        target_name: str, name of the target category

    Kwargs:
        n_bins: int, number of bins for the curve
    """

    plt.figure(figsize=(10, 15))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for y_probs, name in model_types:
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_true, y_probs, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))

        ax2.hist(y_probs, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(target_name + ': Calibration plots (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.set_title(target_name + ': Predicted Target Distribution')
    ax2.legend(loc="upper right", ncol=2)

    plt.tight_layout()


def summarize_df(df, index):
    """
    Summarizes data frame by input category

    Args:
        df: data frame to separate
        index: str or list of str with category names

    Returns:
        data frame
    """
    colnames = [
        'Gender_F',
        'Ethinicity_NW',
        'AvgAge',
        'Party_D',
        'Chamber_H',
        'AvgWordCount'
    ]

    vals = [
        np.mean(df.Female),
        np.mean(df.NonWhite),
        np.mean(pd.to_numeric(df.Age)),
        np.mean(df.Party == 'D'),
        np.mean(df.Chamber == 'H'),
        np.mean(pd.to_numeric(df.word_count))
    ]

    return pd.DataFrame(vals, columns=[index], index=colnames).T


def check_bin_probs_distr(y_probs, ids, descr_df, bins=[0.0, 0.4, 0.6, 1.0]):
    """
    Checks distribution of demo variables by probability bins

    Args:
        y_probs: list of predicted probabilities for each speech
        ids: list of speech ids
        descr_df: data frame with descriptive information

    Kwargs:
        bins: list of bin cut off points
    """

    # summarize main descriptive data frame
    main_df = summarize_df(descr_df, 'base')
    print("Validation sample means:")
    print(main_df)

    # group probabilities by bins
    y_binned = np.digitize(y_probs, bins)

    df = pd.DataFrame()

    # summarize data within each probability bin
    for i in range(1, len(bins)):
        ids_bin = np.array(ids)[y_binned.flatten() == i]
        ids_df = summarize_df(descr_df.loc[np.asarray(ids_bin, dtype=int)], bins[i])
        df = df.append(ids_df)

    # replicate rows in main_df to match number of categories
    main_df = main_df.append([main_df] * (df.shape[0] - 1), ignore_index=True)
    main_df.index = df.index
    # calculate index of category data compared to overall data distribution
    df = round(df.divide(main_df), 2)

    return df


def ngrams_by_bin(data, y_probs,
                  bins=[0.0, 0.4, 0.6, 1.0],
                  ngram_range=(3, 4),
                  top_n=10):
    """
    Prints top_n ngrams by probability bin

    Args:
        data: list of speeches
        y_probs: list of predicted probabilities (same length as data)

    Kwargs:
        bins: list of bin cut off points
        ngram_range: tuple (min, max), range of ngram values to consider
        top_n: int, n most frequent ngrams
    """

    # group probabilities by bins
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
    """
    Gets top_k ngrams that differentiate the most across probability bins

    Args:
        data: list of speeches
        y_probs: list of predicted probabilities (same length as data)

    Kwargs:
        bins: list of bin cut off points
        ngram_range: tuple (min, max), range of ngram values to consider
        top_k: int, k most differentiating ngrams
    """

    # group probabilities by bins
    y_binned = np.digitize(y_probs, bins).flatten()

    kwargs = {
        'ngram_range': ngram_range,
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',  # Split text into word tokens.
        'min_df': int(len(data) / 500),
        'max_df': 0.1
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
    ngram_df = pd.DataFrame({'ngram': top_ngrams, 'score': np.round(top_scores)})

    cols = selector.get_support(indices=True)
    data_vec = pd.DataFrame(data_vec.toarray())
    data_vec = pd.concat([data_vec, pd.Series(y_binned)], axis=1)
    data_vec.columns = list(all_ngrams[cols]) + ['bin']
    data_vec = data_vec.groupby('bin').sum().T.reset_index()

    cols = ['<' + str(round(p, 2)) for p in y_probs.groupby(y_binned).max()]
    data_vec.columns = ['ngram'] + cols
    data_vec = data_vec.merge(ngram_df, on='ngram')

    print("\nTop {} ngrams by differentiating score:".format(top_k))

    return data_vec.sort_values(by='score', ascending=False).reset_index(drop=True)


def get_mispredictions(y_true, y_probs, data, ids, true, prob):
    """
    Gets examples of speeches that were classified as TP, TN, FP, FN

    Args:
        y_true: list of true target labels
        y_probs: list of predicted probabilities for each speech
        data: list of speeches
        ids: list of speech ids
        true: int, 0 or 1 for non-target or target value
        prob: float between 0 and 1 for probability cut off

    Returns:
        tuple (speech, probability, id of speech)
    """

    # get indices for positive and negative predicitons
    if prob > 0.5:
        # true and false positives
        indices = (np.array(y_true) == true) & (y_probs.flatten() > prob)
    else:
        # true and false negatives
        indices = (np.array(y_true) == true) & (y_probs.flatten() < prob)

    data_sel, y_probs_sel, ids_sel = [], [], []
    for i in range(len(indices)):
        if indices[i]:
            data_sel.append(data[i])
            y_probs_sel.append(y_probs[i])
            ids_sel.append(ids[i])
    # get index of a random speech that satisfies criteria
    ind = np.random.choice(len(data_sel))
    return data_sel[ind], y_probs_sel[ind][0], int(ids_sel[ind])


def print_mispredictions(y_true, y_probs, data, ids, descr_df):
    """
    Prints examples of speeches that were classified as TP, TN, FP, FN

    Args:
        y_true: list of true target labels
        y_probs: list of predicted probabilities for each speech
        data: list of speeches
        ids: list of speech ids
        descr_df: data frame with descriptive information
    """

    parameters = {
        'True positive': (1, 0.9),
        'True negative': (0, 0.1),
        'False positive': (0, 0.9),
        'False negative': (1, 0.1)
    }

    for par in parameters:
        preds = get_mispredictions(y_true, y_probs, data, ids, parameters[par][0], parameters[par][1])
        print("\n{} (Predicted prob: {:.2f}):\n".format(par, preds[1]))
        print(descr_df.loc[preds[2]])
        print("\n", preds[0])
        print("-" * 20)
