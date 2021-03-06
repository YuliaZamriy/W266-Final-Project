#!/usr/bin/env python

"""
Module to preprocess data
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

import numpy as np
import random


def split_train_val_test(data, ids, target, descr,
                         balance=1,
                         splits=[0.6, 0.2, 0.2],
                         word_count=30):
    """
    Splits speeches into train, validation and test samples

    Args:
        data: list of speeches
        ids: list of speech ids
        target: list of target values (0,1,-1)
        descr: dictionary with speech descriptive information

    Kwards:
        balance: desired ratio of 0s to 1s
        splits: list of floats for [train, validation, test] splits
        word_count: min number of words in the speech

    Returns:
        tuple of lists: train speeches, speech ids and target,
                        validation speeches, speech ids and target,
                        test speeches, speech ids and target
    """

    # initialize lists to hold speeches
    ones, zeroes = [], []
    ones_ids, zeroes_ids = [], []

    for i in range(len(target)):
        # -1 indicated non-matched speeches. They are dropped
        # select only speeches that have at least min
        # desired length
        if (target[i] != -1) and (int(descr[ids[i]]['word_count']) > word_count):
            if target[i] == 1:
                ones.append(data[i])
                ones_ids.append(ids[i])
            elif target[i] == 0:
                zeroes.append(data[i])
                zeroes_ids.append(ids[i])

    # get lengths of train, validation and test lists
    train_ones_len = int(len(ones) * splits[0])
    val_ones_len = int(len(ones) * splits[1])
    test_ones_len = len(ones) - train_ones_len - val_ones_len

    if int(len(ones) * balance) < len(zeroes):
        train_zeroes_len = int(train_ones_len * balance)
        val_zeroes_len = int(val_ones_len * balance)
        test_zeroes_len = int(test_ones_len * balance)
    else:
        train_zeroes_len = int(len(zeroes) * splits[0])
        val_zeroes_len = int(len(zeroes) * splits[1])
        test_zeroes_len = len(zeroes) - train_zeroes_len - val_zeroes_len

    # create randomly shuffled indices
    np.random.seed(100)
    ones_shuffled = np.random.permutation(np.arange(len(ones)))
    zeroes_shuffled = np.random.permutation(np.arange(len(zeroes)))

    # subset lists of speeches based on shuffled indices
    train_ones = [ones[i] for i in ones_shuffled[:train_ones_len]]
    val_ones = [ones[i] for i in ones_shuffled[train_ones_len:-test_ones_len]]
    test_ones = [ones[i] for i in ones_shuffled[-test_ones_len:]]

    train_ones_ids = [ones_ids[i] for i in ones_shuffled[:train_ones_len]]
    val_ones_ids = [ones_ids[i] for i in ones_shuffled[train_ones_len:-test_ones_len]]
    test_ones_ids = [ones_ids[i] for i in ones_shuffled[-test_ones_len:]]

    train_zeroes = [zeroes[i] for i in zeroes_shuffled[:train_zeroes_len]]
    val_zeroes = [zeroes[i] for i in zeroes_shuffled[train_zeroes_len:train_zeroes_len + val_zeroes_len]]
    test_zeroes = [zeroes[i] for i in zeroes_shuffled[-test_zeroes_len:]]

    train_zeroes_ids = [zeroes_ids[i] for i in zeroes_shuffled[:train_zeroes_len]]
    val_zeroes_ids = [zeroes_ids[i] for i in zeroes_shuffled[train_zeroes_len:train_zeroes_len + val_zeroes_len]]
    test_zeroes_ids = [zeroes_ids[i] for i in zeroes_shuffled[-test_zeroes_len:]]

    # combine lists
    train = train_ones + train_zeroes
    val = val_ones + val_zeroes
    test = test_ones + test_zeroes

    train_ids = train_ones_ids + train_zeroes_ids
    val_ids = val_ones_ids + val_zeroes_ids
    test_ids = test_ones_ids + test_zeroes_ids

    # create target lists
    train_target = [1] * len(train_ones) + [0] * len(train_zeroes)
    val_target = [1] * len(val_ones) + [0] * len(val_zeroes)
    test_target = [1] * len(test_ones) + [0] * len(test_zeroes)

    print("Training split: {} ones and {} zeroes".format(train_ones_len, train_zeroes_len))
    print("Training speech list size: {}".format(len(train)))
    print("Training target list size: {}".format(len(train_target)))

    print("Validation split: {} ones and {} zeroes".format(val_ones_len, val_zeroes_len))
    print("Validation speech list size: {}".format(len(val)))
    print("Validation target list size: {}".format(len(val_target)))

    print("Test split: {} ones and {} zeroes".format(test_ones_len, test_zeroes_len))
    print("Test speech list size: {}".format(len(test)))
    print("Test target list size: {}".format(len(test_target)))

    return train, train_ids, train_target, val, val_ids, val_target, test, test_ids, test_target


def ngram_vectorize(train, train_target, val, test, **kwargs):
    """
    Vectorizes texts as ngram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of ngrams.

    Args:
        train: list of train speeches
        train_target: list of train target labels
        val: list of validation speeches
        test: list of test speeches

    Returns:
        x_train, x_val: vectorized training and validation texts
    """

    # Create keyword arguments to pass to the 'tf-idf' vectorizer
    vec_params = {
        'ngram_range': kwargs['ngram_range'],
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',
        'min_df': kwargs['min_df'],
        'max_df': kwargs['max_df'],
    }
    vectorizer = TfidfVectorizer(**vec_params)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train)
    print("Total vocabulary size: {}".format(len(vectorizer.vocabulary_)))
    print("Number of stop words {}".format(len(vectorizer.stop_words_)))

    # Vectorize validation texts.
    x_val = vectorizer.transform(val)
    x_test = vectorizer.transform(test)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(kwargs['top_k'], x_train.shape[1]))
    selector.fit(x_train, train_target)
    x_train = selector.transform(x_train).astype(dtype=np.float32)
    x_val = selector.transform(x_val).astype(dtype=np.float32)
    x_test = selector.transform(x_test).astype(dtype=np.float32)

    all_ngrams = np.array(vectorizer.get_feature_names())

    top_ngrams = all_ngrams[np.argsort(selector.scores_)[::-1]][:kwargs['top_n']]
    top_scores = selector.scores_[np.argsort(selector.scores_)[::-1]][:kwargs['top_n']]

    bottom_ngrams = all_ngrams[np.argsort(selector.scores_)][:kwargs['top_n']]
    bottom_scores = selector.scores_[np.argsort(selector.scores_)][:kwargs['top_n']]

    print("\nTop {} ngrams by differentiating score:".format(kwargs['top_n']))
    for i in range(len(top_ngrams)):
        print(top_ngrams[i], "\t", round(top_scores[i], 1))

    print("\nBottom {} ngrams by differentiating score:".format(kwargs['top_n']))
    for i in range(len(bottom_ngrams)):
        print(bottom_ngrams[i], "\t", round(bottom_scores[i], 1))

    return x_train, x_val, x_test


def split_speech_to_chunks(data, ids, target, max_len=100):
    """
    Split speeches into chunks

    Args:
        data: list of speeches
        ids: list of speech ids
        target: list of target labels

    Kwards:
        max_len: int, chunk length

    Returns:
        tuple of lists:
            list of chunked speeches
            list of chunked ids
            list of chunked target labels
    """

    data_chunk, ids_chunk, target_chunk = [], [], []
    for i in range(len(ids)):
        words = data[i].split(' ')
        chunk = [' '.join(words[j:j + max_len]) for j in range(0, len(words), max_len)]
        data_chunk.extend(chunk)
        ids_chunk.extend(ids[i] for x in range(len(chunk)))
        target_chunk.extend(target[i] for x in range(len(chunk)))

    print("Original data has {} speeches".format(len(data)))
    print("It was split into {} chunks".format(len(data_chunk)))
    print("Checks on ids and target", len(ids_chunk), len(target_chunk))
    print("Original target mean {}".format(np.mean(target)))
    print("New target mean {}".format(np.mean(target_chunk)))

    return data_chunk, ids_chunk, target_chunk


def sequence_vectorize(train, val, test,
                       num_words=10000,
                       max_seq_length=100):
    """
    Vectorizes texts as sequence vectors.
    1 text = 1 sequence vector with fixed length.

    Args:
        train: list, training speeches
        val: list, validation speeches
        test: list, test speeches

    Kwargs:
        num_words: int, number of words to keep
        max_seq_length: int, make all sequences of this length

    Returns:
        x_train, x_val, x_test, word_index: vectorized training, validation, test
            speeches and word index dictionary.
    """

    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(
        num_words=num_words,
        lower=True,
        oov_token='<unk>')
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    # Transforms each text to a sequence of integers.
    x_train = tokenizer.texts_to_sequences(train)
    x_val = tokenizer.texts_to_sequences(val)
    x_test = tokenizer.texts_to_sequences(test)

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_seq_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_seq_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_seq_length)

    return x_train, x_val, x_test, tokenizer.word_index
