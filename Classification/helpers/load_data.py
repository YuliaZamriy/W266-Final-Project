#!/usr/bin/env python

"""
Module to load data
"""

import os
import numpy as np


def load_speech_data(data_path):
    """
    Loads speeches from the specified data path.
    Separates speeches from speech ids.

    Args:
        data_path: path to speech files

    Returns:
        tuple (list of speech ids, list of speeches)
    """

    speech_ids, speeches = [], []

    for fname in sorted(os.listdir(data_path)):
        if fname.startswith('speeches'):
            with open(os.path.join(data_path, fname), 'rb') as f:

                raw = f.read().decode(errors='replace')
                print("\nFile {} has {} characters".format(fname, len(raw)))
                raw = raw.split('\n')[1:]
                print("and {} speeches".format(len(raw)))

                for speech in raw:
                    speech = speech.split('|')
                    # skipping empty rows
                    if speech:
                        # first element is speech id
                        speech_ids.append(speech[0])
                        # some speeches have pipes in them
                        speeches.append(' '.join(speech[1:]))

                print("\nSpeeches list has {} speeches".format(len(speeches)))

    return speech_ids, speeches


def load_descr_data(descr_file_path, p=False):
    """
    Loads speech descriptive information file into a dictionary

    Args:
        descr_file_path: path to the file
        p: float between 0 and 1; take random sample if not False

    Returns:
        dictionary with speech descriptive information
    """

    np.random.seed(444)
    # hard code number of unique speeches
    full_len = 2914465

    # create a list for random sampling
    if p:
        sample = np.random.choice([0, 1], size=full_len, p=[1 - p, p])
    else:
        sample = np.ones(full_len, dtype=int)

    descr = {}
    # create flags for iterating and counting
    counter, check, = 0, 0
    with open(descr_file_path) as f:
        for line in f:
            # first line is column names
            if line[0] == 's':
                keys = line.strip().split('|')[1:]
            else:
                if sample[counter] == 1:
                    line = line.strip().split('|')
                    if descr.get(line[0], ''):
                        descr[line[0]]['check'] += 1
                    else:
                        descr[line[0]] = {k: v for k, v in zip(keys, line[1:])}
                        descr[line[0]]['check'] = 1
            counter += 1

    # delete speeches with multiple records
    # they were incorrect fuzzy matches
    for d in set(descr):
        if descr[d]['check'] > 1:
            check += 1
            del descr[d]

    print("{} lines have been read".format(counter))
    print("{} keys had duplicates and deleted".format(check))
    print("The dictionary has {} keys".format(len(set(descr.keys()))))

    example = np.random.choice(list(descr.keys()))
    print("\nRandom congressperson: {}".format(example))
    for i in descr[example]:
        print(i, descr[example][i])

    return descr


def create_target_labels(speech_ids, descr):
    """
    Creates lists for target variables
    """

    gender, ethnicity, age, party, chamber, congress = [], [], [], [], [], []

    for i in speech_ids:
        gender.append(int(float(descr.get(i, {}).get('Female', '-1'))))
        ethnicity.append(int(float(descr.get(i, {}).get('NonWhite', '-1'))))
        age.append(int(float(descr.get(i, {}).get('Age_lt_med', '-1'))))
        party.append(descr.get(i, {}).get('Party', 'NA'))
        chamber.append(descr.get(i, {}).get('Chamber', 'NA'))
        congress.append(int(float(descr.get(i, {}).get('Congress', '-1'))))

    return gender, ethnicity, age, party, chamber, congress
