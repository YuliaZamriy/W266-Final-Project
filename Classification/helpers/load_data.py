#!/usr/bin/env python

import os
import numpy as np


def load_speech_data(data_path):

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
                        speech_ids.append(speech[0])
                        # some speeches have pipes in them
                        speeches.append(' '.join(speech[1:]))

                print("\nSpeeches list has {} speeches".format(len(speeches)))

    return speech_ids, speeches


def load_descr_data(descr_file_path, p):

    descr = {}
    with open(descr_file_path) as f:
        for line in f:
            if line[0] == 's':
                keys = line.strip().split('|')[1:]
            else:
                if np.random.choice(a=[0, 1], size=1, p=[1 - p, p]) == 1:
                    line = line.strip().split('|')
                    descr[line[0]] = {k: v for k, v in zip(keys, line[1:])}

    example = np.random.choice(list(descr.keys()))
    print("Random congressperson: {}".format(example))
    for i in descr[example]:
        print(i, descr[example][i])

    return descr


def create_target_labels(speech_ids, descr):

    gender, ethnicity, age, party, chamber, congress = [], [], [], [], [], []

    for i in speech_ids:
        gender.append(int(float(descr.get(i, {}).get('Female', '-1'))))
        ethnicity.append(int(float(descr.get(i, {}).get('NonWhite', '-1'))))
        age.append(int(float(descr.get(i, {}).get('Age_lt_med', '-1'))))
        party.append(descr.get(i, {}).get('Party', 'NA'))
        chamber.append(descr.get(i, {}).get('Chamber', 'NA'))
        congress.append(int(float(descr.get(i, {}).get('Congress', '-1'))))

    return gender, ethnicity, age, party, chamber, congress


def filter_data(data, ids, target, filter, value):
    data_f, ids_f, target_f = [], [], []
    for i in range(len(target)):
        if filter[i] == value:
            data_f.append(data[i])
            ids_f.append(ids[i])
            target_f.append(target[i])

    return data_f, ids_f, target_f
