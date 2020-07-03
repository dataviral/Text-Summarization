import pandas as pd
from collections import Counter
import numpy as np
import contractions


def load_data(path):
    """ read csv from path """
    df = pd.read_csv(path, sep=',', encoding='latin1')
    print("Length of the data: {}".format(len(df)))
    
    df = df.dropna(axis=0)
    print("Length of the data after dropping nan: {}".format(len(df)))
    
    data = []
    for i in range(len(df)):
        summary = df.text.values[i]
        text = df.ctext.values[i]

        summary = contractions.fix(summary).lower().split(" ")
        text = contractions.fix(text).lower().split(" ")
        data.append([summary, text])
    return data


def get_counts(data):
    """ word to index, index to word mappings and word counts """
    def count(word_list, cntr):
        for word in word_list:
            cntr[word] = cntr.get(word, 0) + 1

    freq_word_counter = dict()

    for i in range(len(data)):
        summary, text = data[i]
        count(summary, freq_word_counter)
        count(text, freq_word_counter)
    
    freq_word_counter = Counter(freq_word_counter)
    return freq_word_counter


def map_data(data, max_words, cntr):
    
    w2i = dict()
    i2w = dict()

    # Most common max_words 
    common_words = cntr.most_common(max_words)
    w2i = {j[0]: i for i, j in enumerate(common_words)}
    i2w = {i: j[0] for i, j in enumerate(common_words)}

    # Add SOS, EOS, UNK and PAD 
    SOS_ID, EOS_ID, UNK_ID, PAD_ID = max_words, max_words + 1, max_words + 2, max_words + 3
    SOS_TKN, EOS_TKN, UNK_TKN, PAD_TKN = "<sos>", "<eos>", "<unk>", "<pad>"

    w2i[SOS_TKN], i2w[SOS_ID] = SOS_ID, SOS_TKN
    w2i[EOS_TKN], i2w[EOS_ID] = EOS_ID, EOS_TKN
    w2i[UNK_TKN], i2w[UNK_ID] = UNK_ID, UNK_TKN
    w2i[PAD_TKN], i2w[PAD_ID] = PAD_ID, PAD_TKN
    
    for i in range(len(data)):
        data[i][0] = [w2i.get(word, UNK_ID) for word in data[i][0]]
        data[i][1] = [w2i.get(word, UNK_ID) for word in data[i][1]] 
    
    return w2i, i2w


def get_data(path, num_words):
    # Read data from csv
    data = load_data(path)

    # Count word occurrences 
    cntr = get_counts(data)

    # Map data
    w2i, i2w = map_data(data, num_words, cntr)

    return data, w2i, i2w
