# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
nltk.download('punkt')


UNK_TAG = "<UNK>"  # unknown word assignment
PAD_TAG = "<PAD>"  # Complement value when the sentence length is not enough
UNK = 1
PAD = 0

word_idx = {PAD_TAG: PAD, UNK_TAG: UNK}  # dictionary
word_count = {}  # Statistical word frequency


def fit(sentence):
    """
    Statistical word frequency
    :param sentence:
    :return:
    """
    for word in sentence:
        # Dictionary(Dictionary) get(key,default=None) The function returns the value for the specified key, or the default value if the value is not in the dictionary.
        word_count[word] = word_count.get(word, 0) + 1


def build_vocab(min_count=5, max_count=1000, max_features=25000):
    """

    :param min_count: Minimum word frequency
    :param max_count: Maximum word frequency
    :param max_features: Maximum number of words
    :return:
    """
    global word_count
    global word_idx
    word_count = {word: count for word, count in word_count.items() if count > min_count}
    if max_count is not None:
        word_count = {word: count for word, count in word_count.items() if count <= max_count}
    if max_features is not None:
        # sort
        word_count = dict(sorted(word_count.items(), key=lambda x: x[-1], reverse=True)[:max_features])

    for word in word_count:
        # Correspond the word to its own id
        word_idx[word] = len(word_idx)  # Each word corresponds to a serial number


def transform(sentence, max_len=200):
    """
    convert a sentence into a sequence of numbers
    :param sentence:
    :param max_len: maximum sentence length
    :return:
    """
    if len(sentence) > max_len:
        # Truncate sentences when they are too long
        sentence = sentence[:max_len]
    else:
        # When the length of the sentence is not enough to the standard length, fill it
        sentence = sentence + [PAD_TAG] * (max_len - len(sentence))
    # The word in the sentence has not appeared in the dictionary is set to the number 1
    return [word_idx.get(word, UNK) for word in sentence]


# Generate a dataset in numerical form
def make_data(train_Datas, test_Datas):
    train_inputs = []
    train_labels = []
    train_labels_mul = []

    test_inputs = []
    test_labels = []
    test_labels_mul = []


    for Datas in [train_Datas, test_Datas]:
        for data in Datas:
            fit(data[0])

    build_vocab()  # number the words

    # Get the inputs and labels data converted to a sequence of numbers
    # get the training data set
    for data in train_Datas:
        train_inputs.append(transform(data[0]))
        train_labels.append(data[1])
        train_labels_mul.append(data[2])

    for data in test_Datas:
        test_inputs.append(transform(data[0]))
        test_labels.append(data[1])
        test_labels_mul.append(data[2])
    return train_inputs, train_labels, train_labels_mul, \
           test_inputs, test_labels, test_labels_mul, len(word_idx)



def get_new(text_label):
    # all_category = np.zeros(12, dtype=int)
    all_category = []
    text_label = re.split(r',', text_label)
    # print("text label", text_label)
    for ii in text_label:
        all_category.append(ii)
    return all_category

# Read text data, data format: [[['hello','word'], label]....]
class_names = ['1', '2', '3', '4', '5', '6', '10', '11', '12', '13', '14', '15', '20', '21']


from sklearn.preprocessing import MultiLabelBinarizer

def load_data(data, pre_model):
    def split_word(text):
        data = word_tokenize(text)
        return data

    data['Text'] = data['Text'].apply(split_word)
    data['Label_x'] = data['Label_x'].map(get_new)
    mlb = MultiLabelBinarizer(classes = class_names)

    # mlb_result = mlb.fit_transform([str(data['Label_x']).split(' ') for i in range(len(data))])
    label_result = mlb.fit_transform(data['Label_x'])
    # print("data['Label_x']", data['Label_x'])


    data_app = []

    # for row in data.rows:
    #     label = row['Label']
    #     content = row['Text']
    for index, row in data.iterrows():
        # label_x = row["Label_x"]
        # print("label", label_x)
        # label_xx = mlb.fit_transform(label_x)
        label_xxxx = label_result[index]
        # print("label", label_xxxx)
        label_y = row["Label_y"]
        content = row["Text"]
        # print('labelxx', label_x, label_xx)

        data_app.append([content, label_y-1, label_xxxx])
        # data_app.append([[pre_model[i] for i in content], label])
        # print(content, label)
    return data_app
