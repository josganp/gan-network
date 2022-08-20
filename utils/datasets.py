"""Module for loading and parsing datasets.

"""

import csv
import hashlib

import category_encoders as ce
import numpy as np
import pandas as pd


def load_data(top_5_list_proto, top_5_list_state, filename):
    """Returns the features of a dataset.

    Args:
        filename (str): File location (csv formatted).

    Returns:
        tuple of np.ndarray: Tuple consisiting of the features,
            X, and the labels, Y.
    """

    data = pd.read_csv(filename)

    for label in top_5_list_proto:
        data['proto'+'_'+label] = np.where(data['proto'] == label, 1, 0)

    for label in top_5_list_state:
        data['state' + '_' + label] = np.where(data['state'] == label, 1, 0)

    encoder = ce.OneHotEncoder(verbose=1, cols=['service'], return_df=True,
                               use_cat_names=True)

    # Fit and transform Data
    df_encoded = encoder.fit_transform(data)

    df_new = df_encoded.drop(['proto', 'state'], axis=1)
    # df_new.insert(len(df_new.columns) - 1, 'attack_cat', df_new.pop('attack_cat'))
    df_new.insert(len(df_new.columns) - 1, 'label', df_new.pop('label'))
    # column_to_move_2 = df_new.pop('attack_cat')
    # df_new.insert(len(df_new.columns)-1, 'attack_cat', column_to_move_2)

    # print(df_new.head())

    df_new.to_numpy()

    df_new = df_new.astype(np.float32)

    # with open(filename, encoding="utf-8") as f:
    #     reader = csv.reader(f)
    #     data = np.array([row for row in reader
    #                      if 'stime' not in row[0]]).astype(np.float32)

    X = df_new.values[:, 1:]

    Y = df_new.values[:, -1]

    Y = np.clip(Y, 0, 1)

    return X, Y

def test_data(top_5_list_proto, top_5_list_state, filename):
    """Returns the features of a dataset.

    Args:
        filename (str): File location (csv formatted).

    Returns:
        tuple of np.ndarray: Tuple consisiting of the features,
            X, and the labels, Y.
    """

    data = pd.read_csv(filename)

    for label in top_5_list_proto:
        data['proto'+'_'+label] = np.where(data['proto'] == label, 1, 0)

    for label in top_5_list_state:
        data['state' + '_' + label] = np.where(data['state'] == label, 1, 0)

    encoder = ce.OneHotEncoder(verbose=1, cols=['service'], return_df=True,
                               use_cat_names=True)

    # Fit and transform Data
    df_encoded = encoder.fit_transform(data)

    df_new = df_encoded.drop(['proto', 'state'], axis=1)
    # df_new.insert(len(df_new.columns) - 1, 'attack_cat', df_new.pop('attack_cat'))
    df_new.insert(len(df_new.columns) - 1, 'label', df_new.pop('label'))
    # column_to_move_2 = df_new.pop('attack_cat')
    # df_new.insert(len(df_new.columns)-1, 'attack_cat', column_to_move_2)

    # print(df_new.head())

    df_new.to_numpy()

    df_new = df_new.astype(np.float32)

    # with open(filename, encoding="utf-8") as f:
    #     reader = csv.reader(f)
    #     data = np.array([row for row in reader
    #                      if 'stime' not in row[0]]).astype(np.float32)

    X = df_new.values[:, 1:]

    Y = df_new.values[:, -1]

    Y = np.clip(Y, 0, 1)

    return X, Y

def concat_dataset():
    df = pd.concat(map(pd.read_csv, ['UNSW-NB15-modified_1.csv', 'UNSW-NB15-modified_2.csv', 'UNSW-NB15-modified_3.csv',
                                     'UNSW-NB15-modified_4.csv']), ignore_index=True)

    df['proto'].value_counts().sort_values(ascending=False).head(20)

    # make a list of the most frequent categories of the column
    top_5_proto = [cat for cat in df['proto'].value_counts().sort_values(ascending=False).head(5).index]

    df['state'].value_counts().sort_values(ascending=False).head(20)
    # make a list of the most frequent categories of the column
    top_5_state = [cat for cat in df['state'].value_counts().sort_values(ascending=False).head(5).index]

    return top_5_proto, top_5_state

def batcher(data, batch_size=100):
    """Creates a generator to yield batches of batch_size.
    When batch is too large to fit remaining data the batch
    is clipped.

    Args:
        data (List of np.ndarray): List of data elements to be batched.
            The first dimension must be the batch size and the same
            for all data elements.
        batch_size (int = 100): Size of the mini batches.
    Yields:
        The next mini_batch in the dataset.
    """

    batch_start = 0
    batch_end = batch_size

    while batch_end < data[0].shape[0]:
        yield [el[batch_start:batch_end] for el in data]

        batch_start = batch_end
        batch_end += batch_size

    yield [el[batch_start:] for el in data]


def random_batcher(data, batch_size=100):
    """Creates a generator to yield random mini-batches of batch_size.
    When batch is too large to fit remaining data the batch
    is clipped. Will continously cycle through data.

    Args:
        data (List of np.ndarray): List of data elements to be batched.
            The first dimension must be the batch size and the same
            for all data elements.
        batch_size (int = 100): Size of the mini batches.
    Yields:
        The next mini_batch in the dataset.
    """

    while True:
        for el in data:
            np.random.shuffle(el)

        batch_start = 0
        batch_end = batch_size

        while batch_end < data[0].shape[0]:
            yield [el[batch_start:batch_end] for el in data]

            batch_start = batch_end
            batch_end += batch_size

        yield [el[batch_start:] for el in data]


def rescale(data, _min, _max, start=0.0, end=1.0, axis=0):
    """Rescale features of a dataset

    args:
        data (np.array): feature matrix.
        _min (np.array): list of minimum values per feature.
        _max (np.array): list of maximum values per feature.
        start (float = 0.): lowest value for norm.
        end (float = 1.): highest value for norm.
        axis (int = 0): axis to normalize across

    returns:
        (np.array): normalized features, the same shape as data
    """

    new_data = (data - _min) / (_max - _min)

    # check if feature is constant, will be nan in new_data
    np.place(new_data, np.isnan(new_data), 1)

    new_data = (end - start) * new_data + start

    return new_data


def vector_norm(data, start=0.0, end=1.0):
    """Scaling feature vectors

    args:
        data (np.array): feature matrix.
        _min (np.array): list of minimum values per feature.
        _max (np.array): list of maximum values per feature.
        start (float = 0.): lowest value for norm.
        end (float = 1.): highest value for norm.
        axis (int = 0): axis to normalize across

    returns:
        (np.array): normalized features, the same shape as data
    """

    new_data = data / np.sqrt(np.sum(data * data, axis=1))[:, None]
    new_data = (end - start) * new_data + start

    return new_data
