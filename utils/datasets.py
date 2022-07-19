"""Module for loading and parsing datasets.

"""

import csv
import hashlib

import numpy as np
import pandas as pd


def load_data(filename):
    """Returns the features of a dataset.

    Args:
        filename (str): File location (csv formatted).

    Returns:
        tuple of np.ndarray: Tuple consisiting of the features,
            X, and the labels, Y.
    """
    data = pd.read_csv(filename)
    # data['proto'] = hashlib.sha256(data['proto'].values.tobytes())
    # data['service'] = hashlib.sha256(data['service'].values.tobytes())
    # data['state'] = hashlib.sha256(data['state'].values.tobytes())
    # data['attack_cat'] = hashlib.sha256(data['attack_cat'].values.tobytes())

    data['proto'] = data['proto'].apply(hash)
    data['service'] = data['service'].apply(hash)
    data['state'] = data['state'].apply(hash)
    data['attack_cat'] = data['attack_cat'].apply(hash)

    # data['proto'] = pd.util.hash_pandas_object(data['proto'], encoding='utf8')
    # data['service'] = pd.util.hash_pandas_object(data['service'], encoding='utf8')
    # data['state'] = pd.util.hash_pandas_object(data['state'], encoding='utf8')
    # data['attack_cat'] = pd.util.hash_pandas_object(data['attack_cat'], encoding='utf8')
    # data['proto'] = pd.to_numeric(data['proto'], errors='coerce')
    # data['service'] = pd.to_numeric(data['service'], errors='coerce')
    # data['state'] = pd.to_numeric(data['state'], errors='coerce')
    # data['attack_cat'] = pd.to_numeric(data['attack_cat'], errors='coerce')
    # data = data.replace(np.nan, 0.0, regex=True)
    data.to_numpy()
    data = data.astype(np.float32)
    # print(data)
    # with open(filename, encoding="utf-8") as f:
    #     reader = csv.reader(f)
    #     data = np.array([row for row in reader
    #                      if '#' not in row[0]]).astype(np.float32)

    X = data.values[:, 1:]
    print(X)

    Y = data.values[:, 0]
    print("Before Y")
    print(Y)
    Y = np.clip(Y, 0, 1)
    print("After Y")
    print(Y)

    return X, Y


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
