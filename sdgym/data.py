import json
import logging
import os
import urllib

import numpy as np

from sdgym.constants import CATEGORICAL, ORDINAL

LOGGER = logging.getLogger(__name__)

BASE_URL = 'http://sdgym.s3.amazonaws.com/datasets/'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        url = BASE_URL + filename

        LOGGER.info('Downloading file %s to %s', url, local_path)
        urllib.request.urlretrieve(url, local_path)

    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def numeric_subset(train, test, meta, categorical_columns, ordinal_columns):
    for idx, v in enumerate(meta['columns']):
        if v['name'] == 'label':
            break

    cat_col = categorical_columns + ordinal_columns
    cat_col.remove(idx)
    train = np.delete(train, cat_col, axis=1)
    test = np.delete(test, cat_col, axis=1)

    categorical_columns = []
    ordinal_columns = []

    for idx in reversed(sorted(cat_col)):
        del meta['columns'][idx]

    assert len(meta['columns']) == train.shape[1]
    return train, test, meta, categorical_columns, ordinal_columns


def categorical_subset(train, test, meta, categorical_columns, ordinal_columns):
    for idx, v in enumerate(meta['columns']):
        if v['name'] == 'label':
            break

    num_col = set(range(train.shape[1]))
    num_col.remove(idx)
    num_col -= set(categorical_columns + ordinal_columns)
    num_col = list(num_col)
    train = np.delete(train, num_col, axis=1)
    test = np.delete(test, num_col, axis=1)
    categorical_columns = [v - sum([1 for x in num_col if x < v]) for v in categorical_columns]
    ordinal_columns = [v - sum([1 for x in num_col if x < v]) for v in ordinal_columns]

    for idx in reversed(sorted(num_col)):
        del meta['columns'][idx]

    assert len(meta['columns']) == train.shape[1]
    return train, test, meta, categorical_columns, ordinal_columns


def remove_suffix(s, suffix):
    return s[:-len(suffix)]


def load_dataset(name: str, benchmark=False):
    if name.endswith("_categorical"):
        type_subset = "categorical"
        name = remove_suffix(name, "_categorical")
    elif name.endswith("_numeric"):
        type_subset = "numeric"
        name = remove_suffix(name, "_numeric")
    else:
        type_subset = "all"

    LOGGER.info('Loading dataset %s (%s variables)', name, type_subset)
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    test = data['test']

    if type_subset != "all":
        if type_subset == 'numeric':
            train, test, meta, categorical_columns, ordinal_columns = numeric_subset(train, test, meta, categorical_columns, ordinal_columns)
        elif type_subset == 'categorical':
            train, test, meta, categorical_columns, ordinal_columns = categorical_subset(train, test, meta, categorical_columns, ordinal_columns)
        else:
            raise ValueError("type_subset should be 'numeric' or 'categorical'")

    if benchmark:
        return train, test, meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns
