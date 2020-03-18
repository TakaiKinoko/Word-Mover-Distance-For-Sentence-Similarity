# Semantic Similarity Library: data converter
#
# Copyright (C) 2019-2020 MotleyWorks
# Author: Fang Han <fang@buymecoffee.co>


def filter_positives(data, value):
    """
    :param data: the original dataframe
    :return: a dataframe that contains all the rows from data where 'Quality' is '1'
    """
    filter = data['Quality'] == value # create a boolean variable indicting if each row in data has a Quality of '1'
    return data[filter]


def df_to_dict(df):
    """
    Unpack a dataframe into a mapping from each of its index to the corresponding record.

    :param df: dataframe
    :return: a dict mapping df's index to its corresponding records
    """
    return df.to_dict('index')


def dict_to_sentence_dict(rec_dict):
    """
    Convert a dict mapping index to record whose columns is a superset of {"#1 ID", "#1 String", "#2 ID", "#2 String"}
    into a dict mapping sentence ids to its string.
    :param rec_dict:
    :return:
    """
    sentence_dict = {}
    for k in rec_dict.keys():
        sentence_dict[rec_dict[k]['#1 ID']] = rec_dict[k]['#1 String']
        sentence_dict[rec_dict[k]['#2 ID']] = rec_dict[k]['#2 String']
    return sentence_dict


def dict_to_pair_dict(dict):
    """
    :param dict:
    :return:
    """
    pair_dict = {}
    for k in dict.keys():
        pair_dict[dict[k]['#1 ID']] = dict[k]['#2 ID']
        pair_dict[dict[k]['#2 ID']] = dict[k]['#1 ID']
    return pair_dict