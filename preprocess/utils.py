"""


"""
import pandas as pd
import constants
import csv


#class MSRDataParser:
"""
Functions related to parsing of the MSR test data
"""
def parse(data_path = constants.msr_para_test_path, delimiter = '\t'):
    """
    Define MSR data path and read into tsv into data
    """
    #msr_path = data_path
    # read tsv file -- need to specify quoting otherwise parsing error will occur
    #self.data = pd.read_csv(self.msr_path, delimiter='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
    data = load_data(data_path, delimiter)
    # filter out only matching pairs
    data = filter_positives(data, 1) # for part1 test set
    #self.data = self.filter_positives(self.data, '1')  # for part1 test set
    # convert to dict
    records_dict = df_to_dict(data)
    # create sentence dict that maps id to string
    sent_dict = dict_to_sentence_dict(records_dict)
    # create pair mapping (from ID to ID)
    pair_dict = dict_to_pair_dict(records_dict)
    print("Test data loaded")
    return data, sent_dict, pair_dict


def load_data(data_path, delimiter):
    """
    :param data_path: path to the csv or tsv data file
    :param delimiter: '\t' if its tsv, ',' if its csv
    :return: pandas dataframe
    """
    return pd.read_csv(data_path, delimiter=delimiter, quoting=csv.QUOTE_NONE, encoding='utf-8')


def filter_positives(data, value):
    """
    :param data: the original dataframe
    :return: a dataframe that contains all the rows from data where 'Quality' is '1'
    """
    filter = data['Quality'] == value # create a boolean variable indicting if each row in data has a Quality of '1'
    return data[filter]


def df_to_dict(data):
    """
    :param data: dataframe to be converted
    :return: a dict mapping dataframe index to its corresponding records
    """
    return data.to_dict('index')


def dict_to_sentence_dict(dict):
    """
    :param dict:
    :return:
    """
    sentence_dict = {}
    for k in dict.keys():
        sentence_dict[dict[k]['#1 ID']] = dict[k]['#1 String']
        sentence_dict[dict[k]['#2 ID']] = dict[k]['#2 String']
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