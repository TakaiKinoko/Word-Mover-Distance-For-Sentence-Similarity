# Semantic Similarity Library: data parser
#
# Copyright (C) 2019-2020 MotleyWorks
# Author: Fang Han <fang@buymecoffee.co>

import pandas as pd
import constants
import csv
from preprocess import converter

"""
Functions related to parsing of the test data
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
    data = converter.filter_positives(data, 1) # for part1 test set
    #self.data = self.filter_positives(self.data, '1')  # for part1 test set
    # convert to dict
    records_dict = converter.df_to_dict(data)
    # create sentence dict that maps id to string
    sent_dict = converter.dict_to_sentence_dict(records_dict)
    # create pair mapping (from ID to ID)
    pair_dict = converter.dict_to_pair_dict(records_dict)
    print("Test data loaded")
    return data, sent_dict, pair_dict


def load_data(data_path, delimiter):
    """
    :param data_path: path to the csv or tsv data file
    :param delimiter: '\t' if its tsv, ',' if its csv
    :return: pandas dataframe
    """
    return pd.read_csv(data_path, delimiter=delimiter, quoting=csv.QUOTE_NONE, encoding='utf-8')
