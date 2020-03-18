# Semantic Similarity Library: utility functions
#
# Copyright (C) 2019-2020 MotleyWorks
# Author: Fang Han <fang@buymecoffee.co>

"""
Utility static functions
"""

#import pulp
from collections import defaultdict
#from scipy.spatial.distance import euclidean
from gensim.utils import tokenize
from gensim.parsing.preprocessing import remove_stopwords
#from itertools import product, combinations


def tokens_to_fracdict(tokens):
    """

    :param tokens:
    :return:
    """
    cntdict = defaultdict(lambda: 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt) / totalcnt for token, cnt in cntdict.items()}


def clean_sentences(raw_sentences):
    """
    Remove stopwords from a list of raw sentences
    :param raw_sentences:
    :return:
    """
    sentences = []
    for s in raw_sentences:
        append_to_sentences(sentences, remove_stopwords(s))
    return sentences


def get_token_list(query_string):
    """
    Given a query string, return a list of tokens that

    :param query_string: string representation of a query
    :return:
    """
    return list(tokenize(remove_stopwords(query_string)))


def append_to_sentences(sentences, string):
    """

    :param sentences:
    :param string:
    :return:
    """
    sentences.append(list(tokenize(string)))