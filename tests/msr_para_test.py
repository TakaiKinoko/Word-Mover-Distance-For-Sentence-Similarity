# Semantic Similarity Library: Microsoft paraphrase dataset test
#
# Copyright (C) 2019-2020 MotleyWorks
# Author: Fang Han <fang@buymecoffee.co>

"""
Measure performance of the word-mover-distance model on Microsoft Research Paraphrase Corpus (MSR data for short)

Data source:
    - raw -- https://www.microsoft.com/en-us/download/details.aspx?id=52398
    - split -- https://github.com/brmson/dataset-sts/tree/master/data/para/msr

Performance metrics:
    - accuracy
    - F1 score

@author: Fang
"""

from models.baseline import Baseline
from models.wmd import WMD
from models.utils import wmd_utils
from preprocess.parser import parse


class TestWMD:
    """
    Class that automatically evaluates the Word-Mover-Distance model on msr dataset.
    Model accuracy is printed to StdOut when evaluation is done.

    Usage Example:

    """

    def __init__(self):
        """
        Default constructor that calls evaluate_wmd_model() from within
        and prints out model accuracy when call to evaluate_wmd_model() is returned.
        """
        # parse MSR data
        test_data, sent_dict, pair_dict = parse()
        # word mover model -- take long to load the model!
        wm_model = WMD(test_data, sent_dict, pair_dict)
        # copnvert the ID->String dict to ID-> token dict
        candidate_dict = wmd_utils.sent_dict_to_tok_dict(sent_dict)
        wm_model.evaluate_model(candidate_dict, pair_dict)


# class TestBaseline:
#     """
#     Class that automatically evaluates the cosine-similarity model on msr dataset.
#     Model accuracy is printed to StdOut when evaluation is done.
#
#     Usage Example:
#         test = TestBaseline()
#     """
#
#     def __init__(self):
#         """
#         Default constructor that calls evaluate_wmd_model() from within
#         and prints out model accuracy when call to evaluate_wmd_model() is returned.
#         """
#         # parse MSR data
#         self.test_data = MSRDataParser()
#         print("%d test items loaded. " % len(self.test_data.sent_dict))
#         # baseline model
#         self.model = Baseline(self.test_data)
#         self.scores_dict = self.model.scores_dict
#
#         # evaluate accuracy
#         print("Accuracy %f" % (self.evaluate_baseline()))
#
#     def evaluate_baseline(self):
#         """
#         Computes the percentage of queries that are matched successfully to their closest queries.
#
#         :return: percentage of correctly predicted matches
#         """
#         correct = 0  # number of correctly predicted matches
#         for k in self.scores_dict.keys():
#             try:
#                 all_scores = self.scores_dict[k]
#                 match = max(all_scores, key=all_scores.get)
#                 print("Matching " + k + " " + match)
#                 print("\tSorted from closest to farthest: " +
#                       ' '.join([str(elem) for elem in sorted(all_scores, key=all_scores.get, reverse=True)]))
#                 if match == self.test_data.pair_dict[k]:
#                     correct += 1
#             except KeyError:
#                 continue
#         return correct / len(self.scores_dict)


if __name__ == "__main__":
    test = TestWMD()
    #test = TestBaseline()
