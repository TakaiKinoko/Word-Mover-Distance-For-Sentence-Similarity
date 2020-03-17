# Semantic Similarity Library: Word-Mover-Distance model
#
# Copyright (C) 2019-2020 MotleyWorks
# Author: Fang Han <fang@buymecoffee.co>

from models.utils import word2vec
from models.utils.wmd_utils import compute_scores_dict
from interface.imodel import ModelInterface
import time

class WMD(ModelInterface):
    """
    Word-Mover-Distance model
    """
    def __init__(self, test_data, sent_dict, pair_dict):
        """
        Default constructor of the Word-Mover-Distance model

        :param test_data:
        :param sent_dict:
        :param pair_dict:
        """
        self.w2vmodel = word2vec.Word2Vec.load_model()  # word-to-vec model used to convert
        self.test_data = test_data
        self.sent_dict = sent_dict
        self.pair_dict = pair_dict

    def compute_pair_sim(self, tok_lst1, tok_lst2) -> float:
        pass

    def compute_sim_list(self, target: list, candidate: dict) -> list:
        pass

    def compute_sim_list_batch(self, candidates: dict) -> dict:
        pass

    def evaluate_wmd_model(self):
        """
        Computes the percentage of queries that are matched successfully to their closest queries.

        :return: percentage of correctly predicted matches
        """
        start_time = time.time()
        correct = 0 # number of correctly predicted matches
        scores_dict = compute_scores_dict(self.sent_dict, self.w2vmodel)

        for k in scores_dict.keys():
            try:
                all_scores = scores_dict[k]
                match = min(all_scores, key=all_scores.get)
                print("Matching " + k + " " + match)
                print("\tSorted from closest to farthest: " +
                      ' '.join([str(elem) for elem in sorted(all_scores, key=all_scores.get)]))
                if match == self.pair_dict[k]:
                    correct += 1
            except KeyError:
                continue
        accuracy = correct / len(scores_dict)
        # TODO get rid of side-effects
        print("Accuracy %f" % accuracy)
        # your code
        elapsed_time = time.time() - start_time
        print("Evaluation time cost: %f"%elapsed_time)
        return accuracy