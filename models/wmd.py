# Semantic Similarity Library: Word-Mover-Distance model
#
# Copyright (C) 2019-2020 MotleyWorks
# Author: Fang Han <fang@buymecoffee.co>

from models.utils import word2vec
from models.utils import wmd_utils
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
        """
        Compute the word mover distance of two sentences represented as list of tokens
        :param tok_lst1: one sentence represented as a list of tokens which has been removed of stop words
        :param tok_lst2: another sentence represented as a list of tokens which has been removed of stop words
        :return: a float in the range of [0, +infinity), which is the word mover distance of the two sentences.
                 a smaller distance indicates stronger similarity between tok_lst1 and tok_lst2.
        """
        return wmd_utils.word_mover_distance_probspec(tok_lst1, tok_lst2, self.w2vmodel)

    def compute_sim_list(self, target: list, candidates: dict) -> list:
        """
        TODO : untested
        Compute the word mover distance between target and every
        :param target: target sentence, which is a token list.
        :param candidates: a dict mapping unique sentence IDs to their corresponding token list.
        :return:
        """
        scores_dict = {}

        for id in candidates.keys():
            scores_dict[id] = self.compute_pair_sim(target, candidates[id])
        return sorted(scores_dict, key=scores_dict.get)

    def compute_sim_list_batch(self, candidates: dict) -> dict:
        """
        :param candidates: a dict mapping unique sentence IDs to their corresponding token list.
        :return:
        """
        return wmd_utils.compute_scores_dict(candidates, self.w2vmodel)

    def evaluate_model(self, candidates: dict) -> float:
        """
        Computes the percentage of queries that are matched successfully to their closest queries.

        :return: percentage of correctly predicted matches
        """
        start_time = time.time()
        correct = 0 # number of correctly predicted matches
        scores_dict = self.compute_sim_list_batch(candidates)

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
        print("Evaluation time cost: %f" % elapsed_time)
        return accuracy
