from models import word2vec
from models.wmd_utils import compute_scores_batch
from models.utils import get_token_list


class WMD:
    """
    Word-Mover-Distance model
    """
    def __init__(self, test_data, sent_dict, pair_dict):
        """
        Default constructor that initializes a word-to-vec model which is used to
        """
        self.w2vmodel = word2vec.Word2Vec.load_model()  # word-to-vec model used to convert
        self.test_data = test_data
        self.sent_dict = sent_dict
        self.pair_dict = pair_dict

    def compute_scores_dict(self):
        # map ID to cleaned sentences
        cleaned_sent_dict = {}
        for k in self.sent_dict.keys(): # ID to sentences
            try:
                cleaned_sent_dict[k] = get_token_list(self.sent_dict[k])
            except TypeError:
                continue
        # compute scores for the cartesian product of all sentences
        # THE FUNCTION BELOW TAKES VERY LONG, WILL REPORT PROGRESS
        return compute_scores_batch(cleaned_sent_dict, self.w2vmodel)
        #self.match_dict = self.match_each_sent(cleaned_sent_dict)
        # evaluate accuracy
        #print("Accuracy %f" % (self.evaluate_wmd_model()))


    def evaluate_wmd_model(self):
        """
        Computes the percentage of queries that are matched successfully to their closest queries.

        :return: percentage of correctly predicted matches
        """
        correct = 0 # number of correctly predicted matches
        scores_dict = self.compute_scores_dict()

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
        #return correct / len(self.scores_dict)
        accuracy = correct / len(scores_dict)
        print("Accuracy %f" % accuracy)