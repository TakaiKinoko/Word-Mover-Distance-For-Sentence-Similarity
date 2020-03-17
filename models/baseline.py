"""

@author: Fang
"""
from models.utils.utils import get_token_list
from models.utils.baseline_utils import compute_scores_batch


class Baseline:
    """
    Baseline Model using one-hot encoding and cosine distance as similarity metric
    """
    def __init__(self, test_data):
        """
        Default constructor that initializes a baseline model
        """
        #self.baseline_model = self.load_model()  # word-to-vec model used to convert

        # map ID to cleaned sentences
        cleaned_sent_dict = {}
        for k in test_data.sent_dict.keys():  # ID to sentences
            try:
                cleaned_sent_dict[k] = get_token_list(test_data.sent_dict[k])
            except TypeError:
                continue

        # compute scores for the cartesian product of all sentences
        self.scores_dict = compute_scores_batch(cleaned_sent_dict)

