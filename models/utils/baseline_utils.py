from numpy import sqrt, dot
from numpy.linalg import norm


def compute_scores_batch(candidates_dict):
    """
    Given a dict from sentence ID to sentence token list, return a dict that maps from any sentence of ID1 in the
    candidates_dict to a dict that maps all sentence of ID2 that is not ID1 to a float that represents the
    cosine distance between ID1 and ID2.

    :param candidates_dict: a dict of sentences mapping its id to its list of tokens
    :param wm_model: word mover model
    :return: a dict mapping sentence id to a list of scores (cosine distance) with all sentences from the candidates_dict
            except for itself
    """
    # a dict mapping sentence id to a list of scores
    scores_dict = {id: {} for id in candidates_dict.keys()}

    total_cnt = len(candidates_dict)
    bow = bag_of_words(candidates_dict)
    cnt = 0 # counter for reporting progress
    for key1 in candidates_dict.keys():
        for key2 in candidates_dict.keys():
            if key1 == key2:
                continue
            else:
                scores_dict[key1][key2] = cosine_distance(one_hot_encode(bow, candidates_dict[key1]), one_hot_encode(bow, candidates_dict[key2]))
        cnt += 1
        # report progress per ten sentences processed
        if cnt % 10 == 0:
            print("Baseline computation completed: %.2f" % (cnt / total_cnt * 100))
    return scores_dict


def bag_of_words(candidates_dict):
    """
    Represent the bag of words of all token lists in candidates_dict as a list
    :param candidates_dict:
    :return: list representation of bag of words
    """
    bag = []
    for key in candidates_dict.keys():
        for token in candidates_dict[key]:
            if token not in bag:
                bag.append(token)
    return bag


def one_hot_encode(bow, tok_list):
    """

    :param bow: bag of words represented as a list
    :param tok_list: a sentence represented as a list of tokens
    :return: one hot encoding of the token list
    """
    onehot_vec = []
    for word in bow:
        if word in tok_list:
            onehot_vec.append(1)
        else:
            onehot_vec.append(0)
    return onehot_vec


def cosine_distance(vec1, vec2):
    """

    :param vec1: one hot encoded vector
    :param vec2:  one hot encoded vector
    :return: cosine distance between vec1 and vec2
    """
    return dot(vec1, vec2)/sqrt(norm(vec1) * norm(vec2))