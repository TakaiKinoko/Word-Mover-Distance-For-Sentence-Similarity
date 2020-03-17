"""
A test class is defined here with several test cases created as functions.

@author: Fang
"""
from models.wmd import WMD
from models import utils
import constants
import pandas as pd


class Test:
    def __init__(self):
        # initialize word mover model
        self.wm_model = WMD()

    def test_1(self):
        sentences = utils.clean_sentences(constants.test_sen_list)
        # print out the word mover distance between all possible combinations of sentences
        utils.compare_all_pairs(sentences, self.wm_model)
        # tests case 1
        print(utils.get_match("I want someone to climb mountains together", sentences, self.wm_model))
        # tests case 2
        print(utils.get_match("I want to go to India", sentences, self.wm_model))

    def test_2(self):
        queries = pd.read_csv(constants.query_csv_path, header=0, names=["ID", "query"],
                              usecols=["ID", 'query'])
        db = []
        for index, row in queries.iterrows():
            utils.append_to_sentences(db, row['query'])

        print(utils.get_match("I want to lose weight", db, self.wm_model))
        print(utils.get_match("I want to go to Japan", db, self.wm_model))
        print(utils.get_match("I want to go dress shopping", db, self.wm_model))