# Semantic Similarity Library: Word-To-Vec model for computing word embeddings
#
# Copyright (C) 2019-2020 MotleyWorks
# Author: Fang Han <fang@buymecoffee.co>

"""

"""
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import constants


class Word2Vec:
    """
    Word-To-Vec model
    """
    @staticmethod
    def load_model():
        """
        :return: Word To Vector Model
        """
        # read gloVe file
        glove_file = datapath(constants.gloVe_path)
        tmp_file = get_tmpfile(constants.gloVe_tmp_path)
        _ = glove2word2vec(glove_file, tmp_file)
        # word to vec model
        w2vmodel = KeyedVectors.load_word2vec_format(tmp_file)
        print("word to vec model loaded")
        return w2vmodel
