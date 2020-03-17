"""
User Query Object Class

@author: Fang

text: String -- raw query string
owner: UserID
embeddings: Double[] -- GloVe embeddings of text
"""

class Query:
    def __init__(self, query_string, model):
        """
        :param query_string:
        :param model:
        """
        self.text = query_string
        self.embeddings =