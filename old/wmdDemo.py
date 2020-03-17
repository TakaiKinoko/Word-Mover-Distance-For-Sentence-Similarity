import pulp
import pandas as pd
from itertools import product, combinations
from collections import defaultdict
from scipy.spatial.distance import euclidean
from gensim.utils import tokenize
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.parsing.preprocessing import remove_stopwords

def tokens_to_fracdict(tokens):
    cntdict = defaultdict(lambda : 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}

def word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
    all_tokens = set()
    for t in list(set(first_sent_tokens+second_sent_tokens)):
        try:
            wvmodel[t]
            all_tokens.add(t)
        except KeyError:
            pass

    #all_tokens = list(set(first_sent_tokens+second_sent_tokens))
    """
    wordvecs = {}
    for token in all_tokens:
        try:
            vec = wvmodel[token]
        except KeyError:
            vec = None
        if vec is not None:
            wordvecs[token] = vec
    """
    wordvecs = {token: wvmodel[token] for token in all_tokens}

    first_sent_buckets = tokens_to_fracdict(set(first_sent_tokens).intersection(all_tokens))
    second_sent_buckets = tokens_to_fracdict(set(second_sent_tokens).intersection(all_tokens))

    T = pulp.LpVariable.dicts('T_matrix', list(product(all_tokens, all_tokens)), lowBound=0)

    prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[token1, token2]*euclidean(wordvecs[token1], wordvecs[token2])
                        for token1, token2 in product(all_tokens, all_tokens)])
    for token2 in second_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets])==second_sent_buckets[token2]
    for token1 in first_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets])==first_sent_buckets[token1]

    if lpFile!=None:
        prob.writeLP(lpFile)

    prob.solve()

    return prob

def word_mover_distance(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
    prob = word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=lpFile)
    return pulp.value(prob.objective)

def append_to_sentences(sentences, string):
    sentences.append(list(tokenize(string)))

glove_file = datapath('/Users/fh/Desktop/Argo/glove.840B.300d.txt')
tmp_file = get_tmpfile("/Users/fh/Desktop/Argo/glove_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
wvmodel = KeyedVectors.load_word2vec_format(tmp_file)

sentences = []
append_to_sentences(sentences, remove_stopwords("I want to find someone to climb Mt. Everest together."))
append_to_sentences(sentences, remove_stopwords("I want to summit all mountains."))
append_to_sentences(sentences, remove_stopwords("I want to find my soulmate."))
append_to_sentences(sentences, remove_stopwords("I want to find a husband."))
append_to_sentences(sentences, remove_stopwords("I want to find a girlfriend."))
append_to_sentences(sentences, remove_stopwords("I want to learn guitar."))
append_to_sentences(sentences, remove_stopwords("I give guitar lessons."))


for s1, s2 in combinations(sentences, 2):
    prob = word_mover_distance_probspec(s1, s2, wvmodel)
    print(s1)
    print(s2)
    print(pulp.value(prob.objective))

new_s1 = list(tokenize(remove_stopwords("I want someone to climb mountains together")))
scores = {sentences.index(s):pulp.value(word_mover_distance_probspec(new_s1, s, wvmodel).objective) for s in sentences}
match = sentences[min(scores, key=scores.get)]

new_s2 = list(tokenize(remove_stopwords("I want to go to India")))
scores = {sentences.index(s):pulp.value(word_mover_distance_probspec(new_s2, s, wvmodel).objective) for s in sentences}
match = sentences[min(scores, key=scores.get)]

queries = pd.read_csv('/Users/fh/Desktop/test_data.csv', header=0, names=["ID", "query.py"], usecols=["ID", 'query.py'])
db = []
for index, row in queries.iterrows():
    append_to_sentences(db, remove_stopwords(row['query.py']))

def get_match(new_sentence, database, queries):
    toks = list(tokenize(remove_stopwords(new_sentence)))
    scores = {database.index(s): pulp.value(word_mover_distance_probspec(toks, s, wvmodel).objective) for s in
              database}
    #min_ind = min(scores, key=scores.get)
    #match = database[min(scores, key=scores.get)]
    match = queries.iloc[min(scores, key=scores.get)]  # return the actual query.py (not preprocessed tokens)
    return match


get_match("I want to lose weight", db, queries)
get_match("I want to go to Japan", db, queries)
get_match("I want to go dress shopping", db, queries)
