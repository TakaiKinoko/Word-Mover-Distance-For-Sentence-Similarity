# Semantic Similarity Comparision For Textual Data Toolkit

## Configure Your GloVe path
`glove.840B.300d.txt` is our current choice of word embeddings. 
It's too big to be uploaded onto github, so please configure it locally following these two steps:
1. download the `glove.840B.300d.txt` file from [here](https://nlp.stanford.edu/projects/glove/).
1. update the two variables below in `constants.py` with the path to your local copy of `glove.840B.300d.txt`
```python
gloVe_path = '<your-path>/glove.840B.300d.txt'
gloVe_tmp_path = "<your-path>/glove_word2vec.txt"
```

## Usage Demo

### 1. load data from file
#### Pre-requisites
This step makes two assumptions about the data file you're passing in: 
- It has the same schema as the [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398), which is: 

| Quality | #1 ID  |  #2 ID | #1 String | #2 String |
| ------- | ------ |:------:| ---------:| ---------:|
|1| 2760337|2760373|The increase reflects lower credit losses and favorable interest rates.|The gain came as a result of fewer credit losses and lower interest rates.|
|0|149718|149404|Last year, Comcast signed 1.5 million new digital cable subscribers.| Comcast has about 21.3 million cable subscribers, many in the largest U.S. cities.|

- It's of format: `csv` or `tsv`.

#### Demo
If you simply want to read the file into a Pandas dataframe, do this:
```python
from preprocess.parser import load_data

# use '\t' as delimiter if it's a tsv file, ',' if it's csv
df = load_data(<data_path>, <delimiter>)
``` 

If instead, you want to do one or all of the listed: 
1. filter the dataframe so that we only have `Quality=1` left (meaning that the pair of Strings in this record 
are best match for each other);
1. have a dictionary mapping all the strings in the file to their respective ID;
1. have a dictionary that maps all the strings to its best match.

do this: 
```python
from preprocess.parser import parse

# use '\t' as delimiter if it's a tsv file, ',' if it's csv
# data is the filtered dataframe where every record has data['Quality'] == 1
# sent_dict maps sentence IDs to its string representation
# pair_dict maps a sentence to its best match
data, sent_dict, pair_dict = parse(<data_path>, <delimiter>)
```

#### But my data looks nothing like the [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)... what do I do?
If your data file is of a different format, don't fret. You can prepare your data so that it fits the above mentioned schema, 
or you can process your data according to the steps talked about below. Our model doesn't depend on the data loading functions 
to work. They exist simply to make testing easier.

### 2. data processing
It's far more likely that the data you have is a hashmap (python dict) mapping some sort of unique IDs to strings. To ensure that
your data is compatible with the models, only one simple step needs to be taken:

- Assuming you already have a python dict called `sent_dict` that maps IDs to Strings. We need to convert every string into 
a list of tokens (during the process also remove stopwords from the string):
```python
from models.utils import wmd_utils

# convert the [ID-to-string] dict to [ID-to-token-list] dict
candidate_dict = wmd_utils.sent_dict_to_tok_dict(sent_dict)
```


### 3. load model 
```python
from models.wmd import WMD # word-mover-distance model

model = WMD()
```

### 4. compare the similarity between two sentences
If you have two raw sentences (strings) `str1` and `str2`, whose semantic similarity you wish to obtain: 

- convert strings to token lists:
```python
from models.utils.common import get_token_list

tok_lst1 = get_token_list(str1)
tok_lst2 = get_token_list(str2)
```

- compute their similarity score:
```python
model.compute_pair_sim(tok_lst1, tok_lst2)
```

### 5. compare one target sentence against a set of other sentences
Assume that you've already converted the sentences to lists of tokens:
```python
# target: list of tokens
# candidate_dict: a dict mapping ID to its list of tokens
ordering = model.compute_sim_list(target, candidate_dict)
```
`ordering` is a list containing all IDs from `candidate_dict`, ordered from the closest to target to farthest. 

### 6. batch process a set sentences 
If you have dict mapping ID to its list of tokens `candidates`, and for each token list with ID `i`, you wish to know how close 
`i` is compared to all the other lists of tokens in `candidates`, you can call the `compute_sim_list_batch()` function that 
calls `compute(tok_lst1, tok_lst2)` under the hood for each pair in `cartesian_product(candidates.values, candidates.values)`.

```python
scores_dict = compute_sim_list_batch(candidates) 
```
Then if you'd like to find out how close token list of `id1` is to another token list `id2`, you can query the dict by" 

```python
scores_dict[id1][id2]
```

### 7. evaluate model -- TODO
TODO: how to generate `pair_dict`
```python
        
model.evaluate_model(candidate_dict, pair_dict)
```



## Code Layout
TODO

## Model Development Roadmap
To add a new model to this toolkit, the new model has to comply with the contract specified in `interface/imodel.py`. 

Abstract methods that need to be implemented in the concrete class are: 
```python

```
