# Semantic Similarity Comparision For Textual Data Toolkit

## Usage Demo

### 1. load data from file
This step makes two assumptions about the data file you're passing in: 
1. It has the same schema as the [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398), which is: 

| Quality | #1 ID  |  #2 ID | #1 String | #2 String |
| ------- | ------ |:------:| ---------:| ---------:|

1. It's of format: `csv` or `tsv`.

If you simply want to read the file into a Pandas dataframe, do this:
```python
from preprocess.parser import load_data

# use '\t' as delimiter if it's a tsv file, ',' if it'ss csv
test_data, sent_dict, pair_dict = load_data(<data_path>, <delimiter>)
``` 

If you want 
#### But my data doesn't satisfy the assumptions... what do I do?
If your data file is of a different format, don't fret. You can prepare your data so that it 

### 2. data processing


### 3. load model 

### 4. compare the similarity between two sentences

### 5. compare one target sentence against a set of other sentences

### 6. batch process a set sentences 


## Code Layout
TODO

## Model Development Roadmap
To add a new model to this toolkit, the new model has to comply with the contract specified in `interface/imodel.py`. 

Abstract methods that need to be implemented in the concrete class are: 
```python

```
