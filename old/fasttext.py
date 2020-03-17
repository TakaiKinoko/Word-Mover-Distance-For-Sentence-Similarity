from gensim.models.wrappers import FastText

model = FastText.load_fasttext_format('/Users/Peachy/Desktop/fastText/result/fil9.bin')

print(model.most_similar('teacher'))

print(model.similarity('teacher', 'teaches'))

print(model.similarity('I want to climb a mountain', 'I wish someone will clime a mountain with me'))

print(model.similarity('I want to climb a mountain', 'I want to go on a food tour'))

print(model.similarity('I want to find a best friend', 'I want to find my soulmate'))