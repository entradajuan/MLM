!curl -L https://raw.githubusercontent.com/PacktPublishing/Transformers-for-Natural-Language-Processing/master/Chapter03/kant.txt --output "kant.txt"
!curl -L https://www.gutenberg.org/cache/epub/4962/pg4962.txt --output "TheStoryOfGermLife.txt"
!curl -L https://www.gutenberg.org/cache/epub/27713/pg27713.txt --output "BacteriologicalTechnique.txt"


import pandas as pd

with open('kant.txt') as f:
    sentences = f.readlines()

type(sentences)

vocab = set()
data = []
MAX_LEN = 0
for sentence in sentences:
  words_in_sentence = sentence[:-1].split(" ")
  data.append(words_in_sentence)
  for word in words_in_sentence:
    vocab.add(word)
  if MAX_LEN < len(sentence):
    MAX_LEN = len(sentence)

print(MAX_LEN)
idx = 2560
print(sentences[idx])
print(data[idx])


# Tokenize
word_2_idx = {w:i for i, w in enumerate(vocab)}
idx_2_word = {i:w for i, w in enumerate(vocab)}
print(word_2_idx["war"], ' is ', idx_2_word[word_2_idx["war"]])



# Embedding
EMBEDDING_DIM = 300
VOCAB_SIZE = len(vocab)

