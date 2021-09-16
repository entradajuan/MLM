!curl -L https://raw.githubusercontent.com/PacktPublishing/Transformers-for-Natural-Language-Processing/master/Chapter03/kant.txt --output "kant.txt"
!curl -L https://www.gutenberg.org/cache/epub/4962/pg4962.txt --output "TheStoryOfGermLife.txt"
!curl -L https://www.gutenberg.org/cache/epub/27713/pg27713.txt --output "BacteriologicalTechnique.txt"

import tensorflow as tf
import pandas as pd
import numpy as np

with open('kant.txt') as f:
    lines = f.readlines()

# Tokenizer

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

print("Count of characters:",tokenizer.word_counts)
print("Length of text:",tokenizer.document_count)
print("Character index",tokenizer.word_index)
print("Frequency of characters:",tokenizer.word_docs)

sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

WINDOW_SIZE = 3
def create_dataset(sentence):
  sentence = sentence.tolist()
  corpus = []
  sentence.append(0)
  sentence.append(0)
  for i, word in enumerate(sentence):
    words_set = sentence[i:i+WINDOW_SIZE]
    if len(words_set)==WINDOW_SIZE:
      if [0, 0, 0] != words_set: 
        corpus.append(words_set)
  print(type(corpus))
  return corpus

data = [create_dataset(sentence) for sentence in sequences]


# Embedding
EMBEDDING_DIM = 300
VOCAB_SIZE = len(vocab)



