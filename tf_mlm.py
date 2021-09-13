!curl -L https://raw.githubusercontent.com/PacktPublishing/Transformers-for-Natural-Language-Processing/master/Chapter03/kant.txt --output "kant.txt"
!curl -L https://www.gutenberg.org/cache/epub/4962/pg4962.txt --output "TheStoryOfGermLife.txt"
!curl -L https://www.gutenberg.org/cache/epub/27713/pg27713.txt --output "BacteriologicalTechnique.txt"

import tensorflow_text as tf_text
import tensorflow as tf
import pandas as pd

with open('kant.txt') as f:
    lines = f.readlines()


#text_input = ["Madrid is the capital of Spain.",
#              "I love to eat ice cream in Summer!!!",
#              "What's your name?"]

MAX_LEN = 0
text_input = []
for line in lines:
  text_input.append(line) 
  if MAX_LEN < len(line):
    MAX_LEN = len(line)

for sentence in text_input[:10]:
  print(sentence)

print(type(text_input))
tokenizer = tf_text.WhitespaceTokenizer()
tokenized = tokenizer.tokenize(text_input)

print(type(tokenized))
print(tokenized[1])

# Embedding
EMBEDDING_DIM = 300
VOCAB_SIZE = len(vocab)



