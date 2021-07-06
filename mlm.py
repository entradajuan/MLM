!curl -L https://www.gutenberg.org/cache/epub/4962/pg4962.txt --output "TheStoryOfGermLife.txt"
!curl -L https://www.gutenberg.org/cache/epub/27713/pg27713.txt --output "BacteriologicalTechnique.txt"

!pip uninstall -y tensorflow
!pip install git+https://github.com/huggingface/transformers
!pip list | grep -E 'transformers|tokenizers'

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [ str(x) for x in Path(".").glob("**/*.txt")]

print(paths)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

import os
from google.colab import drive
drive.mount('/content/gdrive')

output_dir = '/content/gdrive/MyDrive/Machine Learning/datos/MLM/modelos/model_saveTMP'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving tokenizer to %s" % output_dir)

tokenizer.save_model(output_dir)

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

!nvidia-smi

import torch
torch.cuda.is_available()

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

print(config)

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(output_dir, max_length=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model)
print(model.num_parameters())

LP=list(model.parameters())
lp=len(LP)
print(lp)
for p in range(0,lp):
  print(LP[p])





