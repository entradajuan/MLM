!curl -L https://www.gutenberg.org/cache/epub/4962/pg4962.txt --output "TheStoryOfGermLife.txt"
!curl -L https://www.gutenberg.org/cache/epub/27713/pg27713.txt --output "BacteriologicalTechnique.txt"

!pip uninstall -y tensorflow
!pip install git+https://github.com/huggingface/transformers
!pip list | grep -E 'transformers|tokenizers'

%%time

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



