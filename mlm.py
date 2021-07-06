!curl -L https://raw.githubusercontent.com/PacktPublishing/Transformers-for-Natural-Language-Processing/master/Chapter03/kant.txt --output "kant.txt"

!pip uninstall -y tensorflow
!pip install git+https://github.com/huggingface/transformers
!pip list | grep -E 'transformers|tokenizers'

