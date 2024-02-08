
from .llm import tokenizer


###---------------------------------------------------------------------------------
### function
###---------------------------------------------------------------------------------
def train_iter(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i:i+1000]['text']


def dummy(text):
    return text


def preprocess_function(examples):
        return tokenizer(examples['text'], max_length = 512 , padding=True, truncation=True)