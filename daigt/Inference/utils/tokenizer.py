
from tqdm import tqdm

from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer
from transformers import PreTrainedTokenizerFast

from .function import train_iter
from .data import train, test, dataset

from config.setting import CFG


###---------------------------------------------------------------------------------
### Tokenizer
###---------------------------------------------------------------------------------
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"] 

raw_tokenizer = Tokenizer(
    models.BPE(unk_token = "[UNK]")  # vocab = vocab_t0, merges = [],
)

raw_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFC()] + [normalizers.Lowercase()] if CFG.LOWERCASE else []
)

raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=CFG.VOCAB_SIZE, special_tokens=special_tokens)

raw_tokenizer.train_from_iterator(train_iter(dataset), trainer=trainer)

tokenizer = PreTrainedTokenizerFast( 
    tokenizer_object = raw_tokenizer,
    unk_token = "[UNK]",
    pad_token = "[PAD]",
    cls_token = "[CLS]",
    sep_token = "[SEP]",
    mask_token = "[MASK]",
)

tokenized_test = []
tokenized_train = []

for text in tqdm(test['text'].tolist()):
    tokenized_test.append(tokenizer.tokenize(text))

for text in tqdm(train['text'].tolist()):
    tokenized_train.append(tokenizer.tokenize(text))