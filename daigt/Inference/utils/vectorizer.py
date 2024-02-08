
import gc

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .function import dummy
from .tokenizer import tokenized_train, tokenized_test

from config.setting import CFG


###---------------------------------------------------------------------------------
### Vectorizer
###---------------------------------------------------------------------------------
vectorizer = TfidfVectorizer(  # TfidfVectorizer
    ngram_range = (3,5),
    lowercase = CFG.LOWERCASE,
    sublinear_tf = True,
    analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None,
    strip_accents = 'unicode'
)

vectorizer.fit(tokenized_test)  # tokenized_test

vocab = vectorizer.vocabulary_

#print(len(vocab))

# fit train by using test-vocab
vectorizer = TfidfVectorizer(
    ngram_range = (3,5),
    lowercase = CFG.LOWERCASE,
    sublinear_tf = True,
    vocabulary = vocab,
    analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None,
    strip_accents = 'unicode'
)

tf_train = vectorizer.fit_transform(tokenized_train)
tf_test = vectorizer.transform(tokenized_test)

del vectorizer
gc.collect()