# LLM---Detect-AI-Generated-Text

<br><br>
This competition was to classify whether the essay was written by a **student** or **AI**.
<br><br>

Students are given a topic and have to write an essay on it.

In the same way, AI is given a topic as topic ​​and have it output an essay.

We perform binary classification based on these labeled datasets and compete for **ROC-AUC** accuracy.
<br><br>

My team placed **108**/4358 ( **top 3%** ) and received a **silver** medal.

![スクリーンショット_20240208_182634](https://github.com/minaR0404/LLM---Detect-AI-Generated-Text/assets/49789283/7270a12d-2799-4469-906c-8b2732729daa)


# HUGE Shake #

There was a big shake-up in this competition.

Fortunately, my team was able to withstand change in the rankings, but I don't know the exact reason.

What I can say is that even if it does not include essay data, the LLM trained on a wide range of datasets has strong generality and showed robust results in this shake. .
<br><br>

# Dataset #

Datasets are very important. In this case, it would have been more effective to separate the strategies and build an appropriate dataset.

### Sparse model ###

dataset closer to test data.  Like this [DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset).
<br><br>

### LLM model ###

Not limited to test data (essays), but also various large-scale data such as news, Q&A, code and so on. <br> Here are the candidates [the Pile](https://pile.eleuther.ai/), also you can use [RedPajama](https://github.com/togethercomputer/RedPajama-Data).

- ### Tips ###

**A great idea for creating a dataset** to train a versatile LLM model like this one is shared here. [0.963 with BERT - Transformers love diverse data](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882)

> I figured more diverse data might help, so I took ~500,000 documents from The Pile, randomly truncated them, and had a variety of locally hosted LLMs (12 models) generate ~500,000 plausible continuations with a wide variety of sampling configurations. I then tried training deberta-v3-base and deberta-v3-large to classify if document continuations are from the original documents or if they were written by one of the LLMs.

<br>

# Train Tokenizer #

**Training tokenizers** is a very important idea for this competition.

This innovative idea is explained in this notebook. [Using Custom Tokenizers to Boost Your Score](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/458522)

By training a tokenizer on test data, even a **vectorizer using TF-IDF**, that is, a sparse model, becomes effective.
<br><br>

### Tokenizer ###

First set up the tokenizer. Then, tokenize the sentences in the train data and test data. 
The reference code below can be found in `daigt/inference/utils/tokenizer.py`.

```python
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
```


### Vectorizer ###

We train the TF-IDF vectorizer on the tokenized sentences of the test data.
This allows us to obtain a versatile vectorizer for test data.
It was an effective method because the tokens that appear in both human and AI texts have their own characteristics, and it was also expected to work on private datasets.
The reference code below can be found in `daigt/inference/utils/vectorizer.py`.

```python
###---------------------------------------------------------------------------------
### Vectorizer
###---------------------------------------------------------------------------------
vectorizer = TfidfVectorizer( 
    ngram_range = (3,5),
    lowercase = CFG.LOWERCASE,
    sublinear_tf = True,
    analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None,
    strip_accents = 'unicode'
)

vectorizer.fit(tokenized_test) 

vocab = vectorizer.vocabulary_

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
```
<br>

# Train LLM #

In fact, LLM is a more powerful solution for the private data set that provides the final score.
However, the data set for this competition is only a few thousand items, which is too small to be used as input to an LLM.
For this reason, **data expansion became the key to LLM**, and I believe that organizing this data was the most difficult.

A baseline for LLM using an essay-focused dataset is explained in this notebook. [[Train]DetectAI DistilRoberta[0.927]👍](https://www.kaggle.com/code/mustafakeser4/train-detectai-distilroberta-0-927)

On the other hand, there are also more advanced solutions. This discussion [0.963 with BERT - Transformers love diverse data](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882) provides some great ideas for how large language corpus, not only from essays but from all other domains, can be repurposed for this task.
<br><br>

The reference code `daigt/training/` was created based on these two ideas.
However, constructing a unique dataset using a large-scale language corpus was difficult, and the results could not be as expected.

In my case, I was acquiring multiple Open-Sourse datasets as a dataset of AI-generated sentences, but in order to build a higher quality AI sentence dataset, I think I should use the latest LLM.
Famous examples include `llama`, `mistral`, and `falcon`. However, as recent LLMs have become larger in scale, the required GPU specifications have also increased.

Furthermore, in order to improve the quality, it is necessary to generate a large amount of text data.
Parallelization using `vLLM` will probably speed up the process.
<br><br>

# Ensemble #

In this competition, the ensemble was not effective. Adding a bad model made the score worse.
In addition, there was a large difference in the ranking between the public and final results, probably because there were considerable differences between the test dataset and the private dataset.

So honestly, there was a lot of luck involved when it came to the ensemble. This is especially true for sparse models because the solution method relied on test data.
The only teams that didn't rely on luck were those at the top who had developed adequate and strong LLMs.
<br><br>

My ensemble strategy is a two-tier stacking model.

### 1st Layer ###

The first layer consists of multiple sparse models using tfidf vectors as input, and an LLM model using raw text data as input.
Regarding CV strategy, it is divided into 5-fold.
You can check the settings for each model at `daigt/inference/config/setting.py`.

```python
first_model = [
        SGD_model,
        bayes_model,
        LGBM_model,
        Cat_model,
        RF_model,
        LR_model,
    ]
```

### 2nd Layer ###

In the second layer, the predicted values ​​of each model output in the first layer were input, and the final predicted values ​​were output mainly for LGBM.

```
second_model = [
        ("lgb", LGBM_model),
        ("lr", LR_model),
    ]
```
<br>

# Score #

| Medal | Public Score | Private Score |
----|----|---- 
| Gold (18th) | 0.974 | 0.933 |
| Our Team (108th) | **0.965** | **0.909** |
| Silver (217th) | 0.964 | 0.904 |
| Blonze (435th) | 0.963 | 0.900 |

As a side note, by using [BM25](https://github.com/yutayamazaki/okapi-bm25), the private score improved from 0.909 to 0.916, which was the best.
However, it was not submitted because the public score was low.
