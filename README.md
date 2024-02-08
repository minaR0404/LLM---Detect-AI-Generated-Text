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


# Dataset #

Datasets are very important. In this case, it would have been more effective to separate the strategies and build an appropriate dataset.

### Sparse model ###

dataset closer to test data.  Like this [DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset).

### LLM model ###

Not limited to test data (essays), but also various large-scale data such as news, Q&A, code and so on. <br> Here are the candidates [the Pile](https://pile.eleuther.ai/), also you can use [RedPajama](https://github.com/togethercomputer/RedPajama-Data).
<br><br>

- ### Tips ###

**A great idea for creating a dataset** to train a versatile LLM model like this one is shared here. [0.963 with BERT - Transformers love diverse data](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882)

> I figured more diverse data might help, so I took ~500,000 documents from The Pile, randomly truncated them, and had a variety of locally hosted LLMs (12 models) generate ~500,000 plausible continuations with a wide variety of sampling configurations. I then tried training deberta-v3-base and deberta-v3-large to classify if document continuations are from the original documents or if they were written by one of the LLMs.
