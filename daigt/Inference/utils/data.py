
import pandas as pd
import random

from datasets import Dataset


###---------------------------------------------------------------------------------
### data
###---------------------------------------------------------------------------------
test = pd.read_csv('./data/llm-detect-ai-generated-text/test_essays.csv')
submission = pd.read_csv('./data/llm-detect-ai-generated-text/sample_submission.csv')
origin = pd.read_csv('./data/llm-detect-ai-generated-text/train_essays.csv')
train = pd.read_csv('./data/daigt-v2-train-dataset/train_v2_drcat_02.csv', sep=',')

train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)


x_num = len(train)
x = random.sample(range(len(train)), x_num)

tmp = pd.DataFrame([train.iloc[i] for i in x])
train = tmp
train.reset_index(drop=True, inplace=True)


data_df = pd.concat([test.copy(), train.copy()])
data_df = data_df.drop_duplicates(subset=['text'])
data_df = data_df.reset_index(drop=True)

dataset = Dataset.from_pandas(data_df[['text']])