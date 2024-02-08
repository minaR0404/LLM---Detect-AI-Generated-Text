
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from .function import preprocess_function


###---------------------------------------------------------------------------------
### data
###---------------------------------------------------------------------------------
df = pd.read_parquet("./data/3.4m-human-ai.parquet")

train, test = train_test_split(df, test_size=0.1)
train, valid = train_test_split(train, test_size=0.1)

submission = test[["label"]]

ds_train = Dataset.from_pandas(train)
ds_valid = Dataset.from_pandas(valid)

ds_train_enc = ds_train.map(preprocess_function, batched=True)
ds_valid_enc = ds_valid.map(preprocess_function, batched=True)