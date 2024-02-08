
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import roc_auc_score

from .data import test, submission
from .tokenizer import tokenizer


###---------------------------------------------------------------------------------
### function
###---------------------------------------------------------------------------------
def preprocess_function(examples):
    return tokenizer(examples['text'], max_length=256, padding=True, truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    auc = roc_auc_score(labels, probs[:,1], multi_class='ovr')
    return {"roc_auc": auc}


def get_cv(trainer):
    test_ds = Dataset.from_pandas(test)
    test_ds_enc = test_ds.map(preprocess_function, batched=True)
    test_preds = trainer.predict(test_ds_enc)
    logits = test_preds.predictions
    probs = (np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))[:,1]

    submission['generated'] = probs.astype(np.float16)

    target = pd.DataFrame({"generated" : submission["label"]})
    auc = roc_auc_score(target["generated"], submission["generated"])
    print(f"ROC-AUC : {auc}")