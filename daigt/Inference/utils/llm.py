
import numpy as np
from datasets import Dataset
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
import torch

from .data import train, test
from .function import preprocess_function


###---------------------------------------------------------------------------------
### LLM
###---------------------------------------------------------------------------------
if len(test.text.values) <= 5:
    pass
else:
    model_checkpoint = "./data/llm-distil/distilroberta-finetuned_v5/checkpoint-13542"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device);
    trainer = Trainer(
        model,
        tokenizer=tokenizer,
    )

    train_ds = Dataset.from_pandas(train[['text']])
    train_ds_enc = train_ds.map(preprocess_function, batched=True)
    train_preds = trainer.predict(train_ds_enc)
    logits_tr = train_preds.predictions
    probs_tr = (np.exp(logits_tr) / np.sum(np.exp(logits_tr), axis=-1, keepdims=True))[:,0]


    test_ds = Dataset.from_pandas(test)
    test_ds_enc = test_ds.map(preprocess_function, batched=True)
    test_preds = trainer.predict(test_ds_enc)
    logits = test_preds.predictions
    probs = (np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))[:,0]