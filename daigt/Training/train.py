
###---------------------------------------------------------------------------------
### library
###---------------------------------------------------------------------------------
import torch
from transformers import Trainer

from config.setting import CFG

from utils.data import ds_train_enc, ds_valid_enc
from utils.function import compute_metrics, get_cv
from utils.model import model
from utils.tokenizer import tokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device);


###---------------------------------------------------------------------------------
### Training
###---------------------------------------------------------------------------------
if __name__ == '__main__':
    
    trainer = Trainer(
        model,
        CFG.args,
        train_dataset = ds_train_enc,
        eval_dataset = ds_valid_enc,
        tokenizer = tokenizer,
        callbacks = [CFG.early_stopping],
        compute_metrics = compute_metrics
    )

    trainer.train()

    get_cv(trainer)