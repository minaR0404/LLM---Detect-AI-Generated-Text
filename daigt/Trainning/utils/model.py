
from transformers import AutoModelForSequenceClassification

from config.setting import CFG


###---------------------------------------------------------------------------------
### model
###---------------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(CFG.model_checkpoint, num_labels=CFG.num_labels)