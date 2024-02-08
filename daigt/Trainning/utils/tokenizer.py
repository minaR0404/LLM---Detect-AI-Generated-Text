
from transformers import AutoTokenizer

from config.setting import CFG


###---------------------------------------------------------------------------------
### tokenizer
###---------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(CFG.model_checkpoint)