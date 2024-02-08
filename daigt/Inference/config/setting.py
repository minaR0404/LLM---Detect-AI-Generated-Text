
from utils.model import *


###---------------------------------------------------------------------------------
### config
###---------------------------------------------------------------------------------
class CFG:
    LOWERCASE = False
    VOCAB_SIZE = 300000

    first_model = [
        SGD_model,
        bayes_model,
        LGBM_model,
        Cat_model,
        RF_model,
        LR_model,
    ]

    second_model = [
        ("lgb", LGBM_model),
        ("lr", LR_model),
    ]
