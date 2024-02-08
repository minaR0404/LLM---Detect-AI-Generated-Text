
from transformers import TrainingArguments, EarlyStoppingCallback


###---------------------------------------------------------------------------------
### config
###---------------------------------------------------------------------------------
class CFG:
    model_checkpoint = "./data/distilroberta-base"

    num_labels = 2
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    num_train_epochs = 10.0

    metric_name = "roc_auc"
    model_name = "distilroberta"  # "deberta-large"
    batch_size = 48

    args = TrainingArguments(
        f"{model_name}-finetuned",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        lr_scheduler_type = "cosine",
        
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        report_to='none',
        save_total_limit=2
    )