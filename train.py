import yaml
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from text_utils import combine_title_content  # Helper for text processing


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_wlaes(examples, w_like, w_collect, w_comment):
    """Computes the Weighted Logarithmic Accumulated Engagement Score (WLAES)."""
    log_accum_like = np.log1p(np.array(examples["accum_like_num"], dtype=np.float32))
    log_accum_collect = np.log1p(
        np.array(examples["accum_collect_num"], dtype=np.float32)
    )
    log_accum_comment = np.log1p(
        np.array(examples["accum_comment_num"], dtype=np.float32)
    )

    wlaes = (
        w_like * log_accum_like
        + w_collect * log_accum_collect
        + w_comment * log_accum_comment
    )
    examples["labels"] = wlaes.tolist()  # Trainer expects 'labels'
    return examples


def preprocess_data(examples, tokenizer, max_length):
    """Tokenizes the combined title and content."""
    texts = []
    for title, content in zip(examples["note_title"], examples["note_content"]):
        texts.append(combine_title_content(title, content))

    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",  # Ensure all sequences have the same length for batching
        max_length=max_length,
        return_tensors="pt",  # Will be handled by DataCollator, but good for consistency
    )
    # The 'labels' are already added by compute_wlaes
    # Just ensure tokenized_inputs doesn't overwrite it if map is chained differently.
    # Here, we return a dict that will update the examples.
    return {
        "input_ids": tokenized_inputs["input_ids"].tolist(),
        "attention_mask": tokenized_inputs["attention_mask"].tolist(),
    }


def compute_metrics_for_regression(eval_pred):
    """Computes MSE, MAE, and R2 for regression."""
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {"mse": mse, "mae": mae, "r2": r2}


def main():
    config = load_config()

    # --- 1. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model_path"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Common practice for base models

    # --- 2. Load Dataset ---
    print(f"Loading dataset from path: {config['dataset_path']}")
    raw_dataset = load_dataset(
        config["dataset_path"],
        name=config.get("dataset_subset"),
        split=config.get("dataset_split"),
    )
    print(f"Raw dataset loaded with {len(raw_dataset)} samples.")

    # --- 3. Filter out video posts (note_type == 2) ---
    # Ensure note_type is float if it comes as float from dataset
    dataset_non_video = raw_dataset.filter(lambda x: float(x["note_type"]) == 1.0)
    print(f"Dataset after filtering video posts: {len(dataset_non_video)} samples.")

    if len(dataset_non_video) == 0:
        print("No non-video posts found. Exiting.")
        return

    # --- 4. Compute WLAES (target variable) ---
    wlaes_cfg = config["wlaes_weights"]
    dataset_with_labels = dataset_non_video.map(
        compute_wlaes,
        batched=True,
        fn_kwargs={
            "w_like": wlaes_cfg["like"],
            "w_collect": wlaes_cfg["collect"],
            "w_comment": wlaes_cfg["comment"],
        },
    )
    print("WLAES labels computed.")

    # --- 5. Preprocess and Tokenize Text ---
    tokenized_dataset = dataset_with_labels.map(
        preprocess_data,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config["max_length"]},
        remove_columns=[
            name
            for name in dataset_with_labels.column_names
            if name not in ["input_ids", "attention_mask", "labels"]
        ],
    )
    print("Dataset tokenized and cleaned.")
    tokenized_dataset.set_format("torch")

    # --- 6. Split dataset (if not already split) ---
    # Split the "train" from Qilin into train/eval for finetuning
    if (
        "validation" not in tokenized_dataset.column_names
    ):  # A bit of a hacky check, better to have explicit splits
        print("Splitting dataset into train and validation sets.")
        train_test_split = tokenized_dataset.train_test_split(
            test_size=config["training"]["val_size"], seed=config["training"]["seed"]
        )
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
    else:  # If your dataset loading already provides splits
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["validation"]  # or "test"
    print(
        f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}"
    )

    # --- 7. Load Base Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model_path"],
        num_labels=1,  # For regression
        trust_remote_code=True,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    print("Base model loaded.")

    # --- 8. Configure LoRA ---
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Using Sequence Classification head for regression
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("LoRA configured.")

    # --- 9. Training Arguments ---
    train_cfg = config["training"]
    training_args = TrainingArguments(
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        dataloader_pin_memory=True,
        output_dir=config["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),  # Ensure float
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,  # Important for getting the best model
        metric_for_best_model="mse",  # or "mae", lower is better
        greater_is_better=False,  # For MSE/MAE
        fp16=train_cfg["fp16"],
        seed=train_cfg["seed"],
        report_to=train_cfg.get("report_to", "tensorboard"),  # Add default
        # max_grad_norm=train_cfg.get('max_grad_norm', 1.0) # Optional
    )
    print("Training arguments set.")

    # --- 10. Initialize Trainer ---
    # Data collator will handle dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_regression,
    )
    print("Trainer initialized.")

    # --- 11. Train ---
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- 12. Save the fine-tuned model (LoRA adapters) and tokenizer ---
    # The trainer saves checkpoints, and if load_best_model_at_end=True,
    # the best model (adapters) is loaded. We can save it explicitly.
    trainer.save_model(config["output_dir"])  # Saves adapter & tokenizer
    # model.save_pretrained(config['output_dir']) # Saves adapter
    # tokenizer.save_pretrained(config['output_dir'])
    print(f"Fine-tuned LoRA adapters and tokenizer saved to {config['output_dir']}")

    # --- 13. Evaluate the best model on the eval set ---
    print("Evaluating the best model on the evaluation set...")
    eval_results = trainer.evaluate(eval_dataset)
    print("Evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
