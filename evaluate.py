import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import PeftModel
from train import (
    compute_wlaes,
    preprocess_data,
    compute_metrics_for_regression,
    load_config,
)


# disable parallelism for tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    config = load_config()

    # --- 1. Load Tokenizer ---
    # Important: Load tokenizer from the fine-tuned model's directory if it was saved there,
    # or from base model path if it wasn't specifically modified and saved with adapter.
    # Trainer.save_model saves the tokenizer.
    tokenizer_path = config.get("trained_lora_path", config["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:  # Should be set if saved correctly
        print("Warning: pad_token not set in loaded tokenizer. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load Dataset (e.g., a test split or the same eval split) ---
    print(f"Loading dataset from path: {config['dataset_path']} for evaluation")
    raw_dataset_eval = load_dataset(
        config["dataset_path"],
        name=config.get("dataset_subset"),
        split=config.get("dataset_split"),
    )  # Or your test split
    dataset_non_video_eval = raw_dataset_eval.filter(
        lambda x: float(x["note_type"]) == 1.0
    )

    if len(dataset_non_video_eval) == 0:
        print("No non-video posts found for evaluation. Exiting.")
        return

    wlaes_cfg = config["wlaes_weights"]
    dataset_with_labels_eval = dataset_non_video_eval.map(
        compute_wlaes,
        batched=True,
        fn_kwargs={
            "w_like": wlaes_cfg["like"],
            "w_collect": wlaes_cfg["collect"],
            "w_comment": wlaes_cfg["comment"],
        },
    )
    tokenized_dataset_eval = dataset_with_labels_eval.map(
        preprocess_data,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config["max_length"]},
        remove_columns=[
            name
            for name in dataset_with_labels_eval.column_names
            if name not in ["input_ids", "attention_mask", "labels"]
        ],
    )
    tokenized_dataset_eval.set_format("torch")
    # Here, we'll use the same train_test_split logic to get a consistent eval set with train.py.
    # In practice, you'd have a separate, unseen test set.
    temp_splits = tokenized_dataset_eval.train_test_split(
        test_size=config["training"]["val_size"], seed=config["training"]["seed"]
    )
    final_eval_dataset = temp_splits["test"]  # This is just to get a dataset portion
    print(f"Evaluation dataset size: {len(final_eval_dataset)}")

    # --- 3. Load Base Model and LoRA Adapters ---
    # Load the base model first
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model_path"], num_labels=1, trust_remote_code=True
    )
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load the PeftModel (LoRA adapters) on top of the base model
    model = PeftModel.from_pretrained(base_model, config["trained_lora_path"])
    model = (
        model.merge_and_unload()
    )  # Optional: merge for faster inference if not training further
    model.eval()  # Set to evaluation mode
    print(f"Trained LoRA model loaded from {config['trained_lora_path']}")

    # --- 4. Evaluation Arguments (minimal) ---
    eval_cfg = config.get("evaluation", {})
    eval_args = TrainingArguments(
        output_dir=f"{config['output_dir']}/eval_run",  # Temporary output for evaluation
        per_device_eval_batch_size=eval_cfg.get("per_device_eval_batch_size", 16),
        fp16=config["training"]["fp16"],
        report_to="none",  # No need to report during standalone evaluation
    )

    # --- 5. Initialize Trainer for Evaluation ---
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    evaluator = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=final_eval_dataset,  # Use your prepared test/eval dataset
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_regression,
    )
    print("Evaluator initialized.")

    # --- 6. Evaluate ---
    print("Starting evaluation...")
    results = evaluator.evaluate()
    print("Evaluation results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
