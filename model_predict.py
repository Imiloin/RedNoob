import json
import torch
import random
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from text_utils import combine_title_content
from train import load_config  # Re-use config loader


def predict_popularity(
    title: str, content: str, model, tokenizer, device, max_length: int
):
    """Predicts WLAES for a given title and content."""
    model.eval()  # Ensure model is in evaluation mode

    processed_text = combine_title_content(title, content)

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",  # Or False, and handle batching if predicting multiple
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_wlaes = logits.item()  # Assuming single prediction
    return predicted_wlaes


def save_predictions_to_jsonl(
    author_results, output_file="results/predictions.jsonl", model_name="lora_finetuned"
):
    """Save prediction results to JSONL format."""
    # make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    data = {}
    # Add new prediction results
    for author_id, result in author_results.items():
        data[author_id] = result

    # Write all results back to the file
    with open(output_file, "w", encoding="utf-8") as f:
        for author_data in data.values():
            f.write(json.dumps(author_data, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_file}")


def main(
    json_file: str = "data/f_data.json",
    output_file: str = "results/predictions.jsonl",
    model_name: str = "lora_finetuned",
):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Tokenizer ---
    tokenizer_path = config.get("trained_lora_path", config["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load Base Model and LoRA Adapters ---
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model_path"], num_labels=1, trust_remote_code=True
    )
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, config["trained_lora_path"])
    # For inference, it's often better to merge the LoRA weights into the base model
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    print(f"Trained LoRA model loaded from {config['trained_lora_path']} and merged.")

    # --- 3. Load JSON ---
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # --- 4. Inference and collect results ---
    author_results = {}

    for author_id, author_data in json_data.items():
        # make sure the author has exactly 4 notes
        if len(author_data["notes"]) != 4:
            print(
                f"\n------\nWarning: Author {author_id} ({author_data['nickname']}) has {len(author_data['notes'])} notes, skipping...\n------\n"
            )
            continue

        print(
            f"\n--- Inference for Author: {author_id} ({author_data['nickname']}) ---"
        )

        # Initialize a list to hold notes with their predicted scores
        notes_with_scores = []

        for i, note in enumerate(author_data["notes"]):
            predicted_score = predict_popularity(
                note["title"],
                note["content"],
                model,
                tokenizer,
                device,
                config["max_length"],
            )

            print(f"Title: {note['title']}")
            print(f"Content: {note['content'][:32]}...")
            print(f"Predicted WLAES: {predicted_score:.4f}")

            notes_with_scores.append(
                {
                    "original_index": i,
                    "note_id": note["id"],
                    "title": note["title"],
                    "content": note["content"],
                    "like_num": note["like_num"],
                    "collect_num": note["collect_num"],
                    "comment_num": note["comment_num"],
                    "share_num": note["share_num"],
                    "predicted_score": predicted_score,
                }
            )

        # shuffle the notes to simulate random order
        random.seed(42)  # set seed for reproducibility
        random.shuffle(notes_with_scores)

        # sort notes by predicted score in descending order
        sorted_by_score = sorted(
            notes_with_scores, key=lambda x: x["predicted_score"], reverse=True
        )

        # create a mapping of original indices to their rankings based on predicted scores
        score_rankings = {}
        for rank, note in enumerate(sorted_by_score, 1):
            # find the index of the note in the shuffled list
            shuffled_index = next(
                i
                for i, n in enumerate(notes_with_scores)
                if n["note_id"] == note["note_id"]
            )
            score_rankings[shuffled_index] = rank

        # construct the result record for this author as jsonl format
        result_record = {
            "author_id": author_id,
            "nickname": author_data["nickname"],
            "note_ids": [note["note_id"] for note in notes_with_scores],
            "note_titles": [note["title"] for note in notes_with_scores],
            "note_contents": [note["content"] for note in notes_with_scores],
            "like_nums": [note["like_num"] for note in notes_with_scores],
            "collect_nums": [note["collect_num"] for note in notes_with_scores],
            "comment_nums": [note["comment_num"] for note in notes_with_scores],
            "share_nums": [note["share_num"] for note in notes_with_scores],
            "predictions": {
                model_name: [
                    score_rankings[i] for i in range(4)
                ]  # rankings based on shuffled indices
            },
        }

        author_results[author_id] = result_record

        # print the ranking results
        print(f"\nRanking for {author_data['nickname']} (by {model_name}):")
        for i, ranking in enumerate(result_record["predictions"][model_name]):
            print(
                f"  Position {i+1}: {notes_with_scores[i]['title'][:8]}... -> Rank {ranking} (Score: {notes_with_scores[i]['predicted_score']:.4f})"
            )

    # --- 5. Save results to JSONL ---
    save_predictions_to_jsonl(author_results, output_file, model_name)


if __name__ == "__main__":
    json_file = "data/f_data.json"
    output_file = "results/predictions.jsonl"
    model_name = "lora_finetuned"
    main(json_file=json_file, output_file=output_file, model_name=model_name)
