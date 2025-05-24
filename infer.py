import yaml
import torch
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


def main():
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

    # --- 3. Example Inference ---
    sample_title = "这是一个很棒的标题！"
    sample_content = (
        "这里是一些帖子的内容。我们来测试一下这个模型的效果如何。\n"
        "它应该能根据文本预测受欢迎程度。包含一些特殊字符 \u200c 和表情 [大笑R] 以及话题 #测试一下[话题]#。"
    )

    predicted_score = predict_popularity(
        sample_title, sample_content, model, tokenizer, device, config["max_length"]
    )
    print(f"\n--- Example Inference ---")
    print(f"Title: {sample_title}")
    print(f"Content: {sample_content[:100]}...")  # Print first 100 chars of content
    print(f"Predicted WLAES: {predicted_score:.4f}")

    sample_title_2 = "今天是甜妹！！！"
    sample_content_2 = "嘻嘻甜甜的\n\n#甜妹[话题]# #少女感[话题]# #拍照[话题]# #浅尝一下邻家女孩风[话题]# #满满少女感[话题]#"
    predicted_score_2 = predict_popularity(
        sample_title_2, sample_content_2, model, tokenizer, device, config["max_length"]
    )
    print(f"\n--- Example Inference 2 (from your sample) ---")
    print(f"Title: {sample_title_2}")
    print(f"Content: {sample_content_2[:100].strip()}...")
    print(f"Predicted WLAES: {predicted_score_2:.4f}")


if __name__ == "__main__":
    main()
