import re
import os
import yaml
import torch
import traceback
from pathlib import Path
from collections import deque
import pandas as pd
import gradio as gr
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --- Configuration Loading ---
CONFIG_PATH = "config.yaml"
MAX_HISTORY_LENGTH = 5
prediction_history = deque(
    maxlen=MAX_HISTORY_LENGTH
)  # Stores (title, content_summary, score) dicts

# Global model and tokenizer variables
config = None
device = None
tokenizer = None
model = None
MAX_LENGTH = None


def load_config_from_yaml(config_path=CONFIG_PATH):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse configuration file {config_path}: {e}")
        return None
    except Exception as e:
        print(
            f"Unknown error occurred while loading configuration file {config_path}: {e}"
        )
        return None


# --- Text Utilities (copied from text_utils.py and adapted) ---
def clean_text(text):
    """Cleans text by removing specific unicode characters and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    # Original project's clean_text doesn't remove emojis/topics by default, maintaining that.
    # text = re.sub(r"\[.*?R\]", "", text) # Example: [大笑R]
    # text = re.sub(r"#.*?\[话题\]#", "", text) # Example: #测试[话题]#
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_title_content(title, content):
    """Combines cleaned title and content into a single string for model input."""
    cleaned_title = clean_text(title)
    cleaned_content = clean_text(content)
    return f"帖子标题：{cleaned_title}\n\n帖子内容：{cleaned_content}"


# --- Model Initialization ---
def initialize_model_and_tokenizer():
    """Loads the tokenizer and model based on the configuration."""
    global config, device, tokenizer, model, MAX_LENGTH

    config = load_config_from_yaml()
    if config is None:
        print("Failed to load configuration. Using default settings.")
        trained_lora_path = "qwen3_wlaes_finetuned"
        base_model_path = "Qwen3-0.6B-Base"
        max_length = 768
    else:
        trained_lora_path = config.get("trained_lora_path")
        base_model_path = config.get("base_model_path")
        max_length = config.get("max_length", 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # --- 1. Load Tokenizer ---
        tokenizer_path = trained_lora_path
        if not os.path.isdir(tokenizer_path):
            print(
                f"Error: Tokenizer path '{tokenizer_path}' does not exist or is not a directory."
            )
            return

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer successfully loaded from {tokenizer_path}.")

        # --- 2. Load Base Model and LoRA Adapters ---
        if not os.path.isdir(base_model_path):
            print(
                f"Error: Base model path '{base_model_path}' does not exist or is not a directory."
            )
            return

        base_model_instance = AutoModelForSequenceClassification.from_pretrained(
            base_model_path, num_labels=1, trust_remote_code=True
        )
        if base_model_instance.config.pad_token_id is None:
            base_model_instance.config.pad_token_id = tokenizer.pad_token_id
        print(f"Base model successfully loaded from {base_model_path}.")

        # LoRA model path is typically the same as tokenizer_path after training
        peft_model_path = tokenizer_path
        model = PeftModel.from_pretrained(base_model_instance, peft_model_path)
        model = model.merge_and_unload()  # Merge LoRA weights for inference
        model.to(device)
        model.eval()
        print(
            f"Trained LoRA model successfully loaded and merged from {peft_model_path}."
        )

        MAX_LENGTH = max_length
        print(f"Maximum sequence length set to: {MAX_LENGTH}")

    except Exception as e:
        print(f"Error occurred during model initialization: {e}")
        traceback.print_exc()
        model = None  # Ensure model is None so predict function can check


# --- Core Prediction Logic ---
def internal_predict_popularity(title_str: str, content_str: str) -> float:
    """Internal function to predict WLAES score."""
    if model is None or tokenizer is None or device is None or MAX_LENGTH is None:
        raise gr.Error("Model or tokenizer not initialized. Please check the setup.")

    model.eval()  # Ensure model is in evaluation mode
    processed_text = combine_title_content(title_str, content_str)

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_wlaes = logits.item()
    return predicted_wlaes


# --- Gradio Interaction Functions ---
def format_history_for_dataframe(history_deque: deque):
    """Formats the prediction history into a Pandas DataFrame for Gradio."""
    if not history_deque:
        # Return an empty DataFrame with headers if history is empty
        return pd.DataFrame(columns=["标题", "内容", "预测 WLAES"])

    # Convert deque of dicts to list of lists for DataFrame creation
    # or directly to a list of dicts if pandas handles it well
    history_list = []
    for item in history_deque:  # deque iterates from left (newest) to right (oldest)
        history_list.append(
            {
                "标题": str(item["标题"]),
                "内容": str(item["内容"]),
                "预测 WLAES": str(item["预测WLAES"]),
            }
        )
    return pd.DataFrame(history_list)


def predict_and_update_history(title: str, content: str):
    """Gradio interface function for prediction and history update."""
    global prediction_history

    if not title and not content:  # Handle empty inputs
        return None, format_history_for_dataframe(prediction_history)

    try:
        predicted_score = internal_predict_popularity(title, content)

        # Add to history (deque automatically handles max length)
        content_summary = content[:100] + "..." if len(content) > 100 else content
        prediction_history.appendleft(
            {
                "标题": title,
                "内容": content_summary,
                "预测WLAES": f"{predicted_score:.4f}",
            }
        )

        return f"{predicted_score:.4f}", format_history_for_dataframe(
            prediction_history
        )
    except gr.Error as ge:  # Catch Gradio-specific errors to re-raise
        raise ge
    except Exception as e:
        print(f"Error occurred during prediction: {e}")
        traceback.print_exc()
        raise gr.Error(f"Prediction failed: {str(e)}")


# --- Gradio Interface Definition ---
def create_gradio_interface(assets_dir: os.PathLike = "static"):
    """Creates and returns the Gradio Blocks interface."""
    examples = [
        [
            "谁家大学生这么聪明…",
            "大学生不语 只是一味剪开包装继续吃面[吧唧R]\n（神奇的是！面居然软了哈哈哈哈哈哈#大学生精神状态很稳定呀[话题]# #当代读书人亲测[话题]# #大学生稳定精神状态时刻[话题]#",
        ],
        [
            "谁来解释下🙋",
            "为啥\n飞机都是烧油的\n没有充电的飞机？\n#不懂就问有问必答[话题]# #每日分享[话题]# #谁能解释这种现象[话题]# #这是为什么呢[话题]# #多少有点离谱[话题]# #怎么会这样[话题]# #我真的不理解[话题]# #有趣日常[话题]# #飞机[话题]# #航空[话题]#",
        ],
    ]

    rednoob_banner = os.path.join(assets_dir, "rednoob-banner.svg")

    with gr.Blocks(theme=gr.themes.Soft()) as app:
        # Banner and description
        gr.HTML(
            f"""
            <div align="center">
                <img src="/gradio_api/file={rednoob_banner}" width="300" alt="banner"/>
                <br />
                <h5>输入笔记的标题和内容，模型将预测其受欢迎程度分数 (WLAES)。</h5>
            </div>
        """
        )

        # Input and Output Components
        with gr.Row():
            with gr.Column(scale=2):
                title_input = gr.Textbox(
                    label="笔记标题",
                    placeholder="请输入笔记标题...",
                    elem_id="title_input",
                )
                content_input = gr.Textbox(
                    label="笔记内容",
                    placeholder="请输入笔记内容...",
                    lines=5,
                    max_lines=10,
                    elem_id="content_input",
                )
                predict_button = gr.Button(
                    "Run Prediction", variant="primary", elem_id="predict_button"
                )

            with gr.Column(scale=1):
                score_output = gr.Textbox(
                    label="预测 WLAES 分数", interactive=False, elem_id="score_output"
                )

        gr.Examples(
            examples=examples,
            inputs=[title_input, content_input],
            label="示例输入",
            elem_id="examples",
        )

        # History display as a DataFrame
        gr.Markdown("### 最近预测记录")
        history_display = gr.DataFrame(
            value=format_history_for_dataframe(prediction_history),
            headers=["标题", "内容", "预测 WLAES"],
            interactive=False,
            elem_id="history_display",
            max_chars=64,
        )  # Initial history display

        predict_button.click(
            fn=predict_and_update_history,
            inputs=[title_input, content_input],
            outputs=[score_output, history_display],
        )
    return app


# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing model, please wait...")
    initialize_model_and_tokenizer()

    if model is None or tokenizer is None:
        print(
            "Critical components (model or tokenizer) failed to initialize. Gradio app may not work properly."
        )

    print("Model initialization complete. Starting Gradio app...")

    current_dir = Path(__file__).parent
    assets_dir = os.path.join(current_dir, "static")

    gradio_app = create_gradio_interface(assets_dir=assets_dir)
    gradio_app.launch(allowed_paths=[assets_dir])
