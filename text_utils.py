import re


def clean_text(text):
    """
    Cleans the input text by removing specific unicode characters,
    platform-specific emojis, and topic tags.
    """
    if not isinstance(text, str):
        return ""  # Or handle as an error

    # Remove zero-width non-joiner and other similar characters
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    # # Remove platform-specific emojis like [大笑R], [种草R]
    # text = re.sub(r"\[.*?R\]", "", text)
    # # Remove platform-specific topic tags like #补课[话题]#
    # text = re.sub(r"#.*?\[话题\]#", "", text)

    # Normalize whitespace (replace multiple spaces/newlines with a single one)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_title_content(title, content):
    """
    Combines title and content into a single string for model input.
    """
    cleaned_title = clean_text(title)
    cleaned_content = clean_text(content)
    # Using a clear separator, as Base models don't understand specific roles
    return f"帖子标题：{cleaned_title}\n\n帖子内容：{cleaned_content}"
