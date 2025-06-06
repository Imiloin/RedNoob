import os
import re
import json
import time
import openai


def get_api_response(
    endpoint,
    api_key=None,
    messages=[{"role": "user", "content": "Hello!"}],
    model="openai/gpt-4.1-nano",
    max_retries=3,
    **kwargs,
):
    """
    Get response from OpenAI API with rate limit handling.
    Args:
        endpoint (str): API endpoint URL.
        api_key (str): API key for authentication.
        messages (list): List of messages to send to the model.
        model (str): Model name to use.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        max_retries (int): Maximum number of retries for rate limit errors.
        **kwargs: Additional parameters for the API request.
    Returns:
        str: The response text from the API.
    """
    if api_key is None:
        api_key = "xxxxxx"
    client = openai.OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            print(f"Rate limit error: {e}")
            if attempt < max_retries - 1:
                # Extract wait time from error message if available
                error_msg = str(e)
                if "wait" in error_msg.lower():
                    # Try to extract the wait time from the error message
                    wait_match = re.search(r"wait (\d+) seconds", error_msg)
                    if wait_match:
                        wait_time = int(wait_match.group(1)) + 5  # Add 5 seconds buffer
                    else:
                        wait_time = 65  # Default to 65 seconds if can't parse
                else:
                    wait_time = 65  # Default wait time

                print(
                    f"Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
            else:
                print("Max retries reached for rate limit. Raising error.")
                raise


def construct_prompt(
    note_title: list[str],
    note_content: list[str],
):
    """
    Construct a prompt for the model based on note titles and contents.
    Args:
        note_title (list[str]): List of note titles.
        note_content (list[str]): List of note contents.
    Returns:
        str: The constructed prompt.
    """
    assert (
        len(note_title) == 4 and len(note_content) == 4
    ), "Both lists must contain exactly 4 items."
    # Task context
    prompt = "我希望你扮演一位社交媒体分析师。"
    # Detailed task description and rules
    prompt += "\n\n我将提供给你 4 个小红书的笔记，包含笔记的标题和内容。请你根据笔记的标题和内容，对它们的热度进行排序。排序结果**用降序表示**，从最热到最冷。"
    # Examples
    prompt += "\n\n例如 `2, 1, 4, 3` 代表第二个笔记最热，第三个笔记最冷。"
    # Input data to process
    prompt += "\n\n"
    for i, (title, content) in enumerate(zip(note_title, note_content)):
        prompt += f"--- Note {i + 1} ---\n"
        prompt += f"<title>\n  {title}\n</title>\n<content>\n  {content}\n</content>\n"
        prompt += f"--- End of Note {i + 1} ---\n\n"
    prompt += "\n"
    # Precognition (thinking step by step)
    prompt += "\n\n请注意，笔记的热度可能受到多个因素的影响，包括但不限于标题的吸引力、内容的质量、话题的流行程度等。请综合考虑这些因素进行排序。"
    # Output formatting
    prompt += "\n\n"
    prompt += """首先，请你简要分析帖子的内容，将你的想法写在 <analysis> 标签中。然后，按照热度对笔记进行从高到低排序，并将排序结果写在 <ranking> 标签中，排序结果中只能包含数字、逗号和空格，不需要其他文字或解释。

你的回答应当有下面的格式：

```response
<analysis>
  ...在这里写下你的分析...
</analysis>

<ranking>
  3, 1, 4, 2
</ranking>
```
"""
    return prompt


def extract_ranking(response: str):
    """
    Extract the ranking from the model's response.
    Args:
        response (str): The response from the model.
    Returns:
        list[int]: The extracted ranking as a list of integers.
    """
    # Use regex to find the ranking in the <ranking> tag
    match = re.search(r"<ranking>\s*([\d,\s]+)\s*</ranking>", response, re.DOTALL)
    if match:
        ranking_str = match.group(1).strip()
        # Split by comma and convert to integers
        return [int(x.strip()) for x in ranking_str.split(",")]
    else:
        raise ValueError("No valid ranking found in the response.")


def construct_messages(
    note_title: list[str],
    note_content: list[str],
):
    """
    Construct messages for the model based on note titles and contents.
    Args:
        note_title (list[str]): List of note titles.
        note_content (list[str]): List of note contents.
    Returns:
        list: The constructed messages.
    """
    prompt = construct_prompt(note_title, note_content)
    return [
        {
            "role": "system",
            "content": "你是一位社交媒体分析师，专注于用户内容分析。",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def main(
    jsonl_file="results/predictions.jsonl",
    backup_file=None,
    endpoint="https://models.github.ai/inference",
    model="openai/gpt-4.1-nano",
    token=None,
    max_retries=3,
    sleep_time=3,
    extra_api_params: dict = None,
):
    # Copy the original file to a backup if specified
    if backup_file is not None:
        if os.path.exists(backup_file):
            print(f"Backup file {backup_file} already exists. Overwrite it.")
        with open(jsonl_file, "r", encoding="utf-8") as original, open(
            backup_file, "w", encoding="utf-8"
        ) as backup:
            for line in original:
                backup.write(line)

    # Create an empty list to store results
    result = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Check if the rankings for this model already exist
                if model in data.get("predictions", {}):
                    print(
                        f"Skipping author {data['author_id']} ({data['nickname']}) as predictions for model '{model}' already exist."
                    )
                    result.append(data)
                    continue
                # Construct messages for the API call
                messages = construct_messages(
                    data["note_titles"],
                    data["note_contents"],
                )
                # Get the API response
                for attempt in range(max_retries):
                    response = get_api_response(
                        endpoint=endpoint,
                        messages=messages,
                        model=model,
                        api_key=token,
                        **(extra_api_params or {}),  # add extra parameters
                    )
                    # Parse the response as a list of integers
                    try:
                        sorted_indices = extract_ranking(response)
                        # Ensure we have exactly 4 indices
                        if len(set(sorted_indices)) != 4:
                            raise ValueError("Ranking must contain 4 unique indices.")
                        else:
                            data["predictions"][model] = sorted_indices
                    except ValueError as e:
                        print(
                            f"Error parsing response: {e}. Response: {response.strip()}"
                        )
                        if attempt < max_retries - 1:
                            print("Retrying...")
                        else:
                            print("Max retries reached. Skipping this entry.")
                            time.sleep(sleep_time)
                            continue
                    break
                # Save the updated data to the result list
                print(
                    f"Author: {data['author_id']} ({data['nickname']}), Response: {sorted_indices}"
                )
                result.append(data)
                time.sleep(sleep_time)

    # Save the results back to the JSONL file
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    # Set your API endpoint, model, and token here
    endpoint = "https://api-inference.modelscope.cn/v1/"
    model = "Qwen/Qwen3-235B-A22B"
    token = os.environ["MODELSCOPE_SDK_TOKEN"]
    # Extra parameters for the API call
    extra_api_params = {
        "temperature": 0.7,
        "top_p": 0.8,
        # "reasoning_effort": "low",  # For Gemini model
        "extra_body": {"enable_thinking": False},  # For Qwen3 model
    }

    main(
        jsonl_file="results/predictions.jsonl",
        backup_file="results/predictions_backup.jsonl",
        endpoint=endpoint,
        model=model,
        token=token,
        max_retries=3,
        sleep_time=3,
        extra_api_params=extra_api_params,
    )
    print("API predictions completed.")
