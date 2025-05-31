import os
import json
import time
from openai import OpenAI


def get_api_response(
    endpoint,
    api_key=None,
    messages=[{"role": "user", "content": "Hello!"}],
    model="openai/gpt-4.1-nano",
    temperature=0.2,
    top_p=1.0,
    **kwargs,
):
    """
    Get response from OpenAI API.
    Args:
        endpoint (str): API endpoint URL.
        api_key (str): API key for authentication.
        messages (list): List of messages to send to the model.
        model (str): Model name to use.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        **kwargs: Additional parameters for the API request.
    Returns:
        str: The response text from the API.
    """
    if api_key is None:
        api_key = "xxxxxx"
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, top_p=top_p, **kwargs
    )

    return response.choices[0].message.content


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
    prompt = "我希望你扮演一位社交媒体分析师。我将提供给你 4 个小红书的笔记，包含笔记的标题和内容。请你根据笔记的标题和内容，对它们的热度进行排序。\n\n"
    for i, (title, content) in enumerate(zip(note_title, note_content)):
        prompt += f"--- Note {i + 1} ---\n"
        prompt += f"Title:\n{title}\nContent:\n{content}\n"
        prompt += f"--- End of Note {i + 1} ---\n\n"
    prompt += "\n"
    prompt += "请你为这 4 个笔记的热度进行排序，用降序排列，从最热到最冷。请你直接返回一个排序结果，例如 `2, 1, 4, 3` 代表第二个笔记最热，第四个笔记最冷。\n\n"
    prompt += "请注意，笔记的热度可能受到多个因素的影响，包括但不限于标题的吸引力、内容的质量、话题的流行程度等。请综合考虑这些因素进行排序。\n\n"
    prompt += "你的回答中只需要包含数字、逗号和空格，不需要其他文字或解释。请直接返回类似 `3, 1, 4, 2` 的排序结果，不要添加任何解释或其他内容。\n\n"
    return prompt


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
            "content": "你是一位社交媒体分析师，你的任务是根据笔记的标题和内容，对它们的热度进行排序。",
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
                    )
                    # Parse the response as a list of integers
                    try:
                        sorted_indices = list(map(int, response.strip().split(",")))
                        if len(sorted_indices) != 4:
                            raise ValueError(
                                f"Expected 4 indices, got {len(sorted_indices)}"
                            )
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
                    f"Author: {data['author_id']} ({data['nickname']}), Response: {response.strip()}"
                )
                result.append(data)
                time.sleep(sleep_time)

    # Save the results back to the JSONL file
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    # Set your API endpoint, model, and token here
    endpoint = "https://models.github.ai/inference"
    model = "openai/gpt-4.1-nano"
    token = os.environ["GITHUB_TOKEN"]  # Ensure you set this environment variable

    main(
        jsonl_file="results/predictions.jsonl",
        backup_file="results/predictions_backup.jsonl",
        endpoint=endpoint,
        model=model,
        token=token,
        max_retries=3,
        sleep_time=3,
    )
    print("API predictions completed.")
