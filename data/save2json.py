import sqlite3
import json
import re
from collections import defaultdict


def parse_chinese_number(text):
    """Convert Chinese numerical expressions to numbers"""
    if not text or text == "":
        return 0

    # If it's already a number, return it directly
    try:
        return int(text)
    except (ValueError, TypeError):
        pass

    # Convert to string for processing
    text = str(text).strip()

    # Handle formats like "2.1万", "1.5千", "10万+"
    # Match pattern: number + unit
    pattern = r"(\d+\.?\d*)\s*([万千百十亿]+)"
    match = re.search(pattern, text)

    if match:
        number = float(match.group(1))
        unit = match.group(2)

        # Convert based on the unit
        if "万" in unit:
            return int(number * 10000)
        elif "千" in unit:
            return int(number * 1000)
        elif "百" in unit:
            return int(number * 100)
        elif "十" in unit:
            return int(number * 10)
        elif "亿" in unit:
            return int(number * 100000000)

    # Handle pure numbers with extra characters, e.g., "123+"
    number_match = re.search(r"(\d+)", text)
    if number_match:
        return int(number_match.group(1))

    # Return 0 if parsing fails
    print(f"Failed to parse number from: {text}")
    return 0


# Connect to the database
conn = sqlite3.connect("ExploreData.db")
cursor = conn.cursor()

# Fetch all data
cursor.execute("SELECT * FROM explore_data")
rows = cursor.fetchall()

# Get column names
cursor.execute("PRAGMA table_info(explore_data)")
columns = [col[1] for col in cursor.fetchall()]

# Group by author ID
table = defaultdict(dict)

for row in rows:
    data = dict(zip(columns, row))
    author_id = data["作者ID"]
    
    # Check if author_id exists in the table, if not, initialize it
    if author_id not in table:
        nickname = data["作者昵称"]
        table[author_id] = {"nickname": nickname, "notes": []}

    note = {
        "id": data["作品ID"],
        "title": (
            data["作品标题"] if data["作品标题"] else data["作品描述"].split("\n")[0]
        ),
        "content": data["作品描述"] if data["作品描述"] else data["作品标题"],
        "like_num": parse_chinese_number(data["点赞数量"]),
        "collect_num": parse_chinese_number(data["收藏数量"]),
        "comment_num": parse_chinese_number(data["评论数量"]),
        "share_num": parse_chinese_number(data["分享数量"]),
    }

    table[author_id]["notes"].append(note)

# Check each author has 4 notes
for author_id, author_data in table.items():
    if len(author_data["notes"]) != 4:
        print(
            f"\n------\nWarning: Author {author_id} ({author_data['nickname']}) has {len(author_data['notes'])} notes, skipping...\n------\n"
        )
        # del table[author_id]

# Save as JSON
with open("note_data.json", "w", encoding="utf-8") as f:
    json.dump(dict(table), f, ensure_ascii=False, indent=2)

print(f"Done! Data for {len(table)} authors has been processed.")

conn.close()
