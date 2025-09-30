import re

input_file = "chat.txt"
output_file = "cleaned.txt"

def clean_message(msg: str) -> str:
    # remove links
    msg = re.sub(r"http\S+", "", msg)
    # remove emojis / non-text symbols
    msg = re.sub(r"[^\w\s.,!?;:()'\-]", "", msg)
    # collapse extra spaces
    msg = re.sub(r"\s+", " ", msg)
    return msg.strip()

cleaned_lines = []
current_user = None

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Match user line: [timestamp] username
        match = re.match(r"\[(.*?)\]\s(.*)", line)
        if match and not line.endswith("{Attachments}") and not line.startswith("{"):
            _, user = match.groups()
            current_user = user.strip()
            continue

        # Otherwise it's a message line
        if current_user:
            msg = clean_message(line)

            # skip system-like logs
            if not msg:
                continue
            if "Started a call that lasted" in msg:
                continue
            if msg.lower().startswith("changed the channel"):
                continue
            if msg.lower().startswith("added") and "to the group" in msg.lower():
                continue

            cleaned_lines.append(f"{current_user}: {msg}")

with open(output_file, "w", encoding="utf-8") as f:
    for msg in cleaned_lines:
        f.write(msg + "\n")

print(f"✅ Done. Saved {len(cleaned_lines)} cleaned messages with authors to {output_file}")
