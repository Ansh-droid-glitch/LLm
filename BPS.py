import sys
import os
import io
from minbpe import BasicTokenizer

# Force UTF-8 output in Windows console (optional, helps with print)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure output folder exists
os.makedirs("output/tokenizer", exist_ok=True)

# Load the sequence
with open("output/cleaned.txt", "r", encoding="utf-8") as f:
    text_sequence = f.read()

print(f"Length of text sequence: {len(text_sequence)}")

# Initialize BPE tokenizer
tokeinizer = BasicTokenizer()

# Train tokenizer
tokeinizer.train(text_sequence, vocab_size=1024)

# Add special tokens
vocab = tokeinizer.vocab
max_vocab_id = list(vocab.keys())[-1]
tokeinizer.special_tokens = {
    max_vocab_id + 1: "⬅️startoftext➡️",
    max_vocab_id + 2: "⬅️seperator➡️",
    max_vocab_id + 3: "⬅️endoftext➡️",
    max_vocab_id + 4: "⬅️unk➡️",
}

# Encode example (optional)
encoded_len = len(tokeinizer.encode(text_sequence))
print(f"Length of encoded sequence: {encoded_len}")

# Save the tokenizer
# PATCH: ensure minbpe/base.py save() uses encoding="utf-8"
model_prefix = "output/tokenizer/my_tokenizer"
tokeinizer.save(file_prefix=model_prefix)

print(f"Tokenizer saved successfully to {model_prefix}.model")
