import sys
import os
import io
from minbpe import BasicTokenizer

# Force UTF-8 output in Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure output folder exists
os.makedirs("output/tokenizer", exist_ok=True)

# Load the sequence
with open("output/cleaned.txt", "r", encoding="utf-8") as f:
    text_sequence = f.read()

print(f"Length of text sequence: {len(text_sequence)}")

# Initialize BPE tokenizer
tokenizer = BasicTokenizer()

# Train tokenizer
tokenizer.train(text_sequence, vocab_size=1024)

# Add special tokens (string → int mapping ✅)
vocab = tokenizer.vocab
max_vocab_id = list(vocab.keys())[-1]
tokenizer.special_tokens = {
    "startoftext": max_vocab_id + 1,
    "seperator": max_vocab_id + 2,
    "endoftext": max_vocab_id + 3,
    "unk": max_vocab_id + 4,
}

# Encode example (optional)
encoded_len = len(tokenizer.encode(text_sequence))
print(f"Length of encoded sequence: {encoded_len}")

# Save the tokenizer
model_prefix = "output/tokenizer/my_tokenizer"
tokenizer.save(file_prefix=model_prefix)

print(f"Tokenizer saved successfully to {model_prefix}.model")
