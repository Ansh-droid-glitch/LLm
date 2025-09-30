LLm: Language Model Training Pipeline

This repository demonstrates how to train a language model using Byte-Pair Encoding (BPE) and the transformer architecture.

A tokenizer is used to convert text into a sequence of numbers. Each number represents an individual token that the model trains on. The transformer architecture is used to build the language model.

How to Use

Data Extraction
Extract the text data you want to train on. This step is manual and may take some time.

Combine Extracted Text
Merge all the extracted text into a single file.

BPE Algorithm
Apply Byte-Pair Encoding to create a vocabulary of subword tokens.

Text Encoding
After BPE, each token in the vocabulary has a unique ID. The text is then encoded using these IDs.

Create the Model
Define the transformer model architecture.

Data Splitting & Model Training
Split your dataset into training and validation sets and train the model. After training, you will have a base model that can predict the next token in a sequence. This is the first step toward building a usable model.

Prepare Fine-Tuning Dataset
Create a fine-tuning dataset so the model can mimic a specific person or style. Fine-tuning makes the model more practical and personalized.

Text Encoding
Methods

Character-level: ~100–500 tokens

BPE (Byte-Pair Encoding): ~10,000–50,000 tokens

Word-level: 50,000+ tokens

Tokenization	Vocabulary Size	Example Tokens	Sequence Length
Character Level	~100–500	"T","e","x","t"	78
BPE	~10,000–50,000	"Text","enco","ding"	14
Word Level	50,000+	"Text","encoding","is"	13

We use BPE because it balances vocabulary size and sequence length:

Character-level: small vocabulary, long sequences

Word-level: large vocabulary, short sequences (computationally expensive)

BPE: balanced, controllable vocabulary size

BPE (Byte-Pair Encoding)

BPE is an algorithm to compress text into smaller sequences by merging frequent token pairs.

Example:

Text: aaabdaaabac
Initial IDs: [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]

Merge steps:
1. (97,97) -> 666
2. (666,97) -> 777
...


Each merge reduces sequence length.

You can control vocabulary size by choosing the number of merge steps.

Transformer Model

Architecture: Transformers are used for language modeling.

Encoder vs Decoder: For auto-regressive models (predicting next token), only the decoder is used.

Decoder Components:

Token Embedding: Represents each token as a vector

Positional Encoding: Preserves token order

Self-Attention: Tracks relationships between tokens

Residual Connections: Helps gradient flow

Layer Normalization: Normalizes inputs

MLP Layers: Expands model capacity

Blocks: Stacks of layers

Final Layers: Outputs token predictions

Example: Token Embedding
Token	Dim 0	Dim 1	Dim 2	…	Dim n-1
hel	0.21	-2.1	0.62	…	-0.33
imad	-2.81	0.1	0.23	…	0.55
you	0.71	0.11	1.77	…	-0.23

Positional encoding adds order information to token embeddings.