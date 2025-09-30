A tokeinizer is used to convert the text into a sequence of numbers. These are the indiviusal tokens that the model will train on. The transformer architecture is used ti create language model. 
How to?
    1)Data Extraction
    2)Combine the extracted text
    3)BPE alogorithm
    4)Text encoding 

After running the BPE alogorithm we will have a vocablury, each word or part of word will have a unique id then we will encode the text using these id's 

    5)Create the model
    6)Data splitting & model training
    The 6 step will be use to check how well the model is learning,
    After training we will have a base model, This model has learned to simulate the training data.
    It can only do next token prediction.
    Not useful yet, To make it useful we will follow the next step.
    7)Preapare the fine-tuning dataset
    We will create a fine tuning dataset, after this the model will be able t mimic the person we chose in the dataset. The model is now useful. Finding the data is a changelling dataset.

1) Extracting the data is manual so it will take time.

Text encoding:
    1) Methods for text encoding: Character level, World, level, BPE
    2) Input example: Text encoding is an important step that affects later stages in pipeline.
    3) You should find a compromise between the vocubulary size and the sequence length.
    if vocablury length increases then sequence length decreases and vice verca.


|--------------------------------------------------------------|
|Tokeinization|Vocubulary size|Example Tokens| Sequence length |
|--------------------------------------------------------------|
|Character Level| ~100-500  |  "T","e","x","t"  |     78       |
|--------------------------------------------------------------|
|      BPE      |~10,000-50,000|"Text","enco","ding"|   14     |
|--------------------------------------------------------------|
|World level    |~50,000+      |"Text","encoding","is"|  13    |
|--------------------------------------------------------------|

We are gonna be using BPE bcz the vocubulary size and sequence length is balanced and in the Character level the vocubulary size is low but the sequence length is high, and in the word level the vocablury size is computationally expensive but the sequence length is low and the BPE is balance so we are gonna use it.

And the advantage of BPE is that we can control the size of the vocubulary based on our hardware and data we work with

Text Encoding - BPE:
    BPE (Byte pair Encoding)
    It is an algorithm designed to compress the text into a small sequence

    aaabdaaabac = [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99] = 
    (97, 97):4
    (97, 98):2 ... = [666, 97, 98, 100,
                      666, 97, 98, 97,
                      99]

    this step is called a merge step and this is a merge step 1.
    and then we will take this list and compute the same thing 
    [666, 97, 98, 100,
     666, 97, 98, 97,  =      (666, 97):2     =  [777, 98, 100, 777, 98, 
     99]                      (97, 98):2 ...     97, 99]

     #merge2
     ...

     we will continue to search until the maximum number of occurences and we will replace it with a new token

     the bps allow us to perform as many times we wanna merge it, so hence we can control the size of the vocablury.


The transformer model:
    1)The transformer architecture is used to train language models.
    2)It is divided into 2 parts: The encode and the decode
    3)For tasks like machine translation you use both parts. The encoder is used to encode the sentences in the 1st language and the decoder tries to generate the translation of that text based on the information give to it by the encoder, but if you want to train an auto aggresive model to learn from that data and generate text that is similar to the data distribution then just use the decoder.
    4)The decoder is composed of these components:
        1)Token Embedding (Represent token with a vector)
        |--------------------------------------|
        |Embedding dimensions|0 |1 |2| ...| n-1|
        |--------------------------------------|
        |hel          |0.21|-2.1|0.62|...|-0.33|
        |--------------------------------------|
        |imad         |-2.81|0.1|0.23|...|0.55 |
        |--------------------------------------|
        |you          |0.71|0.11|1.77|...|-0.23|
        |--------------------------------------|
        2)Positional encoding(Preserves the token order)
        Hi, How are you doing?
        if we break this sentece into tokens - each tokens gets a value that is its position , for eg - hi is the 1st token so it gets - 0, then how - 1, then are -2, then you -3, then doing -4 but instead of using a number we use a vectore to store more information.
        3)Self attention(Keeps the track of the relation between tokens)
        4)Residual connections(Helps the gradient flow easily through the network)
        5)Layer normalizations(Normalizes the input)
        6)MLP Layers(Expand the model's capacity to learn)
        7)Block
        8)Final layers(To get predictions)
        