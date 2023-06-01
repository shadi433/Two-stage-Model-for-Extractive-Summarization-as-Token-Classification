from transformers import (RobertaTokenizerFast, RobertaForTokenClassification)
import numpy as np
from keras.layers import TimeDistributed
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Load the CNN/Daily Mail dataset
train_data = pd.read_csv("train.csv")
train_data = train_data[:5001]

# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=2)

# Tokenize articles
def tokenize_articles(data):
    stop_words = set(stopwords.words("english"))
    tokenized_articles = []

    for article in data["article"]:
        # Lowercase the text
        article = article.lower()
        # Tokenize and remove stop words
        tokens = word_tokenize(article)
        tokens = [word for word in tokens if word.lower() not in stop_words]
        tokens = [token for token in tokens if token not in string.punctuation]
        tokenized_articles.append(tokens)
  
    return tokenized_articles
print('start')
train_data["tokens"] = tokenize_articles(train_data)

# Preprocess dataset and create binary label sequences
def add_binary_labels(data):
    binary_label_sequences = []

    for _, row in data.iterrows():
        input_tokens = row["tokens"]
        summ = row["highlights"]
        summ = summ.lower()
        summary_tokens = word_tokenize(summ)
        binary_label_sequence = []
        for token in input_tokens:
            if token in summary_tokens:
                binary_label_sequence.append(1)
                summary_tokens.remove(token)
            else:
                binary_label_sequence.append(0)
        binary_label_sequences.append(binary_label_sequence)

    return binary_label_sequences

train_data["binary_labels"] = add_binary_labels(train_data)
train_tokens = train_data["tokens"]
train_labels = train_data["binary_labels"]
train_tokens = train_tokens.tolist()
train_labels = train_labels.tolist()

print(train_tokens[2])
print(train_labels[2])


print('start gitting embbidings')
embbidings = []
new_labels = []

for row, labels in zip(train_tokens, train_labels):
    # Calculate the number of chunks needed for the current row
    num_chunks = (len(row) + 249) // 250
    row_emb = []
    row_labels = []
    # Split tokens and labels into chunks
    for i in range(num_chunks):
        start_idx = i * 250
        end_idx = (i + 1) * 250
        chunk_tokens = row[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]

        tokenized_input = tokenizer(chunk_tokens, is_split_into_words=True, return_offsets_mapping=True, add_special_tokens= False,truncation=True, return_tensors="pt")
        new_chunk_labels = []
        current_orig_token = 0
        offset_mapping = tokenized_input.offset_mapping[0].tolist()
        for offset in offset_mapping:
            # If this is the first subword token and not a special token
            if offset[0] == 0 and (offset[1] != 0 and offset[1] != 1):
                new_chunk_labels.append(chunk_labels[current_orig_token])
                current_orig_token += 1
            else:
                new_chunk_labels.append(-1)  #-1 will be ignored

        row_labels.append(new_chunk_labels)
    
        tokenized_input.pop("offset_mapping")
        # print(tokenized_input)

        outputs = model(**tokenized_input, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        last_hidden_states = last_hidden_states[0]
        row_emb.append(last_hidden_states)
        # print(last_hidden_states[0])
    
    row_emb = [tensor.tolist() for tensor in row_emb]
    row_emb = [item for sublist in row_emb for item in sublist]
    embbidings.append(row_emb)

    row_labels = [item for sublist in row_labels for item in sublist]
    new_labels.append(row_labels)

print(len(embbidings))
print(len(new_labels))

np.save('Rob_embeddings.npy', embbidings)
np.save('Rob_new_labels.npy', new_labels)

