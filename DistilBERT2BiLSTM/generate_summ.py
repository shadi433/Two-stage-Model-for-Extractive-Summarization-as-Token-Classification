from transformers import (DistilBertTokenizerFast, DistilBertForTokenClassification)
from datasets import load_metric
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Load the CNN/Daily Mail dataset
test_data = pd.read_csv("test.csv")
test_data = test_data.head(2000)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
# Tokenize articles
def tokenize_articles(data):
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

test_data["tokens"] = tokenize_articles(test_data)
test_tokens = test_data["tokens"]
test_tokens = test_tokens.tolist()


# Load pre-trained model and tokenizer
model = DistilBertForTokenClassification.from_pretrained('fine_tuned_distbert', num_labels=2)


embbidings = []
# new_labels = []
IDs = []
for row in test_tokens:
    # Calculate the number of chunks needed for the current row
    num_chunks = (len(row) + 149) // 150
    row_emb = []
    sub_ids = []
    # Split tokens and labels into chunks
    for i in range(num_chunks):
        start_idx = i * 150
        end_idx = (i + 1) * 150
        
        chunk_tokens = row[start_idx:end_idx]
        # chunk_labels = labels[start_idx:end_idx]

        tokenized_input = tokenizer(chunk_tokens, is_split_into_words=True, add_special_tokens= False, return_tensors="pt")
        chunk_IDs = tokenized_input.input_ids
        sub_ids.append(chunk_IDs)

        outputs = model(**tokenized_input, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        row_emb.append(last_hidden_states[0])

    row_emb = [tensor.tolist() for tensor in row_emb]
    row_emb = [item for sublist in row_emb for item in sublist]
    embbidings.append(row_emb)

    sub_ids = [tensor.tolist() for tensor in sub_ids]
    sub_ids = [item for sublist in sub_ids for inner_list in sublist for item in inner_list]
    IDs.append(sub_ids)


print(IDs)
print(len(embbidings))
print(len(IDs))

test_inputs=np.array(embbidings)
# Load the best model 
from keras.models import load_model
LSTM_model = load_model('distbert_emb2BiLSTM2.h5')

def join_tokens(tokens):
    sentence = ''
    for token in tokens:
        if token.startswith('#'):
            sentence += token[2:]  # Remove '##' and concatenate
        else:
            sentence += ' ' + token  # Add a space before the token
    return sentence.strip()  # Remove leading/trailing white spaces

generated_summaries = []
for ids, row in zip(IDs, test_inputs):
    # Make predictions
    predictions = LSTM_model.predict([row])

    # The predictions will be probabilities in the range [0, 1]. 
    # convert them to class labels by rounding:
    predicted_labels = np.round(predictions)
    predicted_labels = np.array(predicted_labels)
    predicted_labels = predicted_labels.reshape(-1)
    predicted_labels = predicted_labels.tolist()
    predicted_labels = [int(i) for i in predicted_labels]
    summary_ids = [id for id, prediction in zip(ids, predicted_labels) if prediction]
    summary_tokens = tokenizer.convert_ids_to_tokens(summary_ids)
    summary = join_tokens(summary_tokens)
    generated_summaries.append(summary)
print(generated_summaries)

# Compute metrics function
rouge = load_metric("rouge")

reference_summaries = test_data['highlights'].tolist()
reference_summaries = [text.replace("\n", " ") for text in reference_summaries]
filtered_reference_summaries = []
for text in reference_summaries:
        # Lowercase the text
        text = text.lower()
        # Tokenize and remove stop words
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.lower() not in stop_words]
        tokens = [token for token in tokens if token not in string.punctuation]
        filtered_sentence = ' '.join(tokens)
        filtered_reference_summaries.append(filtered_sentence)
print(filtered_reference_summaries)

rouge_scores = rouge.compute(predictions=generated_summaries, references=filtered_reference_summaries, rouge_types=["rouge1", "rouge2", "rougeL"])

print(rouge_scores["rouge1"].mid)
print(rouge_scores["rouge2"].mid)
print(rouge_scores["rougeL"].mid)