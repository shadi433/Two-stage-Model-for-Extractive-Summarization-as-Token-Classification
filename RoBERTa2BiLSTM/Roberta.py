from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from IPython.display import display
import string

# Adjust pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Load the CNN/Daily Mail dataset
train_data = pd.read_csv("train.csv")
train_data = train_data.head(20000)

val_data = pd.read_csv("validation.csv")
val_data = val_data.head(2000)

# Initialize tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=2)


# Tokenize articles
def tokenize_articles(data):
    stop_words = set(stopwords.words("english"))
    tokenized_articles = []

    for article in data["article"]:
        # Tokenize and remove stop words
        words = word_tokenize(article)
        tokens = [word for word in words if word.lower() not in stop_words]
        tokens = [token for token in tokens if token not in string.punctuation]
        tokenized_articles.append(tokens)

    data["tokens"] = tokenized_articles
    return data

train_data = tokenize_articles(train_data)
val_data = tokenize_articles(val_data)

# Preprocess dataset and create binary label sequences
def add_binary_labels(data):
    binary_label_sequences = []

    for _, row in data.iterrows():
        input_tokens = row["tokens"]
        summary_tokens = word_tokenize(row["highlights"])
        binary_label_sequence = []
        for token in input_tokens:
            if token in summary_tokens:
                binary_label_sequence.append("1")
            else:
                binary_label_sequence.append("0")
        binary_label_sequences.append(binary_label_sequence)

    return binary_label_sequences

def make_new_data(data):
    # Create a list to store new data
    new_data_list = []

    # Iterate through each row of the train_data DataFrame
    for index, row in data.iterrows():
        tokens = row["tokens"]
        labels = row["binary_labels"]
        
        # Calculate the number of chunks needed for the current row
        num_chunks = (len(tokens) + 511) // 512
        
        # Split tokens and labels into chunks and add them to the new DataFrame
        for i in range(num_chunks):
            start_idx = i * 512
            end_idx = (i + 1) * 512
            
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]
            
            # Append the new data to the list
            new_data_list.append({"tokens": chunk_tokens, "binary_labels": chunk_labels})

    # Convert the list to a DataFrame
    new_data = pd.DataFrame(new_data_list)

    return new_data

train_data["binary_labels"] = add_binary_labels(train_data)
new_train_data = make_new_data(train_data)

# train_data = train_data.iloc[:100]
train_tokens = new_train_data["tokens"].tolist()
train_labels = new_train_data["binary_labels"].tolist()

val_data["binary_labels"] = add_binary_labels(val_data)
new_val_data = make_new_data(val_data)

# val_data = val_data.iloc[:100]
val_tokens = new_val_data["tokens"].tolist()
val_labels = new_val_data["binary_labels"].tolist()



train_encodings = tokenizer(train_tokens, is_split_into_words=True, return_offsets_mapping=True, padding="max_length", truncation=True, max_length=512)
train_labels = tokenizer(train_labels, is_split_into_words=True, return_offsets_mapping=True, padding="max_length", truncation=True, max_length=512).input_ids
val_encodings = tokenizer(val_tokens, is_split_into_words=True, return_offsets_mapping=True, padding="max_length", truncation=True, max_length=512)
val_labels = tokenizer(val_labels, is_split_into_words=True, return_offsets_mapping=True, padding="max_length", truncation=True, max_length=512).input_ids

# We have to make sure that the [PAD], [CLS] and [SEP] tokens is ignored
train_labels= [[-100 if token in (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id) else token for token in labels] for labels in train_labels]
val_labels= [[-100 if token in (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id) else token for token in labels] for labels in val_labels]


print(train_encodings["input_ids"][0])
print(train_labels[0])

def lable_correction(labels, encodings):
    corrected_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # Create a mask to identify positions with first offset equal to 0 and second offset not equal to 0
        mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)

        # Assign the corresponding doc_labels elements to the positions in doc_enc_labels where the mask is true
        doc_enc_labels[mask] = np.array(doc_labels)[mask]

        corrected_labels.append(doc_enc_labels.tolist())

    return corrected_labels


train_labels = lable_correction(train_labels, train_encodings)
train_labels = [[0 if token == 321 else 1 if token == 112 else token for token in labels] for labels in train_labels]
val_labels = lable_correction(val_labels, val_encodings)
val_labels = [[0 if token == 321 else 1 if token == 112 else token for token in labels] for labels in val_labels]

print(train_labels[0])

class ToDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = ToDataset(train_encodings, train_labels)
val_dataset = ToDataset(val_encodings, val_labels)

def write_metrics_to_file(metrics):
    with open("./metrics/metrics_bert_bert.txt", "w") as f:
        f.write("Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Write metrics to a file
    write_metrics_to_file(metrics)

    return metrics


epochs = 10
batch_size = 4

training_args = TrainingArguments(
    output_dir='./roberta_results',          # output directory
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    # eval_steps=150,
    num_train_epochs=epochs,              # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    warmup_steps=5000,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./robertalogs',            # directory for storing logs
    logging_steps=5000,                # Number of update steps between two logs
    save_strategy="epoch",
    # save_steps=300,
    save_total_limit=3,              # Maximum number of checkpoints to keep
    dataloader_drop_last=True,      # Drop the last incomplete batch during training
    load_best_model_at_end=True,     # Whether or not to load the best model found during training at the end of training.
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

trainer.save_model("fine_tuned_RoBERTa")

def find_latest_checkpoint(model_dir):
    checkpoint_re = re.compile(r"checkpoint-(\d+)")
    checkpoints = []

    for entry in os.scandir(model_dir):
        if entry.is_dir() and checkpoint_re.match(entry.name):
            checkpoint_num = int(checkpoint_re.match(entry.name).group(1))
            checkpoints.append((checkpoint_num, entry.path))

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_dir}")

    latest_checkpoint = max(checkpoints, key=lambda x: x[0])
    return latest_checkpoint[1]


latest_checkpoint = find_latest_checkpoint("./roberta_results/")
print(f"Latest bert checkpoint: {latest_checkpoint}")

# bert_model = BertForTokenClassification.from_pretrained("fine_tuned_bert")



