import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences


embbidings = np.load('Rob_embeddings.npy', allow_pickle=True).tolist()
new_labels = np.load('Rob_new_labels.npy', allow_pickle=True).tolist()

embbidings = embbidings[:5000]
new_labels = new_labels[:5000]

print('start LSTM')
token_emb = np.array(embbidings)
token_labels = np.array(new_labels)

# Define padding value
# Pad inputs
inputs_padded = pad_sequences(token_emb, padding='post', dtype='float32', value=-100)
# Pad labels
max_sequence_length = inputs_padded.shape[1]
labels_padded = pad_sequences(token_labels, maxlen=max_sequence_length, padding='post', value=-1)
# Reshape labels
labels_padded = labels_padded.reshape(*labels_padded.shape, 1)
sample_weight = np.where(labels_padded == -1, 0, 1)

# Load the best model 
# from keras.models import load_model
# LSTM_model = load_model('bert_emb2BiLSTM3.h5')

# Define the model
LSTM_model = Sequential()
LSTM_model.add(Masking(mask_value=-100., input_shape=(None, 768)))
LSTM_model.add(Bidirectional(LSTM(units=50, return_sequences=True, dropout=0)))
LSTM_model.add(Dense(1, activation='sigmoid'))

# Compile the model
LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], weighted_metrics=['accuracy'])
# LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.SensitivityAtSpecificity(0.5, num_thresholds=1)])

# Define callbacks
checkpoint = ModelCheckpoint('Roberta_emb2BiLSTM.h5', monitor='val_loss', save_best_only=True, save_weights_only=False)
earlystopping = EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

# Train the model
LSTM_model.fit(inputs_padded, labels_padded, sample_weight=sample_weight, epochs=50, validation_split=0.1, callbacks=[checkpoint, earlystopping, reduce_lr])