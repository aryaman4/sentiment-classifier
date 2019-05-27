import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter='\t', quoting=3)
test = pd.read_csv("data/testData.tsv", header=0, delimiter='\t', quoting=3)

file = open("model_two.txt", encoding='utf-8')
embeddings = {}

for line in file:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:])
	embeddings[word] = coefs
file.close()

tokenizer = Tokenizer()
review_sentences = train['review'].values.tolist()
tokenizer.fit_on_texts(review_sentences)
sequences = tokenizer.texts_to_sequences(review_sentences)
index = tokenizer.word_index
max_length = max([len(s.split()) for s in review_sentences])
padded = pad_sequences(sequences, maxlen=max_length)

tokenizer.fit_on_texts(test['review'].values.tolist())
test_sequences = tokenizer.texts_to_sequences(test['review'].values.tolist())
test_padded = pad_sequences(test_sequences, maxlen=max_length)

num_words = len(index) + 1
embed_matrix = np.zeros((num_words, 200))

for word, i in index.items():
	if i > num_words:
		continue
	vector = embeddings.get(word)
	if vector is not None:
		embed_matrix[i] = vector

model = Sequential()
embed_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embed_matrix), input_length=max_length, trainable=False)

model.add(embed_layer)
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

sentiments = train['sentiment'].values
split = int(0.8*len(padded))
X_train = padded[:split]
print(X_train.shape)
X_test = padded[split:]
y_train = sentiments[:split]
print(y_train.shape)
y_test = sentiments[split:]

model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test), verbose=2)