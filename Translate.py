import pandas as pd
import numpy as np
import tensorflow as tf


data = pd.read_excel("translate.xlsx")

x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

tokenizex = tf.keras.preprocessing.text.Tokenizer(num_words=20000)
tokenizex.fit_on_texts(x)
x = tokenizex.texts_to_sequences(x)
x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=20)

tokenizey = tf.keras.preprocessing.text.Tokenizer(num_words=20000)
tokenizey.fit_on_texts(y)
y = tokenizey.texts_to_sequences(y)
y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=20)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=20000, output_dim=64, input_length=20))
model.add(tf.keras.layers.GRU(64, return_sequences=True))
model.add(tf.keras.layers.GRU(64))
model.add(tf.keras.layers.Dense(20000, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x, y, epochs=10, batch_size=64)
while True:
    prediction = model.predict(input())
    print(prediction)

