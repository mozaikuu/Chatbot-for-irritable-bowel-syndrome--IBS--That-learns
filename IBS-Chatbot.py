import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())  # Tokenize
    return [lemmatizer.lemmatize(w) for w in words]  # Lemmatize


model = Sequential([
    Dense(128, input_shape=(input_shape,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(classes), activation='softmax')  # Number of classes = number of intents
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train = np.array(training_sentences)
y_train = np.array(training_labels)

model.fit(X_train, y_train, epochs=200, batch_size=8)

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

while True:
    message = input("You: ")
    intents = predict_class(message, model)
    response = get_response(intents, intents_json)
    print("Bot:", response)
    
    #TODO integrate transformer models for handling complex conversations if needed
