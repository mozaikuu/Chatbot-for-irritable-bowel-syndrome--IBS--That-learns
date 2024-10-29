import json
import numpy as np
import random
import nltk
from newspaper import Article
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

# Load the intents file
with open('KnowledgeBase.json') as file:
    data = json.load(file)

# Lists for storing the words, classes (tags), and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Preprocess the data
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add the documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

# Create the bag of words for each sentence
for doc in documents:
    bag = []
    word_patterns = doc[0]
    # Lemmatize each word - create the base word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # Bag of words: 1 if word is in the pattern, otherwise 0
        bag.append(1 if word in word_patterns else 0)

    # Output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Append the bag of words and the output_row to training data
    training.append([bag, output_row])

# Convert training data into NumPy arrays
training = np.array(training, dtype=object)

# Separate features and labels
X_train = np.array([i[0] for i in training])
y_train = np.array([i[1] for i in training])

# Shuffle the training data and convert it into a numpy array
random.shuffle(training)
training = np.array(training)

# Split the features (X) and labels (y)
X_train = np.array([i[0] for i in training])
y_train = np.array([i[1] for i in training])

# Create a model - 3 layers. First layer 128 neurons, second layer 64 neurons, third output layer for number of intents
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=200, batch_size=100, verbose=1)

# Save the model
model.save('chatbot_model.h5', history)

print("Model training complete!")

# Save the words and classes
import pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Load necessary files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

with open('KnowledgeBase.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Preprocess input text
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict the class (intent)
def predict_class(sentence):
    bow_data = bow(sentence, words)
    res = model.predict(np.array([bow_data]))[0]
    ERROR_THRESHOLD = 0.8
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)

    # Print predicted intents and probabilities
    print(f"Predictions: {results}")

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you please rephrase?"
    
    tag = intents_list[0]['intent']  # Get the predicted intent tag
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])  # Choose a random response

# Main chatbot interaction
def chatbot_response(text):
    intents = predict_class(text)
    response = get_response(intents, data)
    return response

# Testing the chatbot
while True:
    message = input("You: ")
    response = chatbot_response(message)
    print("Bot:", response)


# TODO 1: Add more intents to the KnowledgeBase.json file 
# TODO 2: Improve the model by adding more layers and neurons
# TODO 3: Train the model with more data Synonym expansion Rephrasing
# TODO 4: 