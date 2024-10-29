import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from newspaper import Article

# Download NLTK data
nltk.download('punkt')

class SimpleChatbot:
    def __init__(self, text_file):
        # Read and preprocess the text
        self.text_data = self.read_text_file(text_file)
        self.sentences, self.words = self.preprocess_text(self.text_data)

    def read_text_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def preprocess_text(self, text):
        # Remove punctuation and convert text to lowercase
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        # Tokenize sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        return sentences, words

    def handle_greetings(self, user_input):
        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        if any(greet in user_input.lower() for greet in greetings):
            return "Hello! How can I assist you today?"
        
        goodbyes = ["bye", "goodbye", "see you", "later"]
        if any(bye in user_input.lower() for bye in goodbyes):
            return "Goodbye! Feel free to ask more questions if needed."
        
        return None

    def find_keywords_response(self, user_query):
        query_tokens = word_tokenize(user_query.lower())
        best_match = None
        max_overlap = 0
        
        for sentence in self.sentences:
            sentence_tokens = word_tokenize(sentence)
            overlap = len(set(query_tokens) & set(sentence_tokens))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = sentence
                
        if best_match:
            return best_match
        else:
            return "Sorry, I don't have enough information about that."

    def respond(self, user_query):
        # Check for greetings
        greeting_response = self.handle_greetings(user_query)
        if greeting_response:
            return greeting_response
        
        # Otherwise, find the most relevant response from the text file
        return self.find_keywords_response(user_query)
      
article= Article('https://en.wikipedia.org/wiki/Irritable_bowel_syndrome')
article.download()
article.parse()
article.nlp()
corpus=article.text
corpus = corpus.lower()
with open('data.txt', 'w', encoding='utf-8') as file:
    file.write(corpus)

# Example usage
chatbot = SimpleChatbot('data.txt')

# User interaction loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "exit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot.respond(user_input)
    print("Chatbot:", response)
