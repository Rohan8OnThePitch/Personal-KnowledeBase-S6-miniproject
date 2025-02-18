import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string  # Import the string module

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Normalize line breaks and remove unnecessary spaces
    text = re.sub(r'\s+', ' ', text.strip())

    # Split alphanumeric combinations (e.g., "hello1234world" -> "hello 1234 world")
    text = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', text)
    text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)

    # Tokenize the text into words, numbers, and special characters
    tokens = word_tokenize(text)

    # Process tokens: lemmatize words, keep numbers and special characters
    cleaned_tokens = []
    for token in tokens:
        if token.isalpha():  # Alphabetic words
            if token not in stop_words:
                cleaned_tokens.append(lemmatizer.lemmatize(token))
        elif token.isnumeric():  # Numbers
            cleaned_tokens.append(token)
        elif not token.isalnum() and token not in string.punctuation:  # Special characters (excluding punctuation)
            cleaned_tokens.append(token)

    # Join the tokens back into a single string
    return ' '.join(cleaned_tokens)
