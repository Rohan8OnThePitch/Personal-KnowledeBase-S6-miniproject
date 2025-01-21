import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (if not already downloaded)
'''nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')'''

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove non-ASCII characters (optional)
    text = ''.join(char for char in text if ord(char) < 128)

    # Remove any special characters, numbers, or unwanted symbols
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_text = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Join the words back into a string
    return ' '.join(cleaned_text)

# Example usage
raw_text = "This is an example sentence with some noisy data! Running..."
processed_text = preprocess_text(raw_text)
print(processed_text)  # Output will be clean, tokenized text without stopwords
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
        if token.isalpha() and token not in stop_words:  # Alphabetic words
            cleaned_tokens.append(lemmatizer.lemmatize(token))
        elif token.isnumeric():  # Numbers
            cleaned_tokens.append(token)
        elif not token.isalnum():  # Special characters
            cleaned_tokens.append(token)

    # Join the tokens back into a single string
    return ' '.join(cleaned_tokens)



