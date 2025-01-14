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
