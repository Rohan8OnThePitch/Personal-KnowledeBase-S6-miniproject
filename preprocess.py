import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK resources (if not already downloaded)
# Specify the directory where you want to save the NLTK data
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Preprocess text data.')
parser.add_argument('--nltk_data_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'nltk_data'),
                    help='Directory to save NLTK data')
args = parser.parse_args()

nltk_data_dir = args.nltk_data_dir

# Ensure the directory exists
os.mkdir(nltk_data_dir)

# Append the NLTK data directory to the NLTK data path
# Append the NLTK data directory to the NLTK data path to ensure NLTK can find the downloaded data
nltk.data.path.append(nltk_data_dir)

nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'], download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

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
