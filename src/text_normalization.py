import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK dictionaries (runs quickly)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the NLP tools globally so they don't reload on every single row
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

print("NLP tools successfully loaded!")


# 1. Boilerplate Removal & Lowercasing
def step1_remove_boilerplate(text):
    text = str(text).lower()
    boilerplates = [
        r"i am writing to file a complaint",
        r"dear cfpb",
        r"to whom it may concern",
        r"consumer financial protection bureau"
    ]
    for phrase in boilerplates:
        text = re.sub(phrase, ' ', text)
    return text

# 2. HTML, URLs, and PII (Personal Identifiable Information)
def step2_remove_noise(text):
    text = re.sub(r'<.*?>', ' ', text)               # HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)      # URLs
    text = re.sub(r'\b\d{3}[-.\s]?\d{4}\b', ' ', text) # 7-digit phone numbers
    
    # PRO TIP: The CFPB masks names and accounts using "XXXX". This removes them!
    text = re.sub(r'[xX]{2,}', ' ', text)            
    return text

# 3. Punctuation and Special Characters
def step3_remove_punctuation(text):
    # Replace anything that IS NOT a letter, number, or space with a space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 4. Tokenization, Stopwords, and Lemmatization
def step4_normalize_text(text):
    tokens = word_tokenize(text)
    
    # Remove standard English stopwords
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize verbs (e.g., "charging" -> "charge")
    lemmas = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    
    # Lemmatize nouns (e.g., "fees" -> "fee")
    lemmas = [lemmatizer.lemmatize(t, pos='n') for t in lemmas]
    
    return " ".join(lemmas)

# --- THE MASTER PIPELINE FUNCTION ---
def execute_nlp_pipeline(text):
    """Passes the text through all 4 mini-functions in exact order."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
        
    text = step1_remove_boilerplate(text)
    text = step2_remove_noise(text)
    text = step3_remove_punctuation(text)
    text = step4_normalize_text(text)
    
    return text

print("Modular NLP pipeline defined!")