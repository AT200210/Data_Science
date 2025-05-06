import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess_text(text, preserve_sentiment_words=True):
    """Preprocess the text for sentiment analysis
    
    Args:
        text: The text to preprocess
        preserve_sentiment_words: If True, preserves words that might indicate sentiment
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    if preserve_sentiment_words:
        # Only remove punctuation but keep words intact
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space instead of removing
        text = re.sub(r'\s+', ' ', text).strip()  # Clean up extra spaces
        
        # Don't remove stopwords or perform stemming for sentiment analysis
        return text
    else:
        # More aggressive preprocessing for other tasks
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Stemming
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
        
        # Join tokens back to text
        processed_text = ' '.join(tokens)
        
        return processed_text

def chunk_text(text, chunk_size=1000):
    """Split text into manageable chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

def scrape_url(url):
    """Scrape content from a URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        st.error(f"Error scraping URL: {e}")
        return None