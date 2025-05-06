import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from bs4 import BeautifulSoup
import sqlite3
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lsa import LsaSummarizer
import yake
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.express as px
import io
import base64


# Initialize session state variables
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = []

def create_database():
    """Create SQLite database for storing sentiment analysis results"""
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text_source TEXT,
        content_snippet TEXT,
        sentiment_score REAL,
        sentiment_label TEXT,
        agenda TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def save_to_database(source, content, sentiment_score, sentiment_label, agenda):
    """Save analysis results to the database"""
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    
    # Insert data
    cursor.execute('''
    INSERT INTO analysis_results (text_source, content_snippet, sentiment_score, sentiment_label, agenda)
    VALUES (?, ?, ?, ?, ?)
    ''', (source, content[:200], sentiment_score, sentiment_label, agenda))
    
    conn.commit()
    conn.close()

def summarize_with_sumy(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)


def get_past_analyses():
    """Retrieve past analysis results from the database"""
    conn = sqlite3.connect('sentiment_analysis.db')
    results = pd.read_sql_query("SELECT * FROM analysis_results ORDER BY timestamp DESC", conn)
    conn.close()
    return results

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

def analyze_sentiment_transformers(text):
    """Analyze sentiment using a pre-trained transformer model"""
    # Load pre-trained sentiment analysis model
    # Using a model that's better at handling short positive statements
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    # Create sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # For very short texts, don't chunk
    if len(text.split()) < 100:
        result = sentiment_analyzer(text)[0]
        return result
    
    # Process text in chunks if it's longer
    chunk_size = 500  # DistilBERT has a max token limit
    chunks = chunk_text(text, chunk_size)
    
    results = []
    for chunk in chunks:
        if chunk.strip():  # Only process non-empty chunks
            result = sentiment_analyzer(chunk)[0]
            results.append(result)
    
    # Aggregate results
    if not results:
        return {"label": "UNKNOWN", "score": 0.5}
    
    # Calculate weighted average based on label and score
    positive_score = 0
    total_chunks = len(results)
    
    for result in results:
        if result["label"] == "POSITIVE":
            positive_score += result["score"]
        elif result["label"] == "NEGATIVE":
            positive_score += (1 - result["score"])
    
    # Average sentiment score (higher means more positive)
    avg_score = positive_score / total_chunks
    
    # Map average score to label
    if avg_score > 0.6:
        label = "POSITIVE"
    elif avg_score < 0.4:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
        
    return {"label": label, "score": avg_score}

def simple_sentiment_analysis(text):
    """A simple rule-based sentiment analysis for short texts"""
    # List of positive and negative words
    positive_words = ["happy", "good", "great", "excellent", "wonderful", "love", "like", "amazing", 
                      "joy", "enjoy", "beautiful", "best", "better", "glad", "positive", "awesome"]
    negative_words = ["sad", "bad", "terrible", "horrible", "hate", "dislike", "awful", "worst",
                      "worse", "unhappy", "angry", "negative", "poor", "sucks", "disappointed"]
    
    # Count occurrences
    text_lower = text.lower()
    
    # Check for negations
    has_negation = any(neg in text_lower for neg in ["not ", "no ", "never ", "don't ", "doesn't ", "didn't "])
    
    # Count sentiment words
    positive_count = sum(1 for word in positive_words if f" {word} " in f" {text_lower} ")
    negative_count = sum(1 for word in negative_words if f" {word} " in f" {text_lower} ")
    
    # Simple scoring
    if has_negation:
        # Flip the sentiment if negation is present
        temp = positive_count
        positive_count = negative_count
        negative_count = temp
    
    # Calculate score (0 to 1, higher means more positive)
    if positive_count + negative_count == 0:
        score = 0.5  # Neutral
    else:
        score = positive_count / (positive_count + negative_count)
    
    # Determine label
    if score > 0.6:
        label = "POSITIVE"
    elif score < 0.4:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    
    return {"label": label, "score": score}

def hybrid_sentiment_analysis(text):
    """Combines transformer-based and rule-based sentiment analysis"""
    # For very short texts, use both methods and combine
    if len(text.split()) < 20:
        transformer_result = analyze_sentiment_transformers(text)
        rule_based_result = simple_sentiment_analysis(text)
        
        # Check if both methods agree on negative sentiment
        if transformer_result["label"] == "NEGATIVE" or rule_based_result["label"] == "NEGATIVE":
            # If either method detects negative sentiment, prioritize it for short negative statements
            combined_score = min(0.7 * transformer_result["score"] + 0.3 * rule_based_result["score"], 0.4)
            label = "NEGATIVE"
        else:
            # For non-negative cases, combine results with more weight to transformer result (70/30 split)
            combined_score = 0.7 * transformer_result["score"] + 0.3 * rule_based_result["score"]
            
            # Determine final label
            if combined_score > 0.6:
                label = "POSITIVE"
            elif combined_score < 0.4:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
        
        return {"label": label, "score": combined_score}
    else:
        # For longer texts, use only transformer-based analysis
        return analyze_sentiment_transformers(text)
    
    
def extract_agenda(text, num_keywords=5):
    """Extract the main agenda or topics from the text"""
    # Initialize keyword extractor
    kw_extractor = yake.KeywordExtractor(
        lan="en", 
        n=2, # Extract phrases of 2 words
        dedupLim=0.9,
        dedupFunc='seqm',
        windowsSize=3,
        top=num_keywords,
        features=None
    )
    
    # Extract keywords
    keywords = kw_extractor.extract_keywords(text)
    agenda_topics = [kw[0] for kw in keywords]
    
    # Generate a summary using Sumy
    summary = ""
    if len(text.split()) > 100:
        try:
            summary = summarize_with_sumy(text, sentences_count=3)
        except:
            # Fallback: First 3 sentences
            sentences = sent_tokenize(text)
            summary = ' '.join(sentences[:3]) if len(sentences) > 3 else ' '.join(sentences)
    
    return {
        "topics": agenda_topics,
        "summary": summary
    }


def generate_wordcloud(text):
    """Generate a word cloud from the text"""
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color="white",
        max_words=100,
        stopwords=stopwords,
        width=800,
        height=400
    ).generate(text)
    
    return wordcloud

def get_sentiment_color(sentiment):
    """Get color based on sentiment"""
    if sentiment == "POSITIVE":
        return "green"
    elif sentiment == "NEGATIVE":
        return "red"
    else:
        return "blue"

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 for display in Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def plot_sentiment_distribution(chunks_sentiments):
    """Plot sentiment distribution across text chunks"""
    # Prepare data
    chunk_ids = list(range(1, len(chunks_sentiments) + 1))
    scores = [s['score'] for s in chunks_sentiments]
    labels = [s['label'] for s in chunks_sentiments]
    
    # Create plot
    fig = px.line(
        x=chunk_ids, 
        y=scores, 
        title="Sentiment Distribution Across Text Chunks",
        labels={"x": "Chunk ID", "y": "Sentiment Score (Higher = More Positive)"},
        color_discrete_sequence=["blue"],
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        yaxis=dict(range=[0, 1]),
        hovermode="x unified"
    )
    
    return fig

def main():
    st.set_page_config(page_title="Advanced Sentiment Analysis", layout="wide")
    
    # Create database
    create_database()
    
    # App title and description
    st.title("🔍 Advanced Sentiment Analysis with Agenda Extraction")
    st.markdown("""
    This app performs sentiment analysis on text or content from URLs.
    It also extracts the main agenda/topics and provides visual insights into the sentiment distribution.
    """)
    
    # Sidebar for input options
    st.sidebar.title("Input Options")
    input_type = st.sidebar.radio("Select Input Type", ["Text", "URL"])
    
    if input_type == "Text":
        text_input = st.sidebar.text_area("Enter your text here:", height=200)
        source = "Direct Text Input"
    else:
        url_input = st.sidebar.text_input("Enter URL:")
        source = url_input
    
    # Analysis options
    st.sidebar.title("Analysis Options")
    chunk_size = st.sidebar.slider("Chunk Size (words)", 100, 1000, 500, 100)
    
    # Add sentiment analysis method selection
    analysis_method = st.sidebar.radio(
        "Sentiment Analysis Method", 
        ["Hybrid (recommended)", "Transformer-Only", "Rule-Based (for very short texts)"]
    )
    
    # Action button
    analyze_button = st.sidebar.button("Analyze Content")
    
    # Main content area
    main_container = st.container()
    
    if analyze_button:
        with st.spinner("Processing content..."):
            # Get content based on input type
            if input_type == "Text" and text_input:
                content = text_input
            elif input_type == "URL" and url_input:
                content = scrape_url(url_input)
                if not content:
                    st.error("Failed to retrieve content from the URL.")
                    return
            else:
                st.warning("Please provide input text or URL.")
                return
            
            # Display raw content
            with main_container.expander("Original Content", expanded=False):
                st.write(content)
            
            # Preprocess text with sentiment preservation
            processed_text = preprocess_text(content, preserve_sentiment_words=True)
            
            # For very short texts, use the selected method directly
            if len(processed_text.split()) < 20:
                if analysis_method == "Hybrid (recommended)":
                    overall_sentiment = hybrid_sentiment_analysis(processed_text)
                elif analysis_method == "Transformer-Only":
                    overall_sentiment = analyze_sentiment_transformers(processed_text)
                else:  # Rule-based
                    overall_sentiment = simple_sentiment_analysis(processed_text)
                
                overall_sentiment_label = overall_sentiment["label"]
                overall_sentiment_score = overall_sentiment["score"]
                
                # Create a single chunk for visualization
                chunks_sentiments = [overall_sentiment]
            else:
                # For longer texts, chunk the text
                chunks = chunk_text(processed_text, chunk_size)
                
                # Analyze sentiment for each chunk
                chunks_sentiments = []
                for chunk in chunks:
                    if analysis_method == "Hybrid (recommended)":
                        sentiment = hybrid_sentiment_analysis(chunk)
                    elif analysis_method == "Transformer-Only":
                        sentiment = analyze_sentiment_transformers(chunk)
                    else:  # Rule-based
                        sentiment = simple_sentiment_analysis(chunk)
                    
                    chunks_sentiments.append(sentiment)
                
                # Calculate overall sentiment
                overall_sentiment_score = sum(s['score'] for s in chunks_sentiments) / len(chunks_sentiments)
                
                if overall_sentiment_score > 0.6:
                    overall_sentiment_label = "POSITIVE"
                elif overall_sentiment_score < 0.4:
                    overall_sentiment_label = "NEGATIVE"
                else:
                    overall_sentiment_label = "NEUTRAL"
            
            # Extract agenda
            agenda_data = extract_agenda(content)
            agenda_text = ", ".join(agenda_data["topics"])
            
            # Save to database
            save_to_database(source, content, overall_sentiment_score, overall_sentiment_label, agenda_text)
            
            # Add to session state
            st.session_state.analyzed_data.append({
                "source": source,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "overall_sentiment": overall_sentiment_label,
                "sentiment_score": overall_sentiment_score,
                "agenda": agenda_data["topics"],
                "summary": agenda_data["summary"]
            })
            
            # Display results
            col1, col2 = main_container.columns([2, 1])
            
            with col1:
                st.subheader("Sentiment Analysis Results")
                
                # Overall sentiment
                sentiment_color = get_sentiment_color(overall_sentiment_label)
                st.markdown(f"""
                <div style='background-color: {sentiment_color}25; padding: 20px; border-radius: 10px; border: 1px solid {sentiment_color}'>
                <h3 style='color: {sentiment_color}'>Overall Sentiment: {overall_sentiment_label}</h3>
                <p>Confidence Score: {overall_sentiment_score:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Sentiment distribution chart (only if multiple chunks)
                if len(chunks_sentiments) > 1:
                    st.subheader("Sentiment Distribution")
                    sentiment_fig = plot_sentiment_distribution(chunks_sentiments)
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                
                # Word cloud
                st.subheader("Word Cloud")
                wordcloud = generate_wordcloud(content)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                # Display agenda extraction results
                st.subheader("Content Agenda")
                st.markdown("**Main Topics:**")
                for topic in agenda_data["topics"]:
                    st.markdown(f"- {topic}")
                    
                if agenda_data["summary"]:
                    st.subheader("Summary")
                    st.write(agenda_data["summary"])
    
    # Display history tab
    st.sidebar.title("Analysis History")
    if st.sidebar.button("Show Past Analyses"):
        past_analyses = get_past_analyses()
        st.subheader("Previous Analysis Results")
        st.dataframe(past_analyses)

if __name__ == "__main__":
    main()