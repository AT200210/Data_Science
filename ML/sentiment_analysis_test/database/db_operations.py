import sqlite3
import pandas as pd

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

def get_past_analyses():
    """Retrieve past analysis results from the database"""
    conn = sqlite3.connect('sentiment_analysis.db')
    results = pd.read_sql_query("SELECT * FROM analysis_results ORDER BY timestamp DESC", conn)
    conn.close()
    return results
