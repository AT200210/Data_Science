import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import matplotlib.pyplot as plt

# Import functionality from modules
from database.db_operations import create_database, save_to_database, get_past_analyses
from analysis.sentiment import hybrid_sentiment_analysis, analyze_sentiment_transformers, simple_sentiment_analysis
from analysis.text_processing import preprocess_text, chunk_text, scrape_url
from analysis.topics import extract_agenda
from visualization.charts import plot_sentiment_distribution
from visualization.wordcloud import generate_wordcloud, get_sentiment_color

# Initialize session state variables
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = []

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