from wordcloud import WordCloud, STOPWORDS

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