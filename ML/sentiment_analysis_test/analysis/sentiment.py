from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from analysis.text_processing import chunk_text

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