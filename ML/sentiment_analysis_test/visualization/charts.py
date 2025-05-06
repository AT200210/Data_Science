import io
import base64
import plotly.express as px

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