import yake
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_with_sumy(text, sentences_count=3):
    """Generate a text summary using LSA algorithm"""
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

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