import re
import logging
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize logger
logger = logging.getLogger(__name__)

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_sm')
    logger.info("Loaded SpaCy model successfully")
except OSError:
    logger.warning("Could not load SpaCy model. Some features may be limited.")
    nlp = None

def preprocess_text(text):
    """
    Preprocess text for summarization
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Handle LaTeX equations
    text = re.sub(r'\$\$.*?\$\$|\$.*?\$', '[EQUATION]', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Handle common section headers in scientific papers
    text = re.sub(r'\b(abstract|introduction|background|related work|methodology|method|experiment|results|discussion|conclusion|references)\b\s*\n', 
                 lambda m: f"\n{m.group(1).upper()}:\n", 
                 text, 
                 flags=re.IGNORECASE)
    
    return text

def split_into_sections(text):
    """
    Split text into sections based on common section headers in scientific papers
    
    Args:
        text (str): Preprocessed text
        
    Returns:
        dict: Dictionary of sections
    """
    section_pattern = r'(?:^|\n)(abstract|introduction|background|related work|methodology|method|experiment|results|discussion|conclusion):\s*\n'
    sections = {}
    
    # Find all section headers
    matches = list(re.finditer(section_pattern, text, re.IGNORECASE))
    
    if not matches:
        # If no sections found, treat entire text as a single section
        sections['text'] = text
        return sections
    
    # Process each section
    for i, match in enumerate(matches):
        section_name = match.group(1).lower()
        start_pos = match.end()
        
        # Get the end position (start of next section or end of text)
        if i < len(matches) - 1:
            end_pos = matches[i+1].start()
        else:
            end_pos = len(text)
        
        # Extract section content
        section_content = text[start_pos:end_pos].strip()
        sections[section_name] = section_content
    
    # Check for abstract/intro at the beginning
    if matches[0].start() > 0:
        start_content = text[:matches[0].start()].strip()
        if start_content:
            if 'abstract' not in sections and len(start_content) < 1000:
                sections['abstract'] = start_content
            else:
                sections['intro_text'] = start_content
    
    return sections

def extract_keywords(text, top_n=10):
    """
    Extract key terms from text
    
    Args:
        text (str): Input text
        top_n (int): Number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    if not text:
        return []
    
    # Use SpaCy for better keyword extraction if available
    if nlp:
        doc = nlp(text)
        # Extract noun phrases and named entities
        keywords = []
        
        # Add named entities
        keywords.extend([ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']])
        
        # Add noun chunks (noun phrases)
        keywords.extend([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3])
        
        # Count occurrences
        keyword_counts = Counter(keywords)
        
        # Return top N keywords
        return [keyword for keyword, _ in keyword_counts.most_common(top_n)]
    
    else:
        # Fallback method using NLTK
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        
        # Filter out stopwords and punctuation
        filtered_words = [word for word in words if word not in stop_words and word not in punctuation and len(word) > 1]
        
        # Count occurrences
        word_counts = Counter(filtered_words)
        
        # Return top N keywords
        return [word for word, _ in word_counts.most_common(top_n)]

def extract_sentences(text, max_sentences=None):
    """
    Split text into sentences
    
    Args:
        text (str): Text to split
        max_sentences (int): Maximum number of sentences to return
        
    Returns:
        list: List of sentences
    """
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    
    if max_sentences and len(sentences) > max_sentences:
        return sentences[:max_sentences]
    
    return sentences

def clean_sentence(sentence):
    """
    Clean a sentence for summarization
    
    Args:
        sentence (str): Sentence to clean
        
    Returns:
        str: Cleaned sentence
    """
    # Remove citations
    sentence = re.sub(r'\[\d+\]|\[[\d,\s]+\]', '', sentence)
    
    # Remove figure and table references
    sentence = re.sub(r'(figure|fig\.|table)\s+\d+', '', sentence, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    return sentence

def is_citation_sentence(sentence):
    """
    Check if a sentence contains a citation
    
    Args:
        sentence (str): Sentence to check
        
    Returns:
        bool: True if the sentence contains a citation
    """
    # Check for citation patterns like [1], [1,2], (Smith et al., 2019)
    citation_patterns = [
        r'\[\d+\]',
        r'\[[\d,\s]+\]',
        r'\(\w+\s+et\s+al\.,?\s+\d{4}\)',
        r'\(\w+\s+and\s+\w+,?\s+\d{4}\)',
        r'\(\w+,?\s+\d{4}\)'
    ]
    
    for pattern in citation_patterns:
        if re.search(pattern, sentence):
            return True
    
    return False

def prepare_for_model(text, max_length=None):
    """
    Prepare text for input to the model
    
    Args:
        text (str): Text to prepare
        max_length (int): Maximum length in words
        
    Returns:
        str: Prepared text
    """
    # Preprocess text
    text = preprocess_text(text)
    
    # Limit length if needed
    if max_length:
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
    
    return text
