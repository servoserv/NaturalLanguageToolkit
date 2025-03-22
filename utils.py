import re
import logging
import hashlib
from typing import List, Dict, Any
import json
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename: str) -> bool:
    """
    Check if a file is allowed based on its extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if the file extension is allowed, False otherwise
    """
    allowed_extensions = {'txt', 'pdf', 'json', 'xml', 'csv', 'md'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing potentially harmful content.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove potential XSS
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?on\w+=".*?".*?>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Replace non-printable characters
    text = ''.join(c if c.isprintable() or c.isspace() else ' ' for c in text)
    
    return text.strip()

def generate_document_id(text: str) -> str:
    """
    Generate a unique ID for a document based on its content.
    
    Args:
        text: Document text
        
    Returns:
        SHA-256 hash of the text (first 16 chars)
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def format_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format numerical metrics for display.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Dictionary with formatted metrics
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = str(value)
    return formatted

def load_summary(summary_id: str) -> Dict[str, Any]:
    """
    Load a summary from file storage.
    
    Args:
        summary_id: ID of the summary to load
        
    Returns:
        Dictionary containing the summary data
    """
    try:
        filepath = f'summaries/{summary_id}.json'
        if not os.path.exists(filepath):
            logger.error(f"Summary file not found: {filepath}")
            return {}
        
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading summary {summary_id}: {str(e)}")
        return {}

def save_summary(summary_data: Dict[str, Any]) -> bool:
    """
    Save a summary to file storage.
    
    Args:
        summary_data: Dictionary containing the summary data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if 'id' not in summary_data:
            logger.error("Cannot save summary: missing ID")
            return False
        
        summary_id = summary_data['id']
        filepath = f'summaries/{summary_id}.json'
        
        with open(filepath, 'w') as f:
            json.dump(summary_data, f)
        
        return True
    except Exception as e:
        logger.error(f"Error saving summary: {str(e)}")
        return False

def get_timestamp() -> str:
    """Get current timestamp in string format."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 300) -> str:
    """
    Truncate text to a maximum length while preserving whole sentences.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (characters)
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > max_length * 0.7:  # Only truncate at sentence if it's not too short
        return truncated[:last_period+1] + " [...]"
    else:
        # Otherwise truncate at word boundary
        last_space = truncated.rfind(' ')
        return truncated[:last_space] + " [...]"

def create_directory_if_not_exists(directory: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def is_valid_summary_id(summary_id: str) -> bool:
    """Check if a summary ID is valid."""
    return bool(re.match(r'^[a-f0-9-]+$', summary_id))
