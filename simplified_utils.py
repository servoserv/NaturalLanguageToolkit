"""
Simplified utility functions for text processing and sample data loading.
"""

import re
import os
import hashlib
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def preprocess_text(text):
    """
    Preprocess text for summarization
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove urls
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Normalize quotes
    text = text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
    
    return text.strip()

def generate_document_id(text):
    """
    Generate a unique ID for a document based on its content.
    
    Args:
        text: Document text
        
    Returns:
        A unique identifier for the document
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def allowed_file(filename):
    """
    Check if file has an allowed extension
    
    Args:
        filename (str): Name of the file
        
    Returns:
        bool: True if the file is allowed
    """
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def truncate_text(text, max_length=300):
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
    
    # Find the last sentence boundary before max_length
    end_pos = text[:max_length].rfind('.')
    if end_pos == -1:
        end_pos = text[:max_length].rfind(' ')
    
    if end_pos == -1:
        return text[:max_length] + '...'
    else:
        return text[:end_pos+1]

def format_metrics(metrics):
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

def load_sample_data():
    """
    Load sample scientific papers for demonstration.
    
    Returns:
        List of sample papers with text and metadata
    """
    samples = [
        {
            'id': 'sample1',
            'title': 'Advances in Neural Information Processing Systems',
            'text': """Abstract: This paper presents a novel approach to neural information processing systems. We investigate the effectiveness of deep learning architectures for natural language understanding tasks.

Introduction: Neural networks have revolutionized artificial intelligence in recent years. The ability to learn complex patterns from large datasets has enabled breakthroughs in various domains including computer vision, natural language processing, and reinforcement learning.

Methods: We propose a hybrid architecture that combines attention mechanisms with convolutional layers. Our model consists of three main components: a feature extraction module, an attention mechanism, and a prediction head. We train our model on a diverse dataset consisting of 50,000 examples.

Results: Experiments show that our approach outperforms existing methods by a significant margin. On the benchmark dataset, we achieve 95% accuracy compared to the previous state-of-the-art result of 89%. We also demonstrate faster convergence and better generalization to unseen data.

Discussion: The results confirm our hypothesis that attention mechanisms can effectively capture long-range dependencies in sequential data. The improved performance can be attributed to the model's ability to focus on relevant parts of the input while ignoring noise.

Conclusion: In this work, we presented a novel neural architecture for information processing. Our experiments demonstrate superior performance compared to existing methods. Future work will focus on extending the approach to multi-modal inputs and investigating more efficient training procedures."""
        },
        {
            'id': 'sample2',
            'title': 'Climate Change Impact Assessment: A Systematic Review',
            'text': """Abstract: Climate change poses significant threats to global ecosystems and human societies. This systematic review synthesizes findings from recent studies on climate change impacts across different regions and sectors.

Introduction: Climate change is one of the most pressing challenges facing humanity in the 21st century. Rising global temperatures, changing precipitation patterns, and increasing frequency of extreme weather events have far-reaching implications for natural ecosystems, agriculture, water resources, and human health.

Methods: We conducted a systematic literature review following PRISMA guidelines. We identified 435 peer-reviewed articles published between 2010 and 2023 that assessed climate change impacts. Studies were categorized by region, sector, methodology, and findings.

Results: Our analysis reveals substantial regional variations in climate change impacts. Tropical regions show the highest vulnerability, with projected agricultural yield reductions of 15-30% by 2050 under high emission scenarios. Coastal areas face increased risks from sea-level rise, with potential displacement of 280-350 million people globally by 2100. Economic analyses estimate global GDP losses of 7-18% under business-as-usual scenarios.

Discussion: The findings highlight the disproportionate impact of climate change on developing nations, despite their historically lower greenhouse gas emissions. Adaptation measures show promise in reducing vulnerability, but their effectiveness varies by context and implementation approach.

Conclusion: This review underscores the urgent need for both mitigation and adaptation strategies. The evidence points to significant and potentially irreversible impacts if global temperature increase exceeds 1.5°C above pre-industrial levels. Future research should focus on identifying effective adaptation strategies and understanding compound climate risks."""
        },
        {
            'id': 'sample3',
            'title': 'Quantum Computing: Recent Advancements and Future Directions',
            'text': """Abstract: Quantum computing has the potential to revolutionize computational capabilities by leveraging quantum mechanical phenomena. This paper reviews recent advancements in quantum computing hardware and algorithms, and discusses future research directions.

Introduction: Quantum computing represents a paradigm shift in computational technology. Unlike classical computers that use bits to represent either 0 or 1, quantum computers use quantum bits or "qubits" that can exist in superpositions of states, potentially enabling exponential speedups for certain problems.

Background: The theoretical foundations of quantum computing were established in the 1980s and 1990s with the development of algorithms like Shor's factoring algorithm and Grover's search algorithm. These algorithms demonstrated that quantum computers could, in principle, solve certain problems much faster than classical computers.

Methods: We surveyed the literature on quantum computing published in the last five years, focusing on hardware platforms, error correction techniques, and algorithmic advancements. We also conducted interviews with leading researchers in the field to identify emerging trends and challenges.

Results: Recent years have seen significant progress in quantum hardware development. Superconducting qubits have reached coherence times of over 100 microseconds, and systems with 50-100 qubits have been demonstrated. Trapped ion systems have achieved even longer coherence times, though with fewer qubits. On the algorithmic front, variational quantum algorithms have emerged as promising approaches for near-term quantum computers.

Discussion: Despite this progress, significant challenges remain. Quantum error correction requires substantial overhead in terms of physical qubits, and current hardware still suffers from high error rates. The path to fault-tolerant quantum computing will likely require breakthroughs in both hardware and error correction techniques.

Conclusion: Quantum computing is transitioning from a purely theoretical field to one with practical implementations. While large-scale, fault-tolerant quantum computers remain a future goal, near-term applications in fields like quantum chemistry and optimization may be realized within the next decade. Continued investment in fundamental research and interdisciplinary collaborations will be crucial for advancing the field."""
        }
    ]
    
    # Generate IDs for samples that don't have them
    for sample in samples:
        if 'id' not in sample:
            sample['id'] = generate_document_id(sample['text'])
    
    return samples