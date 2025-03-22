"""
Simplified evaluation metrics for text summarization.
Since we don't have access to rouge or nltk, we implement
simplified versions of these metrics.
"""

import re
from collections import Counter


def calculate_rouge(candidate, reference):
    """
    Calculate a simplified version of ROUGE-N scores.
    Returns basic precision, recall, and F1 for unigrams and bigrams.
    
    Args:
        candidate: Candidate summary text
        reference: Reference summary text
        
    Returns:
        Dictionary with rouge1_f, rouge2_f scores
    """
    # Tokenize to words
    def tokenize(text):
        # Simple tokenization - split on whitespace and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    # Generate n-grams
    def get_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    # Calculate F1 score
    def calculate_f1(precision, recall):
        if precision == 0 or recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)
    
    # Process texts
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    
    # Calculate unigram (ROUGE-1) metrics
    candidate_unigrams = Counter(candidate_tokens)
    reference_unigrams = Counter(reference_tokens)
    
    # Count overlapping unigrams
    overlap_count = sum((candidate_unigrams & reference_unigrams).values())
    
    # Calculate precision and recall
    precision_1 = overlap_count / max(1, len(candidate_tokens))
    recall_1 = overlap_count / max(1, len(reference_tokens))
    rouge1_f = calculate_f1(precision_1, recall_1)
    
    # Calculate bigram (ROUGE-2) metrics
    candidate_bigrams = Counter(get_ngrams(candidate_tokens, 2))
    reference_bigrams = Counter(get_ngrams(reference_tokens, 2))
    
    # Count overlapping bigrams
    overlap_bigrams = sum((candidate_bigrams & reference_bigrams).values())
    
    # Calculate precision and recall for bigrams
    precision_2 = overlap_bigrams / max(1, len(candidate_bigrams))
    recall_2 = overlap_bigrams / max(1, len(reference_bigrams))
    rouge2_f = calculate_f1(precision_2, recall_2)
    
    # Calculate ROUGE-L by using the longest common subsequence
    # This is simplified and not the exact ROUGE-L implementation
    lcs_length = longest_common_subsequence(candidate_tokens, reference_tokens)
    precision_l = lcs_length / max(1, len(candidate_tokens))
    recall_l = lcs_length / max(1, len(reference_tokens))
    rouge_l = calculate_f1(precision_l, recall_l)
    
    return {
        'rouge1_f': rouge1_f,
        'rouge2_f': rouge2_f,
        'rougeL_f': rouge_l
    }


def longest_common_subsequence(X, Y):
    """
    Calculate the longest common subsequence length between two lists.
    This is a helper function for ROUGE-L.
    """
    m, n = len(X), len(Y)
    
    # Create a table to store the LCS lengths
    L = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the table
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    
    # Return the length of LCS
    return L[m][n]


def calculate_bleu(candidate, reference):
    """
    Calculate a simplified version of BLEU score.
    
    Args:
        candidate: Candidate summary text
        reference: Reference summary text
        
    Returns:
        BLEU score as a float
    """
    # Tokenize texts
    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    
    # Calculate n-gram precision for n=1,2,3,4
    max_n = min(4, len(candidate_tokens), len(reference_tokens))
    if max_n == 0:
        return 0
    
    precisions = []
    
    # Generate n-grams
    def get_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    for n in range(1, max_n + 1):
        candidate_ngrams = Counter(get_ngrams(candidate_tokens, n))
        reference_ngrams = Counter(get_ngrams(reference_tokens, n))
        
        # Count matching n-grams
        matches = sum((candidate_ngrams & reference_ngrams).values())
        total = max(1, sum(candidate_ngrams.values()))
        
        precisions.append(matches / total)
    
    # Calculate geometric mean of precisions
    if not precisions:
        return 0
    
    # Simplified BLEU without brevity penalty
    bleu = 1.0
    for p in precisions:
        bleu *= p
    
    return bleu ** (1.0 / len(precisions))


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