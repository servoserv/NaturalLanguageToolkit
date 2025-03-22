import logging
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Download needed NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize logger
logger = logging.getLogger(__name__)

def calculate_rouge(candidate, reference, use_stemmer=True):
    """
    Calculate ROUGE scores between candidate and reference summaries
    
    Args:
        candidate (str): Candidate summary
        reference (str): Reference summary
        use_stemmer (bool): Whether to use stemming
        
    Returns:
        dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    if not candidate or not reference:
        logger.warning("Empty candidate or reference for ROUGE calculation")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }
    
    try:
        # Create scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        
        # Calculate scores
        scores = scorer.score(reference, candidate)
        
        # Extract F1 scores
        result = {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores: {e}")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }

def calculate_bleu(candidate, reference):
    """
    Calculate BLEU score between candidate and reference summaries
    
    Args:
        candidate (str): Candidate summary
        reference (str): Reference summary
        
    Returns:
        float: BLEU score
    """
    if not candidate or not reference:
        logger.warning("Empty candidate or reference for BLEU calculation")
        return 0.0
    
    try:
        # Tokenize the sentences
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        
        # Use smoothing to avoid zero scores when there's no n-gram overlap
        smoothie = SmoothingFunction().method1
        
        # Calculate BLEU score
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
        
        return score
    
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_metrics(candidate_summaries, reference_summaries):
    """
    Calculate evaluation metrics for multiple summaries
    
    Args:
        candidate_summaries (list): List of candidate summaries
        reference_summaries (list): List of reference summaries
        
    Returns:
        dict: Dictionary containing average metrics
    """
    if len(candidate_summaries) != len(reference_summaries):
        logger.error("Mismatch in the number of candidate and reference summaries")
        return {}
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    
    for candidate, reference in zip(candidate_summaries, reference_summaries):
        rouge_scores = calculate_rouge(candidate, reference)
        rouge1_scores.append(rouge_scores['rouge1'])
        rouge2_scores.append(rouge_scores['rouge2'])
        rougeL_scores.append(rouge_scores['rougeL'])
        
        bleu_score = calculate_bleu(candidate, reference)
        bleu_scores.append(bleu_score)
    
    metrics = {
        'rouge1_avg': np.mean(rouge1_scores),
        'rouge2_avg': np.mean(rouge2_scores),
        'rougeL_avg': np.mean(rougeL_scores),
        'bleu_avg': np.mean(bleu_scores),
        'rouge1_scores': rouge1_scores,
        'rouge2_scores': rouge2_scores,
        'rougeL_scores': rougeL_scores,
        'bleu_scores': bleu_scores
    }
    
    return metrics

def benchmark_models(models, test_data, references):
    """
    Benchmark multiple summarization models
    
    Args:
        models (dict): Dictionary mapping model names to model objects
        test_data (list): List of texts to summarize
        references (list): List of reference summaries
        
    Returns:
        dict: Dictionary containing benchmark results
    """
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Benchmarking model: {model_name}")
        
        start_time = time.time()
        
        # Generate summaries
        summaries = []
        for text in test_data:
            try:
                summary = model.summarize(text)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error generating summary with {model_name}: {e}")
                summaries.append("")
        
        # Calculate metrics
        metrics = calculate_metrics(summaries, references)
        
        # Calculate time
        total_time = time.time() - start_time
        avg_time_per_doc = total_time / len(test_data) if test_data else 0
        
        # Store results
        results[model_name] = {
            'metrics': metrics,
            'total_time': total_time,
            'avg_time_per_doc': avg_time_per_doc
        }
    
    return results

def plot_metrics_comparison(results, metric='rouge1_avg', save_path=None):
    """
    Plot comparison of models based on a specific metric
    
    Args:
        results (dict): Dictionary containing benchmark results
        metric (str): Metric to compare
        save_path (str): Path to save the plot, if None, displays the plot
    """
    model_names = list(results.keys())
    metric_values = [results[model]['metrics'][metric] * 100 for model in model_names]  # Convert to percentage
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metric_values, color='skyblue')
    plt.xlabel('Models')
    
    # Format metric name for display
    metric_display = metric.replace('_avg', '').upper()
    plt.ylabel(f'{metric_display} Score (%)')
    
    plt.title(f'Comparison of Models by {metric_display} Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_time_comparison(results, save_path=None):
    """
    Plot comparison of models based on average processing time
    
    Args:
        results (dict): Dictionary containing benchmark results
        save_path (str): Path to save the plot, if None, displays the plot
    """
    model_names = list(results.keys())
    times = [results[model]['avg_time_per_doc'] for model in model_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, times, color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Average Time per Document (seconds)')
    plt.title('Comparison of Models by Processing Time')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def create_leaderboard(results):
    """
    Create a leaderboard DataFrame from benchmark results
    
    Args:
        results (dict): Dictionary containing benchmark results
        
    Returns:
        pd.DataFrame: Leaderboard DataFrame
    """
    data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        data.append({
            'Model': model_name,
            'ROUGE-1': f"{metrics['rouge1_avg'] * 100:.1f}",
            'ROUGE-2': f"{metrics['rouge2_avg'] * 100:.1f}",
            'ROUGE-L': f"{metrics['rougeL_avg'] * 100:.1f}",
            'BLEU': f"{metrics['bleu_avg'] * 100:.1f}",
            'Avg Time (s)': f"{result['avg_time_per_doc']:.2f}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by ROUGE-L score (primary) and ROUGE-2 score (secondary)
    df['ROUGE-L_sort'] = df['ROUGE-L'].astype(float)
    df['ROUGE-2_sort'] = df['ROUGE-2'].astype(float)
    df = df.sort_values(['ROUGE-L_sort', 'ROUGE-2_sort'], ascending=False)
    df = df.drop(['ROUGE-L_sort', 'ROUGE-2_sort'], axis=1)
    
    # Reset index and add rank
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df = df.rename_axis('Rank')
    
    return df

def evaluate_information_content(predicted_summaries, reference_summaries, key_information):
    """
    Evaluate how well summaries capture key information elements
    
    Args:
        predicted_summaries (list): List of predicted summaries
        reference_summaries (list): List of reference summaries
        key_information (list): List of key information elements to check for
        
    Returns:
        dict: Information content metrics
    """
    pred_info_present = []
    ref_info_present = []
    
    for pred, ref in zip(predicted_summaries, reference_summaries):
        # Check which key information elements are present in each summary
        pred_info = [int(any(key.lower() in pred.lower() for key in info)) for info in key_information]
        ref_info = [int(any(key.lower() in ref.lower() for key in info)) for info in key_information]
        
        pred_info_present.append(pred_info)
        ref_info_present.append(ref_info)
    
    # Calculate precision, recall, F1 for information content
    y_true = np.array(ref_info_present).flatten()
    y_pred = np.array(pred_info_present).flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'information_precision': precision,
        'information_recall': recall,
        'information_f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }
