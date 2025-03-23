import logging
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel
from utils.preprocessor import extract_sentences, clean_sentence, is_citation_sentence, split_into_sections

# Initialize logger
logger = logging.getLogger(__name__)

class TextRankExtractor:
    """Extractive summarizer using TextRank algorithm"""
    
    def __init__(self, use_scientific_features=True):
        """
        Initialize TextRank Extractor
        
        Args:
            use_scientific_features (bool): Whether to use scientific-specific features
        """
        self.use_scientific_features = use_scientific_features
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        logger.info("Initialized TextRank Extractor")
    
    def extract(self, text, max_sentences=5):
        """
        Extract most important sentences from text
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of sentences to extract
            
        Returns:
            list: List of extracted sentences
        """
        if not text:
            return []
        
        # Extract sentences
        sentences = extract_sentences(text)
        if len(sentences) <= max_sentences:
            return sentences
        
        # Clean sentences
        cleaned_sentences = [clean_sentence(s) for s in sentences]
        
        # Create sentence vectors using TF-IDF
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        except ValueError as e:
            logger.error(f"Error computing similarity matrix: {e}")
            # Fallback to simple extraction if vectorization fails
            return sentences[:max_sentences]
        
        # Apply TextRank algorithm
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Add scientific-specific features if enabled
        if self.use_scientific_features:
            # Boost sentences with citations
            for i, sentence in enumerate(sentences):
                if is_citation_sentence(sentence):
                    scores[i] = scores[i] * 1.2  # Boost citation sentences
            
            # Boost sentences from important sections
            sections = split_into_sections(text)
            section_boosts = {
                'abstract': 1.5,
                'introduction': 1.2,
                'conclusion': 1.4,
                'results': 1.3,
                'discussion': 1.2
            }
            
            for section_name, section_text in sections.items():
                if section_name in section_boosts:
                    boost_factor = section_boosts[section_name]
                    
                    # Find sentences that belong to this section
                    for i, sentence in enumerate(sentences):
                        if sentence in section_text:
                            scores[i] = scores[i] * boost_factor
        
        # Select top sentences
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        
        # Extract top sentences in document order
        top_sentence_indices = [ranked_sentences[i][1] for i in range(min(max_sentences, len(ranked_sentences)))]
        top_sentence_indices.sort()
        
        return [sentences[i] for i in top_sentence_indices]
    
    def get_summary(self, text, max_sentences=5):
        """
        Get extractive summary
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of sentences in summary
            
        Returns:
            str: Extractive summary
        """
        extracted_sentences = self.extract(text, max_sentences)
        return ' '.join(extracted_sentences)


class BERTExtractor:
    """Extractive summarizer using BERT embeddings"""
    
    def __init__(self, model_name='allenai/scibert_scivocab_uncased', device='cpu'):
        """
        Initialize BERT Extractor
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to use (cpu or cuda)
        """
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            logger.info(f"Loaded BERT model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            # Fallback to simpler model if main model fails
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)
            logger.info("Loaded fallback DistilBERT model")
    
    def get_sentence_embeddings(self, sentences):
        """
        Get BERT embeddings for sentences
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            np.ndarray: Array of sentence embeddings
        """
        embeddings = []
        
        for sentence in sentences:
            # Tokenize and get embeddings
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token embedding as sentence embedding
            embeddings.append(outputs.last_hidden_state[0, 0, :].cpu().numpy())
        
        return np.array(embeddings)
    
    def extract(self, text, max_sentences=5):
        """
        Extract most important sentences from text using BERT embeddings
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of sentences to extract
            
        Returns:
            list: List of extracted sentences
        """
        if not text:
            return []
        
        # Extract and clean sentences
        sentences = extract_sentences(text)
        if len(sentences) <= max_sentences:
            return sentences
        
        cleaned_sentences = [clean_sentence(s) for s in sentences]
        
        try:
            # Get sentence embeddings
            embeddings = self.get_sentence_embeddings(cleaned_sentences)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Apply PageRank algorithm
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Select top sentences
            ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
            
            # Extract top sentences in document order
            top_sentence_indices = [ranked_sentences[i][1] for i in range(min(max_sentences, len(ranked_sentences)))]
            top_sentence_indices.sort()
            
            return [sentences[i] for i in top_sentence_indices]
        
        except Exception as e:
            logger.error(f"Error in BERT extraction: {e}")
            # Fallback to first max_sentences if BERT extraction fails
            return sentences[:max_sentences]
    
    def get_summary(self, text, max_sentences=5):
        """
        Get extractive summary using BERT
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of sentences in summary
            
        Returns:
            str: Extractive summary
        """
        extracted_sentences = self.extract(text, max_sentences)
        return ' '.join(extracted_sentences)


class CitationAwareExtractor:
    """Extractive summarizer that weighs sentences based on citation information"""
    
    def __init__(self, base_extractor='bert', device='cpu'):
        """
        Initialize Citation-Aware Extractor
        
        Args:
            base_extractor (str): Base extractor to use ('bert' or 'textrank')
            device (str): Device to use (cpu or cuda)
        """
        self.device = device
        self.citation_boost = 1.5  # Boost factor for sentences containing citations
        
        if base_extractor == 'bert':
            self.extractor = BERTExtractor(device=device)
        else:
            self.extractor = TextRankExtractor(use_scientific_features=True)
        
        logger.info(f"Initialized Citation-Aware Extractor with {base_extractor} base")
    
    def extract(self, text, max_sentences=5, citation_contexts=None):
        """
        Extract most important sentences with citation awareness
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of sentences to extract
            citation_contexts (list): List of citation contexts if available
            
        Returns:
            list: List of extracted sentences
        """
        if not text:
            return []
        
        # Extract and clean sentences
        sentences = extract_sentences(text)
        if len(sentences) <= max_sentences:
            return sentences
        
        cleaned_sentences = [clean_sentence(s) for s in sentences]
        
        # Use base extractor to get initial scores
        if isinstance(self.extractor, BERTExtractor):
            embeddings = self.extractor.get_sentence_embeddings(cleaned_sentences)
            similarity_matrix = cosine_similarity(embeddings)
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
        else:
            tfidf_matrix = self.extractor.tfidf_vectorizer.fit_transform(cleaned_sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
        
        # Boost sentences with citations
        for i, sentence in enumerate(sentences):
            if is_citation_sentence(sentence):
                scores[i] = scores[i] * self.citation_boost
        
        # Boost sentences from important sections
        sections = split_into_sections(text)
        section_boosts = {
            'abstract': 2.0,
            'introduction': 1.3,
            'conclusion': 1.8,
            'results': 1.5,
            'discussion': 1.3,
            'method': 1.2,
            'methodology': 1.2
        }
        
        for section_name, section_text in sections.items():
            if section_name in section_boosts:
                boost_factor = section_boosts[section_name]
                
                # Find sentences that belong to this section
                for i, sentence in enumerate(sentences):
                    if sentence in section_text:
                        scores[i] = scores[i] * boost_factor
        
        # If citation contexts are provided, boost sentences similar to citation contexts
        if citation_contexts and len(citation_contexts) > 0:
            try:
                # Get embeddings for citation contexts
                if isinstance(self.extractor, BERTExtractor):
                    context_embeddings = self.extractor.get_sentence_embeddings(citation_contexts)
                    sentence_embeddings = embeddings
                else:
                    # Use TF-IDF for TextRank extractor
                    all_texts = cleaned_sentences + citation_contexts
                    tfidf_matrix = self.extractor.tfidf_vectorizer.fit_transform(all_texts)
                    sentence_embeddings = tfidf_matrix[:len(cleaned_sentences)]
                    context_embeddings = tfidf_matrix[len(cleaned_sentences):]
                
                # Compute similarity between sentences and citation contexts
                cross_similarities = cosine_similarity(sentence_embeddings, context_embeddings)
                
                # Boost sentences that are similar to citation contexts
                for i in range(len(sentences)):
                    # Take max similarity to any citation context
                    max_similarity = np.max(cross_similarities[i])
                    # Apply boost based on similarity
                    similarity_boost = 1.0 + max_similarity
                    scores[i] = scores[i] * similarity_boost
            
            except Exception as e:
                logger.error(f"Error processing citation contexts: {e}")
        
        # Select top sentences
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        
        # Extract top sentences in document order
        top_sentence_indices = [ranked_sentences[i][1] for i in range(min(max_sentences, len(ranked_sentences)))]
        top_sentence_indices.sort()
        
        return [sentences[i] for i in top_sentence_indices]
    
    def get_summary(self, text, max_sentences=5, citation_contexts=None):
        """
        Get citation-aware extractive summary
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of sentences in summary
            citation_contexts (list): List of citation contexts if available
            
        Returns:
            str: Extractive summary
        """
        extracted_sentences = self.extract(text, max_sentences, citation_contexts)
        return ' '.join(extracted_sentences)
