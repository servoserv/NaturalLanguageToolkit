import logging
import time
import torch
from summarizer.extractor import CitationAwareExtractor, BERTExtractor, TextRankExtractor
from summarizer.abstractor import TransformerAbstractor, LongDocumentAbstractor, ScienceSpecificAbstractor
from utils.preprocessor import preprocess_text, split_into_sections, extract_keywords

# Initialize logger
logger = logging.getLogger(__name__)

class HybridSummarizer:
    """
    Hybrid extractive-abstractive summarizer for scientific documents
    """
    
    def __init__(self, device='cpu', extractive_model='bert', abstractive_model='science'):
        """
        Initialize Hybrid Summarizer
        
        Args:
            device (str): Device to use (cpu or cuda)
            extractive_model (str): Type of extractive model ('bert', 'textrank', 'citation')
            abstractive_model (str): Type of abstractive model ('transformer', 'long', 'science')
        """
        self.device = device
        
        # Initialize extractive model
        logger.info(f"Initializing extractive model: {extractive_model}")
        if extractive_model == 'bert':
            self.extractor = BERTExtractor(device=device)
        elif extractive_model == 'citation':
            self.extractor = CitationAwareExtractor(device=device)
        else:  # textrank
            self.extractor = TextRankExtractor(use_scientific_features=True)
        
        # Initialize abstractive model
        logger.info(f"Initializing abstractive model: {abstractive_model}")
        if abstractive_model == 'long':
            self.abstractor = LongDocumentAbstractor(device=device)
        elif abstractive_model == 'science':
            self.abstractor = ScienceSpecificAbstractor(device=device)
        else:  # transformer
            self.abstractor = TransformerAbstractor(device=device, use_scientific_prompt=True)
        
        logger.info("Hybrid summarizer initialized successfully")
    
    def extract(self, text, max_sentences=10):
        """
        Perform extractive summarization
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of sentences to extract
            
        Returns:
            str: Extractive summary
        """
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Extract sentences using extractor
            if isinstance(self.extractor, CitationAwareExtractor):
                # For citation-aware extractor, try to extract citation contexts
                sections = split_into_sections(processed_text)
                citation_contexts = []
                # Look for "related work" or "background" sections which often contain citations
                for section_name in ['related work', 'background', 'introduction']:
                    if section_name in sections:
                        # Extract sentences that likely contain citations
                        from utils.preprocessor import extract_sentences, is_citation_sentence
                        section_sentences = extract_sentences(sections[section_name])
                        citation_sentences = [s for s in section_sentences if is_citation_sentence(s)]
                        citation_contexts.extend(citation_sentences)
                
                summary = self.extractor.get_summary(processed_text, max_sentences=max_sentences, 
                                                    citation_contexts=citation_contexts)
            else:
                summary = self.extractor.get_summary(processed_text, max_sentences=max_sentences)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            # Fallback to simple extraction if error occurs
            sentences = text.split('. ')
            return '. '.join(sentences[:max_sentences])
    
    def abstract(self, text, max_length=250):
        """
        Perform abstractive summarization
        
        Args:
            text (str): Input text
            max_length (int): Maximum length of summary in tokens
            
        Returns:
            str: Abstractive summary
        """
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Generate abstractive summary
            summary = self.abstractor.abstract(processed_text, max_length=max_length)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {e}")
            return "An error occurred during summarization. Please try with shorter text or different settings."
    
    def summarize(self, text, max_length=250, use_extraction_first=True):
        """
        Perform hybrid extractive-abstractive summarization
        
        Args:
            text (str): Input text
            max_length (int): Maximum length of final summary in words
            use_extraction_first (bool): Whether to extract before abstracting
            
        Returns:
            str: Hybrid summary
        """
        if not text or len(text) < 50:
            return text
        
        start_time = time.time()
        
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Extract keywords for focus
            keywords = extract_keywords(processed_text, top_n=5)
            logger.info(f"Extracted keywords: {keywords}")
            
            if use_extraction_first:
                # Two-stage approach: extract first, then abstract
                # Calculate extract length based on text length
                text_word_count = len(processed_text.split())
                extract_sentence_count = min(max(5, text_word_count // 200), 15)
                
                # Extract important sentences
                logger.info(f"Extracting {extract_sentence_count} sentences")
                extracted_summary = self.extract(processed_text, max_sentences=extract_sentence_count)
                
                # Generate abstractive summary from extracted text
                logger.info("Generating abstractive summary from extracted text")
                final_summary = self.abstract(extracted_summary, max_length=max_length)
            
            else:
                # Direct hybrid approach: enhance abstractive model with extracted info
                # Split into sections
                sections = split_into_sections(processed_text)
                
                # Create focused input with key sections
                focused_input = ""
                
                # Add abstract if available
                if 'abstract' in sections:
                    focused_input += f"ABSTRACT: {sections['abstract']}\n\n"
                
                # Add introduction if available
                if 'introduction' in sections:
                    # Extract key sentences from introduction
                    intro_sentences = self.extract(sections['introduction'], max_sentences=3)
                    focused_input += f"INTRODUCTION: {intro_sentences}\n\n"
                
                # Add results if available
                if 'results' in sections:
                    # Extract key sentences from results
                    results_sentences = self.extract(sections['results'], max_sentences=3)
                    focused_input += f"RESULTS: {results_sentences}\n\n"
                
                # Add conclusion if available
                if 'conclusion' in sections:
                    focused_input += f"CONCLUSION: {sections['conclusion']}\n\n"
                
                # If no sections were identified, use whole text
                if not focused_input:
                    # Extract key sentences from whole text
                    extracted_text = self.extract(processed_text, max_sentences=8)
                    focused_input = extracted_text
                
                # Generate abstractive summary from focused input
                logger.info("Generating hybrid summary")
                final_summary = self.abstract(focused_input, max_length=max_length)
            
            # Clean up the final summary
            final_summary = final_summary.replace('\n', ' ').replace('  ', ' ').strip()
            
            logger.info(f"Summary generation took {time.time() - start_time:.2f} seconds")
            return final_summary
        
        except Exception as e:
            logger.error(f"Error in hybrid summarization: {e}")
            # Fallback to basic extractive summary
            try:
                return self.extract(processed_text, max_sentences=3)
            except:
                return "An error occurred during summarization. Please try again with a shorter text."


class MultiDocumentSummarizer:
    """
    Summarizer for multiple scientific documents
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize Multi-Document Summarizer
        
        Args:
            device (str): Device to use (cpu or cuda)
        """
        self.device = device
        self.single_doc_summarizer = HybridSummarizer(device=device)
        logger.info("Multi-document summarizer initialized")
    
    def summarize_documents(self, documents, max_length=300):
        """
        Summarize multiple documents
        
        Args:
            documents (list): List of document texts
            max_length (int): Maximum length of final summary
            
        Returns:
            str: Multi-document summary
        """
        if not documents:
            return ""
        
        try:
            # For a single document, use the single-document summarizer
            if len(documents) == 1:
                return self.single_doc_summarizer.summarize(documents[0], max_length=max_length)
            
            # Generate individual summaries for each document
            individual_summaries = []
            for doc in documents:
                # Use shorter length for individual summaries
                doc_summary = self.single_doc_summarizer.summarize(doc, max_length=max_length // 2)
                individual_summaries.append(doc_summary)
            
            # Combine individual summaries
            combined_text = " ".join(individual_summaries)
            
            # Generate final summary from combined individual summaries
            final_summary = self.single_doc_summarizer.summarize(combined_text, max_length=max_length)
            
            return final_summary
        
        except Exception as e:
            logger.error(f"Error in multi-document summarization: {e}")
            return "An error occurred during multi-document summarization."
