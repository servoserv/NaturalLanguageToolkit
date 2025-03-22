import torch
import logging
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    LEDForConditionalGeneration,
    LEDTokenizer
)
from typing import List, Dict, Any, Optional, Union
from models import PaperStructure, Section

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ScientificSummarizer:
    """
    A hybrid extractive-abstractive summarizer specially designed for scientific papers.
    """
    
    def __init__(self):
        """Initialize the summarizer with required models."""
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        try:
            # BART for abstractive summarization
            logger.info("Loading BART model...")
            self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(self.device)
            
            # Pegasus as an alternative model
            logger.info("Loading Pegasus model...")
            self.pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
            self.pegasus_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum').to(self.device)
            
            # LED for long document summarization
            logger.info("Loading LED model...")
            self.led_tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
            self.led_model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384').to(self.device)
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.info("Falling back to extractive summarization only")
            self.models_loaded = False
        else:
            self.models_loaded = True
    
    def generate_extractive_summary(self, text: Union[str, PaperStructure], num_paragraphs: int = 3) -> str:
        """
        Generate an extractive summary by selecting the most important sentences.
        
        Args:
            text: Either raw text or a PaperStructure object
            num_paragraphs: Target number of paragraphs for the summary
            
        Returns:
            Extractive summary text
        """
        if isinstance(text, PaperStructure):
            # Process structured paper
            return self._generate_structured_extractive_summary(text, num_paragraphs)
        else:
            # Process raw text
            return self._generate_raw_extractive_summary(text, num_paragraphs)
    
    def _generate_raw_extractive_summary(self, text: str, num_paragraphs: int) -> str:
        """Generate extractive summary from raw text."""
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_paragraphs * 3:  # If text is already short
            return text
        
        # Calculate sentence scores based on position and term frequency
        sentence_scores = self._score_sentences(sentences)
        
        # Select top sentences to form paragraphs
        num_sentences = min(num_paragraphs * 3, len(sentences) // 2)  # 3 sentences per paragraph on average
        top_indices = sorted(range(len(sentence_scores)), 
                             key=lambda i: sentence_scores[i], 
                             reverse=True)[:num_sentences]
        
        # Sort indices by original position to maintain flow
        top_indices.sort()
        
        # Group sentences into paragraphs
        selected_sentences = [sentences[i] for i in top_indices]
        paragraphs = []
        current_paragraph = []
        
        for sentence in selected_sentences:
            current_paragraph.append(sentence)
            # Start a new paragraph after every 3 sentences or if a section marker is found
            if len(current_paragraph) >= 3 or re.search(r'^(Introduction|Methods|Results|Discussion|Conclusion)', sentence):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs)
    
    def _generate_structured_extractive_summary(self, paper: PaperStructure, num_paragraphs: int) -> str:
        """Generate extractive summary from structured paper."""
        summary_parts = []
        
        # Always include abstract if available
        if paper.abstract:
            summary_parts.append(paper.abstract)
        
        # Score and select from introduction
        if paper.introduction:
            intro_sentences = sent_tokenize(paper.introduction.content)
            intro_scores = self._score_sentences(intro_sentences)
            top_intro = [intro_sentences[i] for i in 
                         sorted(range(len(intro_scores)), 
                               key=lambda i: intro_scores[i], 
                               reverse=True)[:2]]
            if top_intro:
                summary_parts.append(' '.join(top_intro))
        
        # Score and select from results and discussion
        results_discussion = ''
        if paper.results:
            results_discussion += paper.results.content + ' '
        if paper.discussion:
            results_discussion += paper.discussion.content
            
        if results_discussion:
            rd_sentences = sent_tokenize(results_discussion)
            rd_scores = self._score_sentences(rd_sentences)
            top_rd = [rd_sentences[i] for i in 
                     sorted(range(len(rd_scores)), 
                           key=lambda i: rd_scores[i], 
                           reverse=True)[:min(5, len(rd_sentences))]]
            if top_rd:
                summary_parts.append(' '.join(top_rd))
        
        # Always include conclusion if available
        if paper.conclusion:
            conclusion_sentences = sent_tokenize(paper.conclusion.content)
            if len(conclusion_sentences) > 3:
                # If conclusion is long, summarize it
                conclusion_scores = self._score_sentences(conclusion_sentences)
                top_conclusion = [conclusion_sentences[i] for i in 
                                 sorted(range(len(conclusion_scores)), 
                                       key=lambda i: conclusion_scores[i], 
                                       reverse=True)[:3]]
                summary_parts.append(' '.join(top_conclusion))
            else:
                summary_parts.append(paper.conclusion.content)
        
        # Join all parts
        extractive_summary = '\n\n'.join(summary_parts)
        
        # Trim down if still too long
        if len(extractive_summary.split()) > num_paragraphs * 150:  # Assuming ~150 words per paragraph
            sentences = sent_tokenize(extractive_summary)
            scores = self._score_sentences(sentences)
            num_sentences = min(num_paragraphs * 3, len(sentences))
            top_sentences = [sentences[i] for i in 
                            sorted(range(len(scores)), 
                                  key=lambda i: scores[i], 
                                  reverse=True)[:num_sentences]]
            extractive_summary = ' '.join(top_sentences)
        
        return extractive_summary
    
    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score sentences based on various features:
        - Position in text
        - Length
        - Presence of key phrases
        - Named entity presence
        - Citation presence
        
        Returns:
            List of scores for each sentence
        """
        scores = []
        
        # Initial position weight (higher for first and last sentences)
        num_sentences = len(sentences)
        position_weights = []
        for i in range(num_sentences):
            # First few sentences get high weights
            if i < num_sentences * 0.1:
                position_weights.append(1.0 - (i / (num_sentences * 0.1) * 0.5))  # 1.0 to 0.5
            # Last few sentences get high weights
            elif i > num_sentences * 0.9:
                position_weights.append(0.5 + ((i - num_sentences * 0.9) / (num_sentences * 0.1) * 0.5))  # 0.5 to 1.0
            # Middle sentences get medium weights
            else:
                position_weights.append(0.5)
        
        # Key scientific phrases
        key_phrases = [
            r'\b(significant|important|novel|state-of-the-art|outperform)\b',
            r'\b(demonstrate|show|reveal|find|observe)\b',
            r'\b(conclude|summary|therefore|thus|overall)\b',
            r'\b(propose|introduce|present|develop)\b',
            r'\b(impact|implication|application|future work)\b'
        ]
        
        for i, sentence in enumerate(sentences):
            score = position_weights[i]
            
            # Adjust for sentence length (penalize very short or very long)
            words = len(sentence.split())
            if words < 5:
                score *= 0.5  # Penalize very short sentences
            elif words > 40:
                score *= 0.7  # Penalize very long sentences
            
            # Bonus for key phrases
            for phrase in key_phrases:
                if re.search(phrase, sentence, re.IGNORECASE):
                    score *= 1.2
                    break
            
            # Bonus for numeric content (often important in scientific papers)
            if re.search(r'\d+(?:\.\d+)?(?:%|\s*percent)?', sentence):
                score *= 1.1
            
            # Bonus for citations
            if re.search(r'\[\d+\]|\(\w+\s*et\s*al\..*?\d{4}\)|\(\w+,\s*\d{4}\)', sentence):
                score *= 1.15
            
            scores.append(score)
        
        return scores
    
    def generate_abstractive_summary(self, text: str, num_paragraphs: int = 3) -> str:
        """
        Generate an abstractive summary using pre-trained models.
        
        Args:
            text: Text to summarize
            num_paragraphs: Target number of paragraphs for the summary
            
        Returns:
            Abstractive summary text
        """
        if not self.models_loaded:
            # Fallback to extractive if models not loaded
            logger.warning("Models not loaded, falling back to extractive summarization")
            return self.generate_extractive_summary(text, num_paragraphs)
        
        # Preprocess text
        text = text.replace('\n', ' ').strip()
        
        # Choose model based on text length
        if len(text.split()) > 1024:
            # For long documents, use LED
            try:
                return self._generate_led_summary(text, num_paragraphs)
            except Exception as e:
                logger.error(f"LED summarization failed: {str(e)}")
                # Fall back to chunking with BART or Pegasus
                return self._generate_chunked_summary(text, num_paragraphs)
        else:
            # For shorter documents, try both BART and Pegasus and combine results
            try:
                bart_summary = self._generate_bart_summary(text, num_paragraphs)
                pegasus_summary = self._generate_pegasus_summary(text, num_paragraphs)
                
                # Combine summaries if both are generated successfully
                if bart_summary and pegasus_summary:
                    combined = self._combine_summaries([bart_summary, pegasus_summary])
                    return combined
                # Return whichever worked
                return bart_summary or pegasus_summary
            except Exception as e:
                logger.error(f"Abstractive summarization failed: {str(e)}")
                return self.generate_extractive_summary(text, num_paragraphs)
    
    def _generate_bart_summary(self, text: str, num_paragraphs: int) -> str:
        """Generate summary using BART model."""
        # Calculate max length based on paragraphs
        max_length = min(150 * num_paragraphs, 1024)  # ~150 words per paragraph, capped at 1024
        min_length = max(50 * num_paragraphs, 150)  # at least 50 words per paragraph or 150 words
        
        # Tokenize input
        inputs = self.bart_tokenizer(text, max_length=1024, truncation=True, return_tensors='pt').to(self.device)
        
        # Generate summary
        summary_ids = self.bart_model.generate(
            inputs.input_ids, 
            num_beams=4,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )
        
        summary = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Format into paragraphs
        return self._format_into_paragraphs(summary, num_paragraphs)
    
    def _generate_pegasus_summary(self, text: str, num_paragraphs: int) -> str:
        """Generate summary using Pegasus model."""
        # Calculate max length based on paragraphs
        max_length = min(150 * num_paragraphs, 512)  # ~150 words per paragraph, capped at 512
        min_length = max(50 * num_paragraphs, 150)  # at least 50 words per paragraph or 150 words
        
        # Tokenize input
        inputs = self.pegasus_tokenizer(text, max_length=1024, truncation=True, return_tensors='pt').to(self.device)
        
        # Generate summary
        summary_ids = self.pegasus_model.generate(
            inputs.input_ids,
            num_beams=4,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )
        
        summary = self.pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Format into paragraphs
        return self._format_into_paragraphs(summary, num_paragraphs)
    
    def _generate_led_summary(self, text: str, num_paragraphs: int) -> str:
        """Generate summary using LED model for long documents."""
        # Calculate max length based on paragraphs
        max_length = min(150 * num_paragraphs, 1024)  # ~150 words per paragraph, capped at 1024
        min_length = max(50 * num_paragraphs, 150)  # at least 50 words per paragraph or 150 words
        
        # Tokenize input
        inputs = self.led_tokenizer(text, max_length=16384, truncation=True, return_tensors='pt').to(self.device)
        
        # Generate summary
        summary_ids = self.led_model.generate(
            inputs.input_ids,
            global_attention_mask=torch.zeros_like(inputs.input_ids).to(self.device),
            num_beams=4,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )
        
        summary = self.led_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Format into paragraphs
        return self._format_into_paragraphs(summary, num_paragraphs)
    
    def _generate_chunked_summary(self, text: str, num_paragraphs: int) -> str:
        """
        Process long text by chunking it and summarizing each chunk.
        
        This handles documents that exceed the maximum input length of models.
        """
        sentences = sent_tokenize(text)
        
        # Calculate chunk size based on model limits
        chunk_size = 1000  # ~1000 words per chunk
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            if current_size + words > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = words
            else:
                current_chunk.append(sentence)
                current_size += words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            try:
                # Use BART for each chunk
                summary = self._generate_bart_summary(chunk, max(1, num_paragraphs // len(chunks)))
                chunk_summaries.append(summary)
            except Exception:
                # Fall back to extractive for this chunk
                extractive = self._generate_raw_extractive_summary(chunk, max(1, num_paragraphs // len(chunks)))
                chunk_summaries.append(extractive)
        
        # Combine chunk summaries
        combined = self._combine_summaries(chunk_summaries)
        
        # If combined summary is still too long, summarize again
        if len(combined.split()) > 200 * num_paragraphs:
            try:
                return self._generate_bart_summary(combined, num_paragraphs)
            except Exception:
                return self._format_into_paragraphs(combined, num_paragraphs)
        
        return combined
    
    def _combine_summaries(self, summaries: List[str]) -> str:
        """Intelligently combine multiple summaries, removing redundancies."""
        if len(summaries) == 1:
            return summaries[0]
        
        # Split into sentences
        all_sentences = []
        for summary in summaries:
            all_sentences.extend(sent_tokenize(summary))
        
        # Remove duplicates (based on similarity)
        unique_sentences = []
        for sentence in all_sentences:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
                
            is_duplicate = False
            for existing in unique_sentences:
                similarity = self._sentence_similarity(sentence, existing)
                if similarity > 0.7:  # Threshold for similarity
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sentences.append(sentence)
        
        # Group into coherent paragraphs
        paragraphs = []
        current_paragraph = []
        current_topic = None
        
        for sentence in unique_sentences:
            # Simple topic detection using key phrases
            topic_indicators = {
                'introduction': ['paper', 'study', 'research', 'introduce', 'propose'],
                'methods': ['method', 'approach', 'technique', 'algorithm', 'procedure', 'experiment'],
                'results': ['result', 'performance', 'accuracy', 'show', 'demonstrate', 'achieve'],
                'conclusion': ['conclude', 'summary', 'future', 'limitation', 'implication']
            }
            
            detected_topic = current_topic
            for topic, indicators in topic_indicators.items():
                for indicator in indicators:
                    if re.search(rf'\b{indicator}\b', sentence, re.IGNORECASE):
                        detected_topic = topic
                        break
                if detected_topic != current_topic:
                    break
            
            # Start new paragraph if topic changes or current is getting long
            if (detected_topic != current_topic and current_paragraph) or len(current_paragraph) >= 3:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentence]
                current_topic = detected_topic
            else:
                current_paragraph.append(sentence)
                current_topic = detected_topic
        
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs)
    
    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate simple similarity between two sentences based on word overlap.
        
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def _format_into_paragraphs(self, text: str, num_paragraphs: int) -> str:
        """Format text into a specified number of paragraphs."""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_paragraphs:
            # Not enough sentences to make requested paragraphs
            return text
        
        # Calculate sentences per paragraph
        sentences_per_paragraph = max(1, len(sentences) // num_paragraphs)
        
        paragraphs = []
        for i in range(0, len(sentences), sentences_per_paragraph):
            paragraph_sentences = sentences[i:i+sentences_per_paragraph]
            paragraphs.append(' '.join(paragraph_sentences))
            
            if len(paragraphs) >= num_paragraphs:
                break
        
        return '\n\n'.join(paragraphs)
    
    def generate_hybrid_summary(self, text: Union[str, PaperStructure], num_paragraphs: int = 3) -> str:
        """
        Generate a hybrid summary combining extractive and abstractive approaches.
        
        This approach:
        1. First extracts key sentences from the original text
        2. Then uses abstractive summarization on the extracted content
        3. Finally enhances the abstractive summary with any missing key information
        
        Args:
            text: Text or paper structure to summarize
            num_paragraphs: Target number of paragraphs
            
        Returns:
            Hybrid summary text
        """
        # Step 1: Generate extractive summary (handles both text string and PaperStructure)
        extractive_summary = self.generate_extractive_summary(text, num_paragraphs * 2)
        
        # Step 2: Generate abstractive summary from the extractive summary
        abstractive_summary = self.generate_abstractive_summary(extractive_summary, num_paragraphs)
        
        # If models not loaded, return the extractive summary
        if not self.models_loaded:
            return extractive_summary
        
        # Step 3: Analyze for key information that might be missing
        try:
            # Extract key information from the original text
            original_key_info = self._extract_key_information(text)
            
            # Extract key information from the abstractive summary
            abstract_key_info = self._extract_key_information(abstractive_summary)
            
            # Find missing key information
            missing_info = self._identify_missing_information(original_key_info, abstract_key_info)
            
            # If significant information is missing, enhance the summary
            if missing_info:
                enhanced_summary = self._enhance_summary_with_missing_info(abstractive_summary, missing_info)
                return enhanced_summary
        except Exception as e:
            logger.error(f"Error in hybrid enhancement: {str(e)}")
        
        # Return the abstractive summary if enhancement fails or isn't needed
        return abstractive_summary
    
    def _extract_key_information(self, text: Union[str, PaperStructure]) -> Dict[str, Any]:
        """
        Extract key information from text, such as:
        - Numerical results and percentages
        - Statistical significance indicators
        - Research contributions and findings
        - Methods and approaches used
        - Limitations and future work
        """
        if isinstance(text, PaperStructure):
            # Convert structure to text for processing
            combined_text = ""
            if text.abstract:
                combined_text += text.abstract + " "
            if text.introduction and text.introduction.content:
                combined_text += text.introduction.content + " "
            if text.results and text.results.content:
                combined_text += text.results.content + " "
            if text.discussion and text.discussion.content:
                combined_text += text.discussion.content + " "
            if text.conclusion and text.conclusion.content:
                combined_text += text.conclusion.content
            text_to_analyze = combined_text
        else:
            text_to_analyze = text
        
        key_info = {
            'numerical_results': [],
            'contributions': [],
            'methods': [],
            'limitations': []
        }
        
        # Extract numerical results and percentages
        numerical_pattern = r'(\d+(?:\.\d+)?(?:%|\s*percent|\s*times|\s*fold))'
        key_info['numerical_results'] = re.findall(numerical_pattern, text_to_analyze)
        
        # Extract sentences with contribution indicators
        contribution_indicators = [
            r'contribute[sd]?', r'propose[sd]?', r'introduce[sd]?', r'novel',
            r'outperform[se]?d?', r'improve[sd]?', r'better', r'advance[sd]?'
        ]
        
        sentences = sent_tokenize(text_to_analyze)
        for sentence in sentences:
            # Check for contribution statements
            for indicator in contribution_indicators:
                if re.search(rf'\b{indicator}\b', sentence, re.IGNORECASE):
                    key_info['contributions'].append(sentence)
                    break
            
            # Check for method descriptions
            if re.search(r'\b(method|approach|technique|algorithm|framework)\b', sentence, re.IGNORECASE):
                key_info['methods'].append(sentence)
            
            # Check for limitations
            if re.search(r'\b(limitation|drawback|weakness|future work|improve|challenge)\b', sentence, re.IGNORECASE):
                key_info['limitations'].append(sentence)
        
        # Remove duplicates
        for key in key_info:
            if isinstance(key_info[key], list):
                key_info[key] = list(set(key_info[key]))
        
        return key_info
    
    def _identify_missing_information(self, original_info: Dict[str, Any], abstract_info: Dict[str, Any]) -> Dict[str, Any]:
        """Identify key information present in the original but missing from the abstract."""
        missing_info = {}
        
        # Check numerical results
        missing_numericals = []
        for num in original_info['numerical_results']:
            if num not in abstract_info['numerical_results']:
                missing_numericals.append(num)
        if missing_numericals:
            missing_info['numerical_results'] = missing_numericals[:3]  # Limit to top 3
        
        # Check contributions
        orig_contributions = set(original_info['contributions'])
        abstract_contributions = set(abstract_info['contributions'])
        if orig_contributions and not abstract_contributions:
            # If abstract has no contributions but original does
            missing_info['contributions'] = list(orig_contributions)[:2]  # Limit to top 2
        
        # Check methods
        orig_methods = set(original_info['methods'])
        abstract_methods = set(abstract_info['methods'])
        if orig_methods and not abstract_methods:
            # If abstract has no methods but original does
            missing_info['methods'] = list(orig_methods)[:2]  # Limit to top 2
        
        # Check limitations
        orig_limitations = set(original_info['limitations'])
        abstract_limitations = set(abstract_info['limitations'])
        if orig_limitations and not abstract_limitations:
            # If abstract has no limitations but original does
            missing_info['limitations'] = list(orig_limitations)[:1]  # Limit to top 1
        
        return missing_info
    
    def _enhance_summary_with_missing_info(self, summary: str, missing_info: Dict[str, Any]) -> str:
        """Enhance the summary by incorporating missing key information."""
        enhanced_summary = summary
        
        # Add missing information based on type
        additional_content = []
        
        # Add important numerical results if missing
        if 'numerical_results' in missing_info and missing_info['numerical_results']:
            # Find sentences containing these numbers in the original text
            # For simplicity, we're just mentioning the numbers
            nums = ', '.join(missing_info['numerical_results'][:3])
            additional_content.append(f"The study reported key findings with metrics of {nums}.")
        
        # Add contributions if missing
        if 'contributions' in missing_info and missing_info['contributions']:
            for contribution in missing_info['contributions'][:1]:  # Add just the most important
                additional_content.append(contribution)
        
        # Add methods if missing
        if 'methods' in missing_info and missing_info['methods']:
            for method in missing_info['methods'][:1]:  # Add just the most important
                additional_content.append(method)
        
        # Add limitations if missing
        if 'limitations' in missing_info and missing_info['limitations']:
            for limitation in missing_info['limitations'][:1]:  # Add just the most important
                additional_content.append(limitation)
        
        # Add the additional content to the summary
        if additional_content:
            # Check if summary already ends with a complete sentence
            if summary.rstrip().endswith(('.', '!', '?')):
                enhanced_summary = summary + "\n\n" + " ".join(additional_content)
            else:
                enhanced_summary = summary + ". " + " ".join(additional_content)
        
        return enhanced_summary
