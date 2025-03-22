import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration, 
    T5Tokenizer,
    LEDForConditionalGeneration,
    LEDTokenizer
)
from utils.preprocessor import preprocess_text, split_into_sections

# Initialize logger
logger = logging.getLogger(__name__)

class TransformerAbstractor:
    """Abstractive summarizer using transformer models"""
    
    def __init__(self, model_name='google/pegasus-arxiv', device='cpu', use_scientific_prompt=True):
        """
        Initialize Transformer Abstractor
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to use (cpu or cuda)
            use_scientific_prompt (bool): Whether to use scientific-specific prompts
        """
        self.device = device
        self.model_name = model_name
        self.use_scientific_prompt = use_scientific_prompt
        
        # Special handling based on model type
        try:
            if 'pegasus' in model_name.lower():
                self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
                self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
            elif 'bart' in model_name.lower():
                self.tokenizer = BartTokenizer.from_pretrained(model_name)
                self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
            elif 't5' in model_name.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
            elif 'led' in model_name.lower():
                self.tokenizer = LEDTokenizer.from_pretrained(model_name)
                self.model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
            else:
                # Generic loading for other models
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            
            logger.info(f"Loaded transformer model: {model_name}")
            
            # Set model-specific parameters
            if 'pegasus' in model_name.lower():
                self.max_length = 1024
                self.min_length = 150
            elif 'bart' in model_name.lower():
                self.max_length = 1024
                self.min_length = 100
            elif 't5' in model_name.lower():
                self.max_length = 512
                self.min_length = 50
            elif 'led' in model_name.lower():
                self.max_length = 4096
                self.min_length = 150
            else:
                self.max_length = 1024
                self.min_length = 100
                
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            # Fallback to smaller model if loading fails
            self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
            self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail').to(device)
            self.max_length = 512
            self.min_length = 50
            logger.info("Loaded fallback Pegasus CNN/DailyMail model")
    
    def prepare_input(self, text):
        """
        Prepare input for the model
        
        Args:
            text (str): Input text
            
        Returns:
            str: Prepared input text
        """
        # Preprocess text
        text = preprocess_text(text)
        
        # Handle T5 models that need 'summarize:' prefix
        if 't5' in self.model_name.lower() and not text.startswith('summarize:'):
            text = f"summarize: {text}"
        
        # Use scientific-specific prompts if enabled
        if self.use_scientific_prompt:
            # Split text into sections
            sections = split_into_sections(text)
            
            # Check if key sections exist
            has_introduction = any(section in sections for section in ['introduction', 'background'])
            has_methods = any(section in sections for section in ['method', 'methodology', 'methods'])
            has_results = 'results' in sections
            has_conclusion = 'conclusion' in sections
            
            # If text has a proper scientific structure, use structured prompt
            if has_introduction and (has_methods or has_results) and has_conclusion:
                # Extract key sections
                introduction = sections.get('introduction', sections.get('background', ''))
                methods = sections.get('method', sections.get('methodology', sections.get('methods', '')))
                results = sections.get('results', '')
                conclusion = sections.get('conclusion', '')
                
                # Create a structured prompt
                if 't5' in self.model_name.lower():
                    structured_prompt = f"summarize: The paper introduces {introduction[:200]}... "
                    if methods:
                        structured_prompt += f"The methodology involves {methods[:200]}... "
                    if results:
                        structured_prompt += f"The results show {results[:200]}... "
                    if conclusion:
                        structured_prompt += f"The paper concludes that {conclusion[:200]}..."
                    
                    text = structured_prompt
                else:
                    # For non-T5 models
                    pass  # Keep original text as is
        
        return text
    
    def abstract(self, text, max_length=None):
        """
        Generate abstractive summary using transformer model
        
        Args:
            text (str): Input text
            max_length (int): Maximum length of summary in tokens
            
        Returns:
            str: Abstractive summary
        """
        if not text:
            return ""
        
        try:
            # Prepare input
            prepared_text = self.prepare_input(text)
            
            # Set maximum output length
            output_max_length = min(max_length if max_length else self.max_length, self.max_length)
            output_min_length = min(self.min_length, output_max_length // 2)
            
            # Tokenize input
            inputs = self.tokenizer(prepared_text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True).to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=output_max_length,
                min_length=output_min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Clean up summary
            summary = summary.replace('summarize:', '').strip()
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating abstractive summary: {e}")
            return "Error generating summary. Please try again with a shorter text."


class LongDocumentAbstractor:
    """Abstractive summarizer for long documents"""
    
    def __init__(self, model_name='allenai/led-base-16384', device='cpu', chunk_size=4000, overlap=500):
        """
        Initialize Long Document Abstractor
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to use (cpu or cuda)
            chunk_size (int): Size of text chunks to process
            overlap (int): Overlap between chunks
        """
        self.device = device
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        try:
            # Try to load a long document model
            if 'led' in model_name.lower():
                self.tokenizer = LEDTokenizer.from_pretrained(model_name)
                self.model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
                logger.info(f"Loaded LED model for long documents: {model_name}")
            else:
                # Fallback to other models with chunking strategy
                self.base_abstractor = TransformerAbstractor(model_name=model_name, device=device)
                logger.info(f"Using chunking strategy with model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading long document model: {e}")
            # Fallback to standard abstractor with chunking
            self.base_abstractor = TransformerAbstractor(model_name='facebook/bart-large-cnn', device=device)
            logger.info("Loaded fallback BART model with chunking strategy")
    
    def chunk_text(self, text):
        """
        Split text into chunks for processing
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of text chunks
        """
        # Split text into sentences
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add period back if it was removed during split
            if not sentence.endswith('.'):
                sentence = sentence + '.'
            
            # If adding this sentence exceeds chunk size, store current chunk and start new one
            if len(current_chunk) + len(sentence) > self.chunk_size:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.overlap)
                current_chunk = current_chunk[overlap_start:] + sentence
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def abstract(self, text, max_length=None):
        """
        Generate abstractive summary for long documents
        
        Args:
            text (str): Input text
            max_length (int): Maximum length of summary in tokens
            
        Returns:
            str: Abstractive summary
        """
        if not text:
            return ""
        
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Set default max_length if not provided
            if not max_length:
                max_length = 250  # default token length
            
            # Check if using LED model directly
            if hasattr(self, 'model'):
                # LED model can handle long inputs directly
                inputs = self.tokenizer(processed_text, return_tensors="pt", max_length=16384, truncation=True).to(self.device)
                
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min(50, max_length // 2),
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                return summary
            
            else:
                # Using chunking strategy with base abstractor
                # Split text into chunks
                chunks = self.chunk_text(processed_text)
                
                if not chunks:
                    return ""
                
                # Process each chunk
                chunk_summaries = []
                for chunk in chunks:
                    chunk_summary = self.base_abstractor.abstract(chunk, max_length=max_length // 2)
                    chunk_summaries.append(chunk_summary)
                
                # Combine chunk summaries
                combined_summary = " ".join(chunk_summaries)
                
                # Generate final summary from combined chunk summaries
                if len(combined_summary.split()) > max_length:
                    final_summary = self.base_abstractor.abstract(combined_summary, max_length=max_length)
                    return final_summary
                else:
                    return combined_summary
        
        except Exception as e:
            logger.error(f"Error generating long document summary: {e}")
            return "Error generating summary for long document. Please try with a shorter text."


class ScienceSpecificAbstractor:
    """Abstractive summarizer specific for scientific articles"""
    
    def __init__(self, model_name='google/pegasus-arxiv', device='cpu'):
        """
        Initialize Science-Specific Abstractor
        
        Args:
            model_name (str): HuggingFace model name, preferably science-specific
            device (str): Device to use (cpu or cuda)
        """
        self.device = device
        
        # Preferred models for scientific summarization
        scientific_models = [
            'google/pegasus-arxiv',
            'google/pegasus-pubmed',
            'allenai/led-base-16384',
            'facebook/bart-large-xsum'
        ]
        
        # Use provided model if specified, otherwise try to load a scientific model
        if model_name not in scientific_models:
            for sci_model in scientific_models:
                try:
                    if 'led' in sci_model.lower():
                        self.long_abstractor = LongDocumentAbstractor(model_name=sci_model, device=device)
                        self.model_type = 'led'
                        logger.info(f"Loaded LED model for scientific articles: {sci_model}")
                        break
                    else:
                        self.abstractor = TransformerAbstractor(model_name=sci_model, device=device, use_scientific_prompt=True)
                        self.model_type = 'transformer'
                        logger.info(f"Loaded scientific model: {sci_model}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {sci_model}: {e}")
                    continue
            else:
                # If all scientific models fail, fall back to provided model
                try:
                    if 'led' in model_name.lower():
                        self.long_abstractor = LongDocumentAbstractor(model_name=model_name, device=device)
                        self.model_type = 'led'
                    else:
                        self.abstractor = TransformerAbstractor(model_name=model_name, device=device, use_scientific_prompt=True)
                        self.model_type = 'transformer'
                    logger.info(f"Loaded fallback model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load any model: {e}")
                    # Use a very basic fallback
                    self.abstractor = TransformerAbstractor(model_name='facebook/bart-large-cnn', device=device)
                    self.model_type = 'transformer'
                    logger.info("Loaded very basic fallback model: facebook/bart-large-cnn")
        else:
            # Load the specified model
            try:
                if 'led' in model_name.lower():
                    self.long_abstractor = LongDocumentAbstractor(model_name=model_name, device=device)
                    self.model_type = 'led'
                else:
                    self.abstractor = TransformerAbstractor(model_name=model_name, device=device, use_scientific_prompt=True)
                    self.model_type = 'transformer'
                logger.info(f"Loaded specified model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load specified model {model_name}: {e}")
                # Use a basic fallback
                self.abstractor = TransformerAbstractor(model_name='facebook/bart-large-cnn', device=device)
                self.model_type = 'transformer'
                logger.info("Loaded basic fallback model: facebook/bart-large-cnn")
    
    def preprocess_scientific_text(self, text):
        """
        Preprocess scientific text for better summarization
        
        Args:
            text (str): Input scientific text
            
        Returns:
            str: Preprocessed text
        """
        # Basic preprocessing
        processed_text = preprocess_text(text)
        
        # Split into sections
        sections = split_into_sections(processed_text)
        
        # Format text focusing on key sections
        formatted_text = ""
        
        # Add title if it seems to be at the beginning
        first_lines = processed_text.split('\n')[:2]
        if first_lines and len(first_lines[0]) < 200:  # Title is usually short
            formatted_text += first_lines[0] + "\n\n"
        
        # Add key sections in order of importance for summarization
        important_sections = ['abstract', 'introduction', 'conclusion', 'results', 'discussion', 
                            'method', 'methodology', 'methods', 'background']
        
        for section in important_sections:
            if section in sections:
                formatted_text += f"{section.upper()}:\n{sections[section]}\n\n"
        
        # If no sections were found, return original preprocessed text
        if not formatted_text:
            return processed_text
        
        return formatted_text
    
    def abstract(self, text, max_length=None):
        """
        Generate science-specific abstractive summary
        
        Args:
            text (str): Input scientific text
            max_length (int): Maximum length of summary in tokens
            
        Returns:
            str: Scientific abstractive summary
        """
        if not text:
            return ""
        
        try:
            # Preprocess scientific text
            processed_text = self.preprocess_scientific_text(text)
            
            # Set default max_length if not provided
            if not max_length:
                max_length = 250
            
            # Generate summary using appropriate model
            if self.model_type == 'led':
                summary = self.long_abstractor.abstract(processed_text, max_length=max_length)
            else:
                summary = self.abstractor.abstract(processed_text, max_length=max_length)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating scientific summary: {e}")
            return "Error generating scientific summary. Please try again with a shorter text."
