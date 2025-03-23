import re
import random
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimplifiedSummarizer:
    """
    A simplified summarizer implementation that doesn't rely on PyTorch or transformers.
    Uses basic NLP techniques to generate extractive and abstractive summaries.
    """
    
    def __init__(self):
        """Initialize the summarizer with basic configuration."""
        self.stop_words = set([
            'a', 'an', 'the', 'and', 'but', 'or', 'because', 'as', 'if', 'when', 'than', 'but', 'for', 'with', 
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 
            've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 
            'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
        ])
        
        # Scientific keywords that might indicate important sentences
        self.science_keywords = [
            # General scientific terms
            'novel', 'approach', 'result', 'method', 'analysis', 'experiment', 'data', 'finding', 
            'significant', 'research', 'study', 'contribution', 'propose', 'demonstrate', 'evaluate', 
            'performance', 'conclude', 'framework', 'implementation', 'algorithm', 'technique', 
            'validation', 'accuracy', 'precision', 'recall', 'f1', 'metric', 'benchmark', 'baseline',
            'hypothesis', 'evidence', 'investigate', 'observation', 'methodology', 'empirical',
            'theoretical', 'outperform', 'state-of-the-art', 'sota', 'improvement',
            
            # Quantum computing specific
            'qubit', 'superposition', 'entanglement', 'quantum', 'coherence', 'decoherence',
            'quantum gate', 'quantum circuit', 'quantum algorithm', 'quantum supremacy',
            'quantum advantage', 'quantum error', 'quantum error correction', 'computation',
            
            # Computer science specific terms
            'architecture', 'compiler', 'complexity', 'encryption', 'hardware', 'interface',
            'latency', 'memory', 'network', 'optimization', 'protocol', 'software',
            'throughput', 'virtualization', 'paradigm', 'parameter', 'processing',
            
            # Scientific paper structure terms
            'abstract', 'conclusion', 'discussion', 'evaluation', 'future work',
            'introduction', 'limitation', 'literature', 'motivation', 'objective',
            'procedure', 'recommendation'
        ]
        
        logger.info("SimplifiedSummarizer initialized successfully")
    
    def _preprocess_text(self, text):
        """Preprocess the text by cleaning and normalizing."""
        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'\[[\d,\s]+\]', '', text)  # Remove citation markers like [1] or [1,2,3]
        
        # Split into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _calculate_word_frequencies(self, sentences):
        """Calculate word frequencies for the given sentences."""
        word_frequencies = Counter()
        
        for sentence in sentences:
            for word in re.findall(r'\w+', sentence.lower()):
                if word not in self.stop_words:
                    word_frequencies[word] += 1
        
        # Convert Counter to dictionary for easier manipulation
        word_freq_dict = dict(word_frequencies)
        
        # Normalize word frequencies
        max_frequency = max(word_freq_dict.values()) if word_freq_dict else 1
        normalized_frequencies = {}
        
        for word, count in word_freq_dict.items():
            normalized_frequencies[word] = count / max_frequency
            
        return normalized_frequencies
    
    def _score_sentences(self, sentences, word_frequencies):
        """Score sentences based on word frequencies and other features."""
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            # Score based on position (introduction and conclusion are often important)
            position_score = 1.0
            if i < len(sentences) * 0.2 or i > len(sentences) * 0.8:
                position_score = 1.5
            
            # Score based on word frequencies
            word_score = sum(word_frequencies.get(word.lower(), 0) 
                            for word in re.findall(r'\w+', sentence) 
                            if word.lower() not in self.stop_words)
            
            # Score based on presence of scientific keywords
            science_score = 1.0
            for keyword in self.science_keywords:
                if keyword in sentence.lower():
                    science_score = 1.3
                    break
            
            # Score based on sentence length (penalize very short or very long sentences)
            words_count = len(re.findall(r'\w+', sentence))
            length_score = 1.0
            if words_count < 5:
                length_score = 0.7
            elif words_count > 40:
                length_score = 0.8
            
            # Calculate final score
            final_score = (word_score * position_score * science_score * length_score) / max(1, words_count / 10)
            sentence_scores[i] = final_score
        
        return sentence_scores
    
    def _has_scientific_structure(self, text):
        """Check if the text appears to have a scientific paper structure."""
        sections = ['abstract', 'introduction', 'method', 'experiment', 'result', 'discussion', 'conclusion', 'reference']
        section_count = 0
        
        for section in sections:
            pattern = re.compile(r'\b' + section + r's?\b', re.IGNORECASE)
            if pattern.search(text):
                section_count += 1
        
        return section_count >= 3
    
    def _extract_key_sections(self, text):
        """Extract key sections from the text if it has a scientific structure."""
        sections = {}
        
        # Identify section headers using common patterns and numbered sections
        # First, split the text into lines to identify potential section headers
        lines = text.split('\n')
        potential_headers = []
        
        for i, line in enumerate(lines):
            # Look for lines that resemble headers (short, possibly ending with a colon)
            stripped = line.strip()
            if 3 <= len(stripped) <= 50 and (stripped.endswith(':') or stripped.isupper() or re.match(r'^[0-9]+\.', stripped)):
                potential_headers.append((i, stripped.rstrip(':')))
        
        # Try to find common section headers using regex patterns
        section_patterns = [
            (r'abstract[\s]*:?[\s]*(.*?)(?=\b(introduction|1\.|I\.)\b|\Z)', 'abstract'),
            (r'introduction[\s]*:?[\s]*(.*?)(?=\b(background|related work|method|2\.|II\.)\b|\Z)', 'introduction'),
            (r'(method|methodology|approach|implementation)[\s]*:?[\s]*(.*?)(?=\b(experiment|result|evaluation|4\.|III\.)\b|\Z)', 'methods'),
            (r'(result|experiment|evaluation|finding)[\s]*:?[\s]*(.*?)(?=\b(discussion|conclusion|5\.|IV\.)\b|\Z)', 'results'),
            (r'(discussion|analysis)[\s]*:?[\s]*(.*?)(?=\b(conclusion|future|limitation|6\.|V\.)\b|\Z)', 'discussion'),
            (r'(conclusion|summary)[\s]*:?[\s]*(.*?)(?=\b(reference|acknowledgment|future work|VII\.)\b|\Z)', 'conclusion')
        ]
        
        # Try regex section matching
        for pattern, section_name in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_content = match.group(1).strip()
                if section_content and len(section_content.split()) > 10:  # Ensure we have meaningful content
                    sections[section_name] = section_content
        
        # If we don't have enough sections, try additional methods
        if len(sections) < 3:
            # Look for numbered sections or capitalized headers
            header_patterns = {
                'abstract': [r'abstract', r'summary'],
                'introduction': [r'introduction', r'background', r'1[\.\s]', r'I[\.\s]'],
                'methods': [r'method', r'methodology', r'approach', r'2[\.\s]', r'II[\.\s]', r'implementation'],
                'results': [r'result', r'finding', r'experiment', r'evaluation', r'3[\.\s]', r'III[\.\s]'],
                'discussion': [r'discussion', r'analysis', r'interpretation', r'4[\.\s]', r'IV[\.\s]'],
                'conclusion': [r'conclusion', r'summary', r'future work', r'5[\.\s]', r'V[\.\s]']
            }
            
            # Extract sections based on potential headers
            current_section = None
            section_text = ""
            
            for i, line in enumerate(lines):
                # Check if this line is a potential section header
                is_header = False
                header_type = None
                
                for section_type, patterns in header_patterns.items():
                    if any(re.search(f"\\b{pattern}\\b", line, re.IGNORECASE) for pattern in patterns):
                        if current_section:
                            if current_section not in sections and section_text.strip():
                                sections[current_section] = section_text.strip()
                        
                        current_section = section_type
                        section_text = ""
                        is_header = True
                        break
                
                if not is_header and current_section:
                    section_text += line + "\n"
            
            # Add the last section
            if current_section and current_section not in sections and section_text.strip():
                sections[current_section] = section_text.strip()
        
        # If we still don't have an abstract but we have an introduction, try to extract the first paragraph
        if 'abstract' not in sections and 'introduction' in sections:
            intro_text = sections['introduction']
            # Extract first paragraph as pseudo-abstract
            first_para = intro_text.split('\n\n')[0] if '\n\n' in intro_text else intro_text.split('.')[0] + '.'
            if len(first_para.split()) >= 3:
                sections['abstract'] = first_para
        
        return sections
    
    def extract(self, text, max_length=250):
        """
        Generate an extractive summary by selecting the most important sentences.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary in words
            
        Returns:
            Extractive summary text
        """
        try:
            # Check if text has a scientific structure
            if self._has_scientific_structure(text):
                # Extract key sections
                sections = self._extract_key_sections(text)
                
                # If we found sections, focus on those for summarization
                if sections:
                    # We'll build a structured extractive summary using introduction, methods/results, conclusion
                    section_order = ['introduction', 'methods', 'results', 'conclusion', 'abstract', 'discussion']
                    section_content = {}
                    
                    # Calculate word budget per section
                    available_sections = [s for s in section_order if s in sections]
                    if not available_sections:
                        # No structured sections found
                        input_text = text
                    else:
                        # Allocate words based on section importance
                        section_weights = {
                            'introduction': 0.3,
                            'methods': 0.15,
                            'results': 0.25,
                            'conclusion': 0.2,
                            'abstract': 0.1,  # Lower weight because often already summarized
                            'discussion': 0.15
                        }
                        
                        # Normalize weights for found sections
                        total_weight = sum(section_weights[s] for s in available_sections)
                        normalized_weights = {s: section_weights[s]/total_weight for s in available_sections}
                        
                        # Process each section and select top sentences
                        for section in available_sections:
                            section_text = sections[section]
                            section_sentences = self._preprocess_text(section_text)
                            
                            if not section_sentences:
                                continue
                                
                            # Calculate sentence scores for this section
                            word_frequencies = self._calculate_word_frequencies(section_sentences)
                            sentence_scores = self._score_sentences(section_sentences, word_frequencies)
                            
                            # Determine number of sentences to include from this section
                            section_budget = int(max_length * normalized_weights[section])
                            avg_sent_length = sum(len(s.split()) for s in section_sentences) / len(section_sentences)
                            num_sentences = max(1, min(len(section_sentences), int(section_budget / avg_sent_length)))
                            
                            # Select top sentences from this section
                            top_indices = sorted(sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True)[:num_sentences]
                            top_indices = sorted(top_indices)  # Sort by position in text
                            section_content[section] = ' '.join(section_sentences[i] for i in top_indices)
                        
                        # Build structured summary in logical order
                        structured_parts = []
                        for section in section_order:
                            if section in section_content and section_content[section]:
                                structured_parts.append(section_content[section])
                        
                        input_text = ' '.join(structured_parts)
                else:
                    input_text = text
            else:
                input_text = text
            
            # For non-structured text, or as fallback, use standard extractive approach
            if input_text == text:
                # Preprocess the text
                sentences = self._preprocess_text(input_text)
                
                # Calculate word frequencies
                word_frequencies = self._calculate_word_frequencies(sentences)
                
                # Score sentences with additional positional importance
                sentence_scores = {}
                for i, sentence in enumerate(sentences):
                    # Score based on position - prefer intro and conclusion content
                    position_score = 1.0
                    if i < len(sentences) * 0.2:  # Introduction (first 20%)
                        position_score = 1.5
                    elif i > len(sentences) * 0.8:  # Conclusion (last 20%)
                        position_score = 1.3
                    
                    # Basic scoring from _score_sentences
                    base_score = self._score_sentences([sentence], word_frequencies).get(0, 1.0)
                    
                    # Prioritize sentences with key terms
                    contains_key_term = False
                    for term in self.science_keywords:
                        if term in sentence.lower():
                            contains_key_term = True
                            break
                    
                    term_multiplier = 1.2 if contains_key_term else 1.0
                    
                    # Final score
                    sentence_scores[i] = base_score * position_score * term_multiplier
                
                # Determine number of sentences to include
                average_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
                num_sentences = min(max(3, int(max_length / average_sentence_length)), len(sentences))
                
                # Select top sentences
                top_indices = sorted(sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True)[:num_sentences]
                selected_indices = sorted(top_indices)  # Sort by position in text
                
                # Build summary
                summary = ' '.join(sentences[i] for i in selected_indices)
            else:
                summary = input_text
            
            # Check for repetition
            sentences = self._preprocess_text(summary)
            unique_sentences = []
            
            for sentence in sentences:
                normalized = ' '.join(word.lower() for word in re.findall(r'\w+', sentence))
                is_duplicate = False
                
                for existing in unique_sentences:
                    existing_norm = ' '.join(word.lower() for word in re.findall(r'\w+', existing))
                    
                    # Check overlap percentage
                    if len(normalized) > 20 and len(existing_norm) > 20:
                        overlap = 0
                        for word in normalized.split():
                            if word in existing_norm:
                                overlap += 1
                        
                        # If more than 70% words overlap, consider it duplicate
                        if overlap / len(normalized.split()) > 0.7:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_sentences.append(sentence)
            
            final_summary = ' '.join(unique_sentences)
            
            # Truncate to max_length while preserving complete sentences
            words = final_summary.split()
            if len(words) > max_length:
                # Find a good cutoff point (end of sentence) near max_length
                partial_summary = ' '.join(words[:max_length + 15])  # Give some buffer
                cutoff_sentences = self._preprocess_text(partial_summary)
                
                # Keep as many complete sentences as possible
                kept_sentences = []
                word_count = 0
                
                for sent in cutoff_sentences:
                    sent_words = len(sent.split())
                    if word_count + sent_words <= max_length:
                        kept_sentences.append(sent)
                        word_count += sent_words
                    else:
                        # If we're close enough to max_length, stop
                        break
                
                final_summary = ' '.join(kept_sentences)
            
            return final_summary
        
        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            return "Error generating extractive summary. Please try again with different text."
    
    def abstract(self, text, max_length=250):
        """
        Generate a simplified abstractive summary.
        
        This is a simplified implementation that combines extractive summarization with basic
        sentence compression and reformulation techniques to create a more cohesive, fluent summary.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary in words
            
        Returns:
            Abstractive summary text
        """
        try:
            # First get an extractive summary
            extractive_summary = self.extract(text, max_length=max_length*2)  # Get a longer extractive summary
            sentences = self._preprocess_text(extractive_summary)
            
            # Extract key information from sentences
            key_terms = self._extract_key_terms(extractive_summary)
            
            # Simple sentence compression by removing less important parts
            compressed_sentences = []
            for sentence in sentences:
                # Remove parenthetical expressions
                sentence = re.sub(r'\([^)]*\)', '', sentence)
                
                # Remove clauses that start with certain words
                clause_starters = ['however', 'moreover', 'furthermore', 'in addition', 'for example', 'for instance', 'specifically']
                for starter in clause_starters:
                    sentence = re.sub(r',\s*' + starter + r'[^,.;:]*', '', sentence, flags=re.IGNORECASE)
                
                # Remove some adjectives and adverbs
                sentence = re.sub(r'\b(very|extremely|quite|rather|somewhat|slightly)\b', '', sentence, flags=re.IGNORECASE)
                
                # Fix any double spaces
                sentence = re.sub(r'\s+', ' ', sentence).strip()
                
                if sentence:
                    compressed_sentences.append(sentence)
            
            # Perform basic sentence fusion and reformulation
            fused_sentences = []
            i = 0
            while i < len(compressed_sentences):
                if i + 1 < len(compressed_sentences):
                    sent1 = compressed_sentences[i]
                    sent2 = compressed_sentences[i+1]
                    
                    # Check if sentences are related (share key terms)
                    common_terms = self._get_common_terms(sent1, sent2)
                    
                    if common_terms and len(common_terms) >= 2:
                        # Try to combine sentences
                        fused = self._fuse_sentences(sent1, sent2, common_terms)
                        fused_sentences.append(fused)
                        i += 2  # Skip next sentence as we've used it
                    else:
                        # Process sentence individually
                        reformulated = self._reformulate_sentence(sent1)
                        fused_sentences.append(reformulated)
                        i += 1
                else:
                    # Last sentence, process individually
                    reformulated = self._reformulate_sentence(compressed_sentences[i])
                    fused_sentences.append(reformulated)
                    i += 1
            
            # Generate connective phrases to improve flow between sentences
            connected_sentences = self._add_connectives(fused_sentences)
            
            # Create a more informative summary introduction and conclusion
            if key_terms:
                # Identify important scientific terms for the introduction
                scientific_terms = [term for term in key_terms if term in self.science_keywords]
                if scientific_terms:
                    primary_terms = scientific_terms[:3]
                else:
                    primary_terms = key_terms[:3]
                
                # Create better phrase combinations from primary terms
                if len(primary_terms) >= 2:
                    # Try to form natural phrases from the terms based on common patterns
                    phrases = []
                    
                    # Common domain phrases
                    domain_patterns = [
                        'quantum computing', 'quantum mechanics', 'quantum algorithms', 'quantum physics',
                        'computational model', 'computational paradigm', 'computational approach',
                        'research methodology', 'research framework', 'research paradigm',
                        'computing paradigm', 'computing model', 'computing technology',
                        'algorithm design', 'algorithm implementation', 'algorithm analysis'
                    ]
                    
                    # Check if any of our terms form known domain phrases
                    found_phrases = []
                    for pattern in domain_patterns:
                        # Split the pattern into component terms
                        pattern_terms = pattern.split()
                        
                        # Count how many terms from this pattern are in our primary terms
                        match_count = sum(1 for term in pattern_terms if term in primary_terms)
                        
                        # If at least half the terms in the pattern match our primary terms, include it
                        if match_count >= len(pattern_terms) / 2:
                            found_phrases.append(pattern)
                    
                    # If we found any domain phrases, use them
                    if found_phrases:
                        phrases.extend(found_phrases[:2])  # Use up to 2 domain phrases
                        
                        # Add any primary terms not covered in the phrases
                        for term in primary_terms:
                            if not any(term in phrase for phrase in phrases):
                                phrases.append(term)
                    else:
                        # No domain phrases found, try to form natural combinations
                        if 'quantum' in primary_terms:
                            quantum_pair = None
                            for term in primary_terms:
                                if term != 'quantum' and term in ['computing', 'computer', 'computers', 'algorithm', 'algorithms', 'mechanics', 'physics']:
                                    quantum_pair = f"quantum {term}"
                                    phrases.append(quantum_pair)
                                    break
                            
                            # If we used quantum in a pair, remove the individual terms
                            if quantum_pair:
                                remaining_terms = [t for t in primary_terms if t != 'quantum' and t not in quantum_pair]
                                phrases.extend(remaining_terms)
                            else:
                                phrases = primary_terms
                        else:
                            phrases = primary_terms
                else:
                    phrases = primary_terms
                    
                # Create a more varied and natural introduction using the improved phrases
                # Format the phrases with proper conjunctions
                if len(phrases) == 1:
                    formatted_phrases = phrases[0]
                elif len(phrases) == 2:
                    formatted_phrases = f"{phrases[0]} and {phrases[1]}"
                else:
                    # For 3+ phrases, use commas with final "and"
                    formatted_phrases = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
                
                intro_templates = [
                    f"This research focuses on {formatted_phrases}. ",
                    f"This study investigates {formatted_phrases}. ",
                    f"This paper examines advances in {formatted_phrases}. "
                ]
                intro = random.choice(intro_templates)
                
                # Create a substantive conclusion that includes key findings/contributions
                # Select domain-specific terms when available
                if scientific_terms and len(scientific_terms) >= 2:
                    conclusion_term1 = scientific_terms[0]
                    conclusion_term2 = scientific_terms[1]
                else:
                    conclusion_term1 = random.choice(key_terms)
                    remaining_terms = [t for t in key_terms if t != conclusion_term1]
                    conclusion_term2 = random.choice(remaining_terms) if remaining_terms else conclusion_term1
                
                conclusion_templates = [
                    f" In summary, the study contributes to the advancement of {conclusion_term1} research with significant implications for {conclusion_term2}.",
                    f" The findings demonstrate important progress in understanding {conclusion_term1} and its relationship to {conclusion_term2}.",
                    f" Overall, this work enhances our knowledge of {conclusion_term1} and provides new insights into {conclusion_term2}."
                ]
                conclusion = random.choice(conclusion_templates)
                
                # Determine lengths to fit within max_length
                main_content = ' '.join(connected_sentences)
                total_words = len(intro.split()) + len(main_content.split()) + len(conclusion.split())
                
                if total_words <= max_length:
                    abstractive_summary = intro + main_content + conclusion
                else:
                    # Reduce length of main content to fit intro and conclusion
                    words_to_keep = max_length - len(intro.split()) - len(conclusion.split())
                    # Ensure we keep at least some content
                    words_to_keep = max(words_to_keep, min(50, len(main_content.split())))
                    
                    # If we still need to truncate, prioritize keeping early content
                    main_words = main_content.split()[:words_to_keep]
                    abstractive_summary = intro + ' '.join(main_words) + conclusion
            else:
                abstractive_summary = ' '.join(connected_sentences)
            
            # Truncate to max_length if still too long
            words = abstractive_summary.split()
            if len(words) > max_length:
                abstractive_summary = ' '.join(words[:max_length]) + '...'
            
            return abstractive_summary
        
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {e}")
            return "Error generating abstractive summary. Please try again with different text."
            
    def _extract_key_terms(self, text):
        """Extract key terms from text based on frequency and positioning."""
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Get top terms
        key_terms = [word for word, count in word_freq.most_common(10)]
        
        # Add science keywords that appear in the text
        for keyword in self.science_keywords:
            if keyword in text.lower() and keyword not in key_terms:
                key_terms.append(keyword)
        
        return key_terms[:15]  # Limit to 15 terms
        
    def _get_common_terms(self, sent1, sent2):
        """Find common significant terms between two sentences."""
        words1 = set(re.findall(r'\b\w+\b', sent1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sent2.lower()))
        
        # Remove stop words
        words1 = {w for w in words1 if w not in self.stop_words and len(w) > 3}
        words2 = {w for w in words2 if w not in self.stop_words and len(w) > 3}
        
        # Find common words
        common = words1.intersection(words2)
        return list(common)
    
    def _fuse_sentences(self, sent1, sent2, common_terms):
        """Attempt to fuse two related sentences into one."""
        # Simple case: If one sentence starts with a common term that ends the other
        for term in common_terms:
            if sent1.lower().endswith(term) and sent2.lower().startswith(term):
                return sent1 + sent2[len(term):].strip()
        
        # Look for subject-verb patterns
        s1_parts = sent1.split('. ')
        s2_parts = sent2.split(', ')
        
        # Combine using a conjunction if the sentences are short
        if len(sent1.split()) + len(sent2.split()) < 30:
            connectors = ['and', 'while', 'whereas', 'furthermore']
            connector = random.choice(connectors)
            return f"{sent1.rstrip('.')} {connector} {sent2[0].lower() + sent2[1:]}"
        
        # If sentences are too complex, pick the one with the most key science terms
        science_count1 = sum(1 for term in self.science_keywords if term in sent1.lower())
        science_count2 = sum(1 for term in self.science_keywords if term in sent2.lower())
        
        return sent1 if science_count1 >= science_count2 else sent2
    
    def _reformulate_sentence(self, sentence):
        """Apply basic reformulation to sentences."""
        # Replace phrases with simpler alternatives
        replacements = {
            r'\bconducted an analysis\b': 'analyzed',
            r'\bperformed an evaluation\b': 'evaluated',
            r'\butilized\b': 'used',
            r'\bdemonstrated\b': 'showed',
            r'\bimplemented\b': 'created',
            r'\bin order to\b': 'to',
            r'\bwith respect to\b': 'regarding',
            r'\bin the context of\b': 'in',
            r'\bdespite the fact that\b': 'although',
            r'\bdue to the fact that\b': 'because',
            r'\bin spite of\b': 'despite',
            r'\ba significant number of\b': 'many',
            r'\ba majority of\b': 'most',
            r'\ba substantial amount of\b': 'much'
        }
        
        for pattern, replacement in replacements.items():
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        # Convert passive to active voice in some cases (simplified)
        passive_patterns = [
            (r'was conducted by', 'conducted'),
            (r'were developed by', 'developed'),
            (r'was performed by', 'performed'),
            (r'was implemented by', 'implemented'),
            (r'was utilized by', 'utilized'),
            (r'was analyzed by', 'analyzed')
        ]
        
        for pattern, replacement in passive_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                parts = re.split(pattern, sentence, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    subject = parts[1].strip()
                    # Check if subject ends with punctuation
                    if subject and subject[-1] in '.,:;':
                        subject = subject[:-1].strip()
                    
                    if subject:
                        sentence = f"{subject} {replacement} {parts[0].strip()}"
        
        return sentence
    
    def _add_connectives(self, sentences):
        """Add connective phrases to improve flow between sentences."""
        if not sentences:
            return []
        
        result = [sentences[0]]
        connectives = [
            'Additionally', 'Furthermore', 'Moreover', 
            'In this context', 'Notably', 'Specifically',
            'As a result', 'Consequently', 'Therefore'
        ]
        
        for i in range(1, len(sentences)):
            # Add connective phrase for approximately 1/3 of sentences
            if random.random() < 0.3 and len(sentences) > 3:
                connective = random.choice(connectives)
                result.append(f"{connective}, {sentences[i][0].lower() + sentences[i][1:]}")
            else:
                result.append(sentences[i])
                
        return result
    
    def summarize(self, text, max_length=250):
        """
        Generate a hybrid summary that combines extractive and abstractive approaches.
        
        This enhanced implementation creates a structured, cohesive summary by:
        1. Extracting key section content from scientific papers
        2. Identifying the most important information for each section
        3. Applying abstractive techniques to improve readability
        4. Adding transitional phrases to connect different sections
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary in words
            
        Returns:
            Hybrid summary text
        """
        try:
            # Handle minimum summary length to ensure we get enough content
            target_length = max(max_length, 200)  # Ensure at least 200 words for scientific papers
            
            # Extract key terms for the entire document
            key_terms = self._extract_key_terms(text)
            
            # Check if text has a scientific structure
            is_scientific = self._has_scientific_structure(text)
            
            # For scientific papers, use a structured approach
            if is_scientific:
                # Try to extract sections from the paper
                sections = self._extract_key_sections(text)
                
                # If not enough sections were found, try harder by splitting text into chunks
                if len(sections) < 3:
                    # Split text into chunks and try to identify section-like structures
                    paragraphs = text.split('\n\n')
                    current_section = 'introduction'  # Default to introduction if no clear headers
                    
                    for para in paragraphs:
                        para = para.strip()
                        if len(para) < 20:  # Short line, potential header
                            continue
                            
                        # Simple heuristic to identify paragraph topics
                        lower_para = para.lower()
                        if any(term in lower_para for term in ['methodology', 'approach', 'implementation']):
                            current_section = 'methods'
                            if 'methods' not in sections:
                                sections['methods'] = para
                            else:
                                sections['methods'] += ' ' + para
                        elif any(term in lower_para for term in ['result', 'finding', 'observation']):
                            current_section = 'results'
                            if 'results' not in sections:
                                sections['results'] = para
                            else:
                                sections['results'] += ' ' + para
                        elif any(term in lower_para for term in ['conclude', 'conclusion', 'summary']):
                            current_section = 'conclusion'
                            if 'conclusion' not in sections:
                                sections['conclusion'] = para
                            else:
                                sections['conclusion'] += ' ' + para
                        else:
                            # Add to current section
                            if current_section not in sections:
                                sections[current_section] = para
                            else:
                                sections[current_section] += ' ' + para
                
                # If we still don't have enough sections, fallback to a simpler approach
                if len(sections) < 2:
                    logger.info("Not enough sections found, falling back to simpler approach")
                    # Extract introduction (first quarter), middle (half), conclusion (last quarter)
                    words = text.split()
                    total_words = len(words)
                    intro_end = total_words // 4
                    concl_start = (total_words * 3) // 4
                    
                    sections = {
                        'introduction': ' '.join(words[:intro_end]),
                        'methods': ' '.join(words[intro_end:concl_start]),
                        'conclusion': ' '.join(words[concl_start:])
                    }
                
                # Generate summaries for important sections
                section_summaries = {}
                section_key_points = {}
                
                # Allocate word budget proportionally - increase to ensure more content
                total_sections = sum(1 for s in ['abstract', 'introduction', 'methods', 'results', 'conclusion', 'discussion'] if s in sections)
                if total_sections == 0:
                    logger.warning("No sections found in scientific paper")
                    return self.abstract(self.extract(text, max_length=int(target_length*1.5)), target_length)
                
                # Adjust budget based on number of sections present
                section_budget = {
                    'abstract': 0.15,
                    'introduction': 0.25,
                    'methods': 0.2,
                    'results': 0.25,
                    'conclusion': 0.2,
                    'discussion': 0.15
                }
                
                # Normalize budget based on available sections
                available_budget = sum(budget for section, budget in section_budget.items() if section in sections)
                if available_budget > 0:
                    multiplier = 1.0 / available_budget
                    for section in section_budget:
                        if section in sections:
                            section_budget[section] *= multiplier
                
                # 1. First pass: Generate extractive summaries for each section
                for section, content in sections.items():
                    if section in section_budget and content:
                        # Calculate word budget for this section
                        word_budget = int(target_length * section_budget[section])
                        word_budget = max(word_budget, 40)  # Ensure minimum content per section
                        
                        # For abstract, use it directly if it's concise enough
                        if section == 'abstract' and len(content.split()) <= int(word_budget*1.5):
                            section_summaries[section] = content
                        else:
                            # Generate extractive summary for other sections
                            # Use a larger word budget to get more content, then trim later
                            section_summaries[section] = self.extract(content, max_length=int(word_budget*1.2))
                            
                        # Extract key points from each section
                        section_key_points[section] = self._extract_key_terms(content)
                
                # 2. Second pass: Apply abstractive techniques to each section summary
                # This creates more coherent section summaries
                abstracted_sections = {}
                for section, summary in section_summaries.items():
                    # For shorter sections like abstract, keep as is
                    if section == 'abstract' or len(summary.split()) < 20:
                        abstracted_sections[section] = summary
                    else:
                        # Apply light abstractive techniques
                        compressed = self._compress_text(summary)
                        sentences = self._preprocess_text(compressed)
                        
                        # Apply reformulation to sentences
                        reformulated = [self._reformulate_sentence(s) for s in sentences]
                        abstracted_sections[section] = ' '.join(reformulated)
                
                # 3. Build a coherent summary with section headings and transitions
                final_parts = []
                
                # Create a title-like introduction if we have key terms
                if key_terms:
                    # Prioritize more technical/scientific terms
                    scientific_terms = [term for term in key_terms 
                                      if term in self.science_keywords or len(term) > 5]
                    
                    if scientific_terms:
                        key_terms_to_use = scientific_terms[:3]
                    else:
                        key_terms_to_use = key_terms[:3]
                        
                    if key_terms_to_use:
                        title_intro = f"This research focuses on {', '.join(key_terms_to_use)}. "
                        final_parts.append(title_intro)
                
                # Add each section with appropriate transitions
                sections_to_include = ['introduction', 'methods', 'results', 'conclusion', 'discussion']
                section_transitions = {
                    'introduction': '',
                    'methods': 'The methodology involves ',
                    'results': 'Key findings show that ',
                    'conclusion': 'In conclusion, ',
                    'discussion': 'The research suggests that '
                }
                
                for section in sections_to_include:
                    if section in abstracted_sections and abstracted_sections[section]:
                        section_text = abstracted_sections[section]
                        
                        # Add transition phrase if appropriate
                        transition = section_transitions.get(section, '')
                        
                        # Skip transition for introduction or if section is too short
                        if transition and section != 'introduction' and len(section_text) > 10:
                            # Check if the section already starts with a similar phrase
                            if not any(section_text.lower().startswith(t.lower()) for t in section_transitions.values()):
                                # Extract full first sentence to handle capitalization properly
                                first_sentence_match = re.match(r'^(.*?[.!?])\s', section_text)
                                if first_sentence_match:
                                    first_sentence = first_sentence_match.group(1)
                                    rest_of_text = section_text[len(first_sentence)+1:]
                                    
                                    # Modify first sentence to incorporate transition
                                    first_word = first_sentence.split()[0]
                                    if first_word[0].isupper():
                                        modified_first = transition + first_sentence[0].lower() + first_sentence[1:]
                                    else:
                                        modified_first = transition + first_sentence
                                    
                                    section_text = modified_first + " " + rest_of_text
                                else:
                                    # If we can't identify a clear first sentence, just prepend transition
                                    if section_text[0].isupper():
                                        section_text = transition + section_text[0].lower() + section_text[1:]
                                    else:
                                        section_text = transition + section_text
                                
                        final_parts.append(section_text)
                
                # If we couldn't extract structured sections, use the abstract
                if len(final_parts) <= 1 and 'abstract' in abstracted_sections:
                    final_parts.append(abstracted_sections['abstract'])
                
                # Combine everything
                hybrid_summary = ' '.join(final_parts)
                
                # Check for and remove duplicated sentences
                sentences = self._preprocess_text(hybrid_summary)
                
                # Simple deduplication: compare sentences and remove nearly identical ones
                unique_sentences = []
                for i, sentence in enumerate(sentences):
                    # Normalize the sentence for comparison
                    norm_sentence = re.sub(r'\s+', ' ', sentence.lower().strip())
                    
                    # Check if this sentence is similar to any we've seen
                    is_duplicate = False
                    for prev_sent in unique_sentences:
                        norm_prev = re.sub(r'\s+', ' ', prev_sent.lower().strip())
                        
                        # If strings are very similar or one contains the other
                        if norm_sentence == norm_prev or \
                           (len(norm_sentence) > 20 and norm_sentence in norm_prev) or \
                           (len(norm_prev) > 20 and norm_prev in norm_sentence):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        unique_sentences.append(sentence)
                
                # Rebuild summary with deduplicated sentences
                hybrid_summary = ' '.join(unique_sentences)
                
                # Add a default conclusion if none exists and we have room
                if 'conclusion' not in abstracted_sections and key_terms:
                    if len(hybrid_summary.split()) + 25 <= target_length:
                        # Create a more specific conclusion using key terms
                        technical_terms = [t for t in key_terms if t in self.science_keywords or len(t) > 6]
                        if technical_terms:
                            selected_terms = random.sample(technical_terms, min(2, len(technical_terms)))
                        else:
                            selected_terms = random.sample(key_terms, min(2, len(key_terms)))
                            
                        # Create a more varied conclusion
                        conclusion_templates = [
                            f" In conclusion, this research advances our understanding of {selected_terms[0]} and has significant implications for {selected_terms[-1]}.",
                            f" To summarize, the work presented demonstrates important progress in {selected_terms[0]} research with potential applications in {selected_terms[-1]}.",
                            f" Overall, this study provides valuable insights into {selected_terms[0]} technology and its relationship to {selected_terms[-1]}."
                        ]
                        conclusion = random.choice(conclusion_templates)
                        hybrid_summary += conclusion
                
                # Final pass: ensure it fits within max_length but doesn't cut off mid-sentence
                words = hybrid_summary.split()
                if len(words) > target_length:
                    # Find a good cutoff point (end of sentence) near target_length
                    text_to_truncate = ' '.join(words[:target_length + 30])  # Give some buffer
                    sentences_to_keep = self._preprocess_text(text_to_truncate)
                    
                    # Keep sentences until we reach or exceed target length
                    kept_sentences = []
                    word_count = 0
                    
                    for sent in sentences_to_keep:
                        sent_word_count = len(sent.split())
                        if word_count + sent_word_count <= target_length:
                            kept_sentences.append(sent)
                            word_count += sent_word_count
                        else:
                            # If we're close enough to target, include this sentence
                            if word_count > target_length * 0.85:
                                break
                            else:
                                kept_sentences.append(sent)
                                break
                    
                    hybrid_summary = ' '.join(kept_sentences)
                
                return hybrid_summary
            else:
                # For non-scientific text, use a two-step process
                # 1. Extract important content with a slightly larger budget
                extractive_summary = self.extract(text, max_length=int(target_length*1.5))
                
                # 2. Apply abstractive techniques
                return self.abstract(extractive_summary, target_length)
            
        except Exception as e:
            logger.error(f"Error in hybrid summarization: {e}")
            return "Error generating hybrid summary. Please try again with different text."
            
    def _compress_text(self, text):
        """Compress text by removing redundant information."""
        # Remove redundant phrases
        redundant_phrases = [
            r'it is important to note that',
            r'it should be noted that',
            r'it is worth mentioning that',
            r'as mentioned earlier',
            r'as previously stated',
            r'as discussed above',
            r'in this particular case',
            r'generally speaking',
        ]
        
        for phrase in redundant_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
            
        # Remove parenthetical content
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text