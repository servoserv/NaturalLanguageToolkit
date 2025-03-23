import re
import nltk
import spacy
import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from models import PaperStructure, Section, Author, Citation

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Process scientific paper text and extract structured information."""
    
    def __init__(self):
        """Initialize the data processor with required models."""
        # Load spaCy model for NER and dependency parsing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("Downloading spaCy model...")
            import subprocess
            subprocess.call(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Stopwords for filtering
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Regular expressions for section identification
        self.section_patterns = {
            'abstract': r'(?i)abstract',
            'introduction': r'(?i)introduction|(?i)background',
            'methods': r'(?i)methods|(?i)methodology|(?i)materials and methods|(?i)experimental',
            'results': r'(?i)results',
            'discussion': r'(?i)discussion',
            'conclusion': r'(?i)conclusion|(?i)conclusions|(?i)concluding remarks',
            'references': r'(?i)references|(?i)bibliography|(?i)literature cited'
        }
        
        # Scientific terms dictionary for domain recognition
        self.scientific_domain_terms = {
            'computer_science': ['algorithm', 'computation', 'neural network', 'machine learning', 
                               'artificial intelligence', 'data structure', 'programming'],
            'medicine': ['patient', 'clinical', 'disease', 'treatment', 'diagnosis', 'therapy', 
                       'medical', 'symptom', 'hospital', 'physician'],
            'biology': ['cell', 'organism', 'gene', 'protein', 'enzyme', 'tissue', 'dna', 
                      'molecular', 'biological', 'evolution']
        }
        
    def is_research_paper(self, text: str) -> bool:
        """Determine if text is a research paper based on structure and content."""
        # Check for common section headers
        section_matches = 0
        for section in self.section_patterns.values():
            if re.search(section, text):
                section_matches += 1
        
        # Check for citations
        citation_pattern = r'\[\d+\]|\(\w+\s+et\s+al\.\s*,\s*\d{4}\)|\(\w+\s*,\s*\d{4}\)'
        citations = re.findall(citation_pattern, text)
        
        # Check for academic language
        academic_terms = ['hypothesis', 'methodology', 'analysis', 'experiment', 'investigation',
                         'findings', 'literature', 'framework', 'significant', 'prior work']
        academic_term_count = 0
        for term in academic_terms:
            if re.search(rf'\b{term}\b', text, re.IGNORECASE):
                academic_term_count += 1
        
        # Make decision based on combined evidence
        if (section_matches >= 3 or len(citations) >= 5) and academic_term_count >= 3:
            return True
        return False
    
    def process_research_paper(self, text: str) -> PaperStructure:
        """
        Process text identified as a research paper and extract structured information.
        
        Args:
            text: The full text of the research paper
            
        Returns:
            A PaperStructure object containing the parsed paper
        """
        logger.info("Processing research paper...")
        
        # Split text into sections
        sections = self._extract_sections(text)
        
        # Extract title and abstract
        title = self._extract_title(text)
        abstract = sections.get('abstract', '')
        
        # Extract authors
        authors = self._extract_authors(text)
        
        # Extract keywords
        keywords = self._extract_keywords(text)
        
        # Process each major section
        introduction = Section(title="Introduction", content=sections.get('introduction', '')) if 'introduction' in sections else None
        methods = Section(title="Methods", content=sections.get('methods', '')) if 'methods' in sections else None
        results = Section(title="Results", content=sections.get('results', '')) if 'results' in sections else None
        discussion = Section(title="Discussion", content=sections.get('discussion', '')) if 'discussion' in sections else None
        conclusion = Section(title="Conclusion", content=sections.get('conclusion', '')) if 'conclusion' in sections else None
        
        # Extract citations
        references_text = sections.get('references', '')
        citations = self._extract_citations(references_text)
        
        # Create additional sections for any other identified sections
        additional_sections = []
        for section_name, section_content in sections.items():
            if section_name not in ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'references']:
                additional_sections.append(Section(title=section_name.capitalize(), content=section_content))
        
        # Create paper structure
        paper = PaperStructure(
            title=title,
            abstract=abstract,
            introduction=introduction,
            methods=methods,
            results=results,
            discussion=discussion,
            conclusion=conclusion,
            references=citations,
            sections=additional_sections,
            authors=authors,
            keywords=keywords
        )
        
        # Determine domain
        domain = self._identify_domain(text)
        paper.metadata['domain'] = domain
        
        return paper
    
    def process_general_text(self, text: str) -> str:
        """Process general text that isn't specifically a research paper."""
        logger.info("Processing general text...")
        
        # Basic processing - split into paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        # Merge paragraphs if they're very short
        merged_paragraphs = []
        current_paragraph = ""
        
        for p in paragraphs:
            if len(current_paragraph.split()) + len(p.split()) < 100:  # Threshold of 100 words
                current_paragraph += " " + p if current_paragraph else p
            else:
                if current_paragraph:
                    merged_paragraphs.append(current_paragraph)
                current_paragraph = p
        
        if current_paragraph:
            merged_paragraphs.append(current_paragraph)
        
        # Clean and join the paragraphs
        processed_text = "\n\n".join(merged_paragraphs)
        return processed_text
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from the research paper text."""
        sections = {}
        
        # First, identify section boundaries using regex
        section_boundaries = []
        
        # Find all potential section headers
        header_pattern = r'^([A-Z][A-Za-z\s]+)(?:\n|\r\n?)'
        potential_headers = re.finditer(header_pattern, text, re.MULTILINE)
        
        for match in potential_headers:
            section_name = match.group(1).strip().lower()
            start_pos = match.start()
            
            # Check if this matches one of our known section types
            section_type = None
            for known_section, pattern in self.section_patterns.items():
                if re.search(pattern, section_name):
                    section_type = known_section
                    break
            
            if section_type:
                section_boundaries.append((start_pos, section_type))
            else:
                # For unknown sections, use their actual name
                section_boundaries.append((start_pos, section_name))
        
        # Sort by position
        section_boundaries.sort()
        
        # Extract text for each section
        for i, (start_pos, section_type) in enumerate(section_boundaries):
            if i < len(section_boundaries) - 1:
                section_text = text[start_pos:section_boundaries[i+1][0]]
            else:
                section_text = text[start_pos:]
            
            # Remove the header itself
            section_text = re.sub(header_pattern, '', section_text, count=1, flags=re.MULTILINE)
            sections[section_type] = section_text.strip()
        
        return sections
    
    def _extract_title(self, text: str) -> str:
        """Extract the title from the research paper."""
        # Try to find the title in the first few lines
        lines = text.split('\n')
        for i in range(min(10, len(lines))):
            line = lines[i].strip()
            if line and not line.lower().startswith(('abstract', 'introduction', 'keywords')):
                # If line is all uppercase or title case, and not too long
                if (line.isupper() or line.istitle()) and len(line) < 200:
                    return line
        
        # Fallback to first line if no clear title found
        return lines[0].strip() if lines else "Unknown Title"
    
    def _extract_authors(self, text: str) -> List[Author]:
        """Extract author information from the paper."""
        authors = []
        
        # Look for author section near the beginning
        first_500_chars = text[:500]
        
        # Common patterns for author lists
        author_section_pattern = r'(?:Author|Authors)(?::|s:|\s+)?([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n\s*(?:Abstract|Introduction|Keywords))'
        author_match = re.search(author_section_pattern, first_500_chars, re.IGNORECASE)
        
        if author_match:
            author_text = author_match.group(1)
            
            # Split by commas, ands, or newlines
            author_parts = re.split(r',|\sand\s|\n', author_text)
            
            for part in author_parts:
                name = part.strip()
                if name:  # Skip empty strings
                    # Extract email if present
                    email_match = re.search(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', name)
                    email = email_match.group(1) if email_match else None
                    
                    # Remove email from name if present
                    if email:
                        name = name.replace(email, '').strip()
                    
                    # Extract affiliation if marked with superscript
                    affiliation_match = re.search(r'(\w+)(\d+)', name)
                    affiliation = affiliation_match.group(2) if affiliation_match else None
                    
                    # Clean name
                    name = re.sub(r'\d+', '', name).strip()
                    
                    authors.append(Author(name=name, affiliation=affiliation, email=email))
        
        return authors
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the paper."""
        keywords = []
        
        # Look for keywords section
        keyword_pattern = r'(?:Key\s*words|Keywords)(?::|s:|\s+)?([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n\s*(?:Abstract|Introduction))'
        keyword_match = re.search(keyword_pattern, text, re.IGNORECASE)
        
        if keyword_match:
            keyword_text = keyword_match.group(1)
            
            # Split by commas, semicolons
            keyword_parts = re.split(r'[,;]', keyword_text)
            
            for part in keyword_parts:
                keyword = part.strip().lower()
                if keyword and keyword not in self.stopwords:
                    keywords.append(keyword)
        
        return keywords
    
    def _extract_citations(self, references_text: str) -> List[Citation]:
        """Extract citations from the references section."""
        citations = []
        
        if not references_text:
            return citations
        
        # Split references by numbers or newlines
        reference_pattern = r'(?:\[\d+\]|\d+\.)\s+(.+?)(?=(?:\[\d+\]|\d+\.|\Z))'
        raw_citations = re.findall(reference_pattern, references_text)
        
        for i, citation_text in enumerate(raw_citations):
            # Extract basic information using regex
            author_match = re.search(r'^([^\.]+)', citation_text)
            year_match = re.search(r'\((\d{4})\)', citation_text)
            title_match = re.search(r'\((?:\d{4})\)\s*\.?\s*(.+?)\.', citation_text)
            
            authors = []
            if author_match:
                author_text = author_match.group(1)
                author_parts = re.split(r',|\sand\s', author_text)
                authors = [a.strip() for a in author_parts if a.strip()]
            
            year = int(year_match.group(1)) if year_match else None
            title = title_match.group(1).strip() if title_match else None
            
            citations.append(Citation(
                id=f"cite_{i+1}",
                text=citation_text.strip(),
                authors=authors,
                year=year,
                title=title,
                source=None  # Source would require more complex parsing
            ))
        
        return citations

    def _identify_domain(self, text: str) -> str:
        """Identify the domain of the research paper."""
        # Count occurrences of domain-specific terms
        domain_scores = {domain: 0 for domain in self.scientific_domain_terms}
        
        for domain, terms in self.scientific_domain_terms.items():
            for term in terms:
                count = len(re.findall(rf'\b{term}\b', text, re.IGNORECASE))
                domain_scores[domain] += count
        
        # Return the domain with the highest score
        if max(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
