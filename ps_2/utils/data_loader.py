import os
import json
import pandas as pd
import logging
import requests
from io import BytesIO
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

def load_compscholar_dataset(file_path=None):
    """
    Load the CompScholar dataset
    
    Args:
        file_path (str): Path to the CompScholar CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the dataset
    """
    try:
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded CompScholar dataset with {len(df)} records")
            return df
        else:
            # Return a small sample for demonstration
            logger.warning("CompScholar dataset file not found, returning sample data")
            return create_sample_compscholar_data()
    except Exception as e:
        logger.error(f"Error loading CompScholar dataset: {e}")
        return create_sample_compscholar_data()

def load_arxiv_dataset(file_path=None):
    """
    Load the arXiv dataset
    
    Args:
        file_path (str): Path to the arXiv JSON file
        
    Returns:
        list: List of dictionaries containing arXiv papers
    """
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded arXiv dataset with {len(data)} records")
            return data
        else:
            # Return a small sample for demonstration
            logger.warning("arXiv dataset file not found, returning sample data")
            return create_sample_arxiv_data()
    except Exception as e:
        logger.error(f"Error loading arXiv dataset: {e}")
        return create_sample_arxiv_data()

def load_pubmed_dataset(file_path=None):
    """
    Load the PubMed dataset
    
    Args:
        file_path (str): Path to the PubMed XML or JSON file
        
    Returns:
        list: List of dictionaries containing PubMed papers
    """
    try:
        if file_path and os.path.exists(file_path):
            if file_path.endswith('.xml'):
                # Parse XML
                tree = ET.parse(file_path)
                root = tree.getroot()
                data = []
                for citation in root.findall('.//MedlineCitation'):
                    paper = {}
                    paper['PMID'] = citation.find('PMID').text
                    article = citation.find('Article')
                    if article is not None:
                        paper['title'] = article.find('ArticleTitle').text if article.find('ArticleTitle') is not None else ""
                        abstract = article.find('Abstract/AbstractText')
                        paper['abstract'] = abstract.text if abstract is not None else ""
                    data.append(paper)
                logger.info(f"Successfully loaded PubMed XML dataset with {len(data)} records")
                return data
            elif file_path.endswith('.json'):
                # Parse JSON
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded PubMed JSON dataset with {len(data)} records")
                return data
        else:
            # Return a small sample for demonstration
            logger.warning("PubMed dataset file not found, returning sample data")
            return create_sample_pubmed_data()
    except Exception as e:
        logger.error(f"Error loading PubMed dataset: {e}")
        return create_sample_pubmed_data()

def create_sample_compscholar_data():
    """Create a small sample of CompScholar data for demonstration"""
    logger.info("Creating sample CompScholar data")
    data = {
        'Paper Id': ['CS001', 'CS002', 'CS003'],
        'Paper Title': [
            'Advanced Techniques for Neural Text Summarization', 
            'Document Summarization with BERT and Graph Neural Networks',
            'Multi-Document Summarization for Scientific Research'
        ],
        'Key Word': [
            'text summarization, neural networks, NLP', 
            'BERT, graph neural networks, document summarization',
            'multi-document summarization, scientific research, extractive-abstractive'
        ],
        'Abstract': [
            'This paper presents a novel approach to neural text summarization using transformer-based architectures. We propose a hybrid model that combines extractive and abstractive methods to produce concise and informative summaries.',
            'We introduce a new method for document summarization that leverages BERT embeddings and graph neural networks. Our approach captures both semantic and structural information in documents.',
            'This study addresses the challenge of summarizing multiple scientific research documents. We propose a pipeline that first extracts key information across documents and then generates coherent summaries.'
        ],
        'Conclusion': [
            'Our experiments demonstrate that the proposed hybrid approach outperforms state-of-the-art methods on standard benchmarks, achieving higher ROUGE and BLEU scores while maintaining summary coherence.',
            'The evaluation shows that our BERT-GNN approach significantly improves summarization quality over baseline methods, particularly for technical and scientific documents.',
            'The results indicate that our multi-document summarization approach effectively condenses information from multiple sources while preserving critical research findings and methodological details.'
        ],
        'Document': [
            'Advanced Techniques for Neural Text Summarization. Keywords: text summarization, neural networks, NLP. Abstract: This paper presents a novel approach to neural text summarization using transformer-based architectures. We propose a hybrid model that combines extractive and abstractive methods to produce concise and informative summaries. Conclusion: Our experiments demonstrate that the proposed hybrid approach outperforms state-of-the-art methods on standard benchmarks, achieving higher ROUGE and BLEU scores while maintaining summary coherence.',
            'Document Summarization with BERT and Graph Neural Networks. Keywords: BERT, graph neural networks, document summarization. Abstract: We introduce a new method for document summarization that leverages BERT embeddings and graph neural networks. Our approach captures both semantic and structural information in documents. Conclusion: The evaluation shows that our BERT-GNN approach significantly improves summarization quality over baseline methods, particularly for technical and scientific documents.',
            'Multi-Document Summarization for Scientific Research. Keywords: multi-document summarization, scientific research, extractive-abstractive. Abstract: This study addresses the challenge of summarizing multiple scientific research documents. We propose a pipeline that first extracts key information across documents and then generates coherent summaries. Conclusion: The results indicate that our multi-document summarization approach effectively condenses information from multiple sources while preserving critical research findings and methodological details.'
        ],
        'Paper Type': [
            'Text Summarization',
            'Natural Language Processing',
            'Text Summarization'
        ],
        'Summary': [
            'This paper introduces a hybrid neural text summarization approach combining extractive and abstractive methods. The model uses transformer architectures to generate concise summaries while preserving key information. Experimental results show superior performance compared to existing methods on standard benchmarks.',
            'The research presents a document summarization technique integrating BERT with graph neural networks. This approach captures both semantic content and document structure. Evaluation demonstrates significant improvements over baselines, especially for technical and scientific documents.',
            'This paper addresses multi-document summarization for scientific literature. The proposed pipeline extracts key information across multiple documents before generating coherent summaries. Results show effective condensation of information while maintaining critical research findings and methodological details.'
        ],
        'Topic': [
            'Natural Language Processing',
            'Natural Language Processing',
            'Natural Language Processing'
        ],
        'labels': [
            'Deep Learning and Machine',
            'Deep Learning and Machine',
            'Deep Learning and Machine'
        ]
    }
    return pd.DataFrame(data)

def create_sample_arxiv_data():
    """Create a small sample of arXiv data for demonstration"""
    logger.info("Creating sample arXiv data")
    return [
        {
            "id": "2104.01011",
            "submitter": "John Smith",
            "authors": "John Smith, Jane Doe, Robert Johnson",
            "title": "Transformer-based Models for Long Document Summarization",
            "comments": "Accepted at ACL 2021",
            "journal-ref": "None",
            "doi": "10.1234/example.5678",
            "categories": "cs.CL cs.AI",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "Transformer models have revolutionized NLP tasks but struggle with long documents due to quadratic attention complexity. We propose a novel approach that segments documents and applies hierarchical attention to enable efficient summarization of lengthy scientific articles. Our method outperforms existing approaches on the arXiv and PubMed benchmarks while using significantly less computational resources.",
            "update_date": "2021-04-15"
        },
        {
            "id": "2105.02022",
            "submitter": "Alice Wang",
            "authors": "Alice Wang, Bob Johnson, Carol Martinez",
            "title": "Domain-Specific Pre-training for Scientific Text Summarization",
            "comments": "Under review at EMNLP 2021",
            "journal-ref": "None",
            "doi": "None",
            "categories": "cs.CL cs.IR",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "While general-purpose language models have shown impressive results, domain-specific tasks like scientific text summarization require specialized knowledge. We introduce ScienceBERT, a model pre-trained on a large corpus of scientific articles. Our approach demonstrates significant improvements in summarizing research papers from multiple scientific disciplines, particularly in capturing domain-specific terminology and concepts that general models typically miss.",
            "update_date": "2021-05-20"
        },
        {
            "id": "2106.03033",
            "submitter": "David Chen",
            "authors": "David Chen, Emma Wilson, Frank Lopez",
            "title": "Citation-Aware Scientific Document Summarization",
            "comments": "Accepted at NAACL 2022",
            "journal-ref": "None",
            "doi": "10.5678/example.1234",
            "categories": "cs.CL cs.DL",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "Citations play a crucial role in scientific literature, indicating important concepts and relationships between papers. We present a summarization framework that explicitly models citation context to identify salient information in research articles. By incorporating citation networks into our model architecture, we generate summaries that better capture the key contributions and findings as recognized by the scientific community. Evaluation on computer science and biomedical papers shows substantial improvements in summary quality.",
            "update_date": "2021-06-25"
        }
    ]

def create_sample_pubmed_data():
    """Create a small sample of PubMed data for demonstration"""
    logger.info("Creating sample PubMed data")
    return [
        {
            "id": "PMC1234567",
            "article": "Background: Summarization of medical research papers is essential for clinical decision making but remains challenging due to domain-specific terminology and complex structure. Methods: We developed a hybrid extractive-abstractive summarization approach tailored for biomedical literature. Our system uses a biomedical entity recognition component to identify key medical concepts, followed by a transformer-based generator fine-tuned on medical abstracts. Results: Evaluation on 500 medical papers demonstrates our approach outperforms general-purpose summarization models, with significantly better retention of critical medical information and findings. Conclusion: Domain-specific summarization models are essential for medical literature, where precision and completeness of information are critical for clinical applications.",
            "abstract": "We present a specialized summarization system for biomedical research papers that combines entity recognition with transformer-based generation. Evaluated on 500 papers, our approach significantly outperforms general models in retaining critical medical information, demonstrating the importance of domain-specific summarization for clinical applications."
        },
        {
            "id": "PMC7654321",
            "article": "Introduction: Drug repurposing offers an efficient pathway for identifying new treatments by testing existing approved medications for novel therapeutic uses. This approach is particularly valuable for addressing emerging diseases. Methods: We developed a natural language processing system to analyze 50,000 biomedical articles and identify potential drug repurposing candidates for COVID-19 treatment. Our pipeline extracted drug-disease relationships, validated them against clinical trials, and ranked candidates by evidence strength. Results: Our system identified 17 high-potential drug candidates, 5 of which were subsequently confirmed effective in clinical trials. The NLP approach demonstrated 82% precision and 76% recall in identifying valid drug repurposing opportunities. Discussion: This study demonstrates the effectiveness of large-scale text mining for accelerating drug repurposing research, potentially reducing the time and cost of traditional drug discovery processes.",
            "abstract": "We developed an NLP system analyzing 50,000 biomedical articles to identify drug repurposing candidates for COVID-19. Our approach extracted and validated drug-disease relationships, identifying 17 high-potential candidates, with 5 later confirmed effective in clinical trials. The system achieved 82% precision and 76% recall, demonstrating text mining's value in accelerating drug repurposing research."
        },
        {
            "id": "PMC9876543",
            "article": "Background: Automatic summarization of patient electronic health records (EHRs) can significantly reduce physician cognitive load and improve clinical efficiency. However, EHR summarization presents unique challenges including privacy concerns, medical terminology, and integration of structured and unstructured data. Methods: We present MedSum, a BERT-based summarization system fine-tuned on 10,000 annotated EHR-summary pairs. The model incorporates medical ontologies and handles both structured data fields and free-text clinical notes. We implemented strict de-identification procedures to protect patient privacy throughout the process. Results: In a randomized trial with 45 physicians, MedSum reduced chart review time by 38% while maintaining 96% information retention compared to manual review. Physician satisfaction scores were significantly higher when using the system. Conclusion: MedSum demonstrates that specialized NLP models can effectively summarize complex medical records, potentially saving thousands of physician hours while maintaining high information fidelity.",
            "abstract": "We introduce MedSum, a BERT-based system for summarizing electronic health records, fine-tuned on 10,000 annotated EHR-summary pairs with medical ontology integration. In trials with 45 physicians, the system reduced review time by 38% while maintaining 96% information retention. Results demonstrate specialized NLP models' effectiveness in summarizing complex medical records, with potential for significant time savings and high physician satisfaction."
        }
    ]

def load_sample_data():
    """Load or create sample papers for demonstration in the UI"""
    papers = [
        {
            "id": "sample1",
            "title": "Transformer-based Models for Long Document Summarization",
            "text": """Transformer models have revolutionized NLP tasks but struggle with long documents due to quadratic attention complexity. We propose a novel approach that segments documents and applies hierarchical attention to enable efficient summarization of lengthy scientific articles. Our method outperforms existing approaches on the arXiv and PubMed benchmarks while using significantly less computational resources.

Introduction:
The task of automatic text summarization has seen remarkable progress with the advent of transformer-based architectures. Models such as BART, T5, and PEGASUS have achieved state-of-the-art results on various summarization benchmarks. However, these models face significant challenges when applied to long documents such as scientific papers, which often exceed several thousand tokens.

The key limitation stems from the self-attention mechanism in transformers, which scales quadratically with sequence length in terms of both computation and memory requirements. Most pre-trained models restrict input length to 512-1024 tokens, making them unsuitable for processing full scientific articles that typically range from 3,000 to 15,000 tokens.

Prior approaches to long document summarization have explored extractive methods that identify and extract key sentences, sliding window approaches that process documents in chunks, or sparse attention patterns that reduce computational complexity. While these methods have shown promise, they often fail to capture the global context and hierarchical structure inherent in scientific documents.

Methods:
We propose a hierarchical transformer architecture specifically designed for long scientific document summarization. Our approach, which we call HiTrans, consists of three main components:

1. Document Segmentation: We divide the input document into semantically coherent segments using a combination of structural cues (section boundaries) and a trained segmentation model that identifies topical shifts.

2. Segment Encoding: Each segment is independently encoded using a transformer encoder pre-trained on scientific text. This produces contextualized representations for text within each segment.

3. Hierarchical Attention: A second-level transformer processes the segment representations to capture cross-segment relationships and global document structure. This model attends to segment-level representations rather than individual tokens, dramatically reducing computational complexity.

The final decoder generates summaries using hierarchical attention over both segment-level and token-level representations, allowing it to integrate local and global information effectively.

We fine-tuned our model on the arXiv and PubMed datasets, which contain scientific papers paired with their abstracts as reference summaries. For training efficiency, we initialized the segment encoder with SciBERT weights and the hierarchical components with T5 weights.

Results:
We evaluated HiTrans against several strong baselines, including PEGASUS, BART-Long, LED, and BigBird. Experiments on the arXiv dataset show that HiTrans achieves a ROUGE-1 score of 47.2, ROUGE-2 score of 23.1, and ROUGE-L score of 43.5, outperforming all baseline models. On the PubMed dataset, our model achieves ROUGE-1/2/L scores of 45.8/21.7/42.6 respectively.

Importantly, HiTrans shows particular strength in capturing key contributions and methods from scientific papers, as verified through human evaluation. Experts rated summaries produced by HiTrans as significantly more informative and complete compared to those generated by baseline systems.

Furthermore, our approach reduces computational requirements by approximately 70% compared to models using standard full attention, making it practical for deployment in resource-constrained environments.

Discussion:
The superior performance of HiTrans can be attributed to several factors. First, the hierarchical architecture effectively captures the inherent structure of scientific documents, which are typically organized into sections and subsections with distinct purposes. Second, by processing the document at both local and global levels, our model captures fine-grained details while maintaining awareness of the overall document context.

We also observed that the segmentation strategy significantly impacts summarization quality. Segments based on structural boundaries (e.g., section headers) generally led to better summaries than arbitrary fixed-length chunks, highlighting the importance of respecting document structure.

Conclusion:
We presented HiTrans, a hierarchical transformer architecture for long scientific document summarization. By leveraging document structure and employing a two-level attention mechanism, our approach effectively processes documents of arbitrary length while maintaining computational efficiency. Experimental results demonstrate state-of-the-art performance on scientific paper summarization benchmarks, with particular strengths in capturing key contributions and methodological details.

Future work will explore extending this approach to multi-document summarization and incorporating citation information to better identify influential content in scientific literature."""
        },
        {
            "id": "sample2",
            "title": "Domain-Specific Pre-training for Scientific Text Summarization",
            "text": """While general-purpose language models have shown impressive results, domain-specific tasks like scientific text summarization require specialized knowledge. We introduce ScienceBERT, a model pre-trained on a large corpus of scientific articles. Our approach demonstrates significant improvements in summarizing research papers from multiple scientific disciplines, particularly in capturing domain-specific terminology and concepts that general models typically miss.

Introduction:
Recent advances in transformer-based language models have revolutionized natural language processing tasks, including text summarization. Models like BERT, RoBERTa, T5, and GPT have demonstrated remarkable capabilities in understanding and generating text across various domains. However, when applied to specialized scientific literature, these general-purpose models often struggle with domain-specific terminology, complex concepts, and the structured nature of scientific writing.

Scientific articles present unique challenges for summarization systems. They contain specialized vocabulary, complex relationships between concepts, domain-specific rhetorical structures, and often rely on background knowledge not explicitly stated in the text. General pre-trained language models, while powerful, lack the specialized knowledge required to effectively understand and summarize scientific content.

In this paper, we introduce ScienceBERT, a domain-adapted language model specifically pre-trained on a large corpus of scientific literature from multiple disciplines. We demonstrate that domain-specific pre-training significantly enhances summarization performance for scientific articles across different fields.

Methods:
Our approach consists of three main components:

1. Domain-Specific Pre-training: We collected a corpus of 2.5 million full-text scientific articles from open-access journals across multiple disciplines, including computer science, biomedicine, physics, and chemistry. Using this corpus, we continued pre-training from the RoBERTa checkpoint using the masked language modeling objective. Pre-training was conducted for 100,000 steps with a batch size of 256 sequences.

2. Scientific Structure Modeling: We introduced additional pre-training objectives to capture the structured nature of scientific text:
   - Section classification: Predicting which section a given passage belongs to (e.g., Introduction, Methods, Results)
   - Citation prediction: Predicting whether a sentence contains a citation
   - Term definition matching: Linking specialized terms with their definitions

3. Summarization Fine-tuning: We fine-tuned the resulting model on scientific summarization tasks using datasets derived from paper abstracts. During fine-tuning, we employed a hybrid extractive-abstractive approach where the model first identifies key sentences and then generates a coherent summary.

For evaluation, we used the arXiv, PubMed, and a new multi-discipline scientific summarization dataset we created called SciSum, which includes papers from computer science, physics, chemistry, and biology with expert-written summaries.

Results:
We compared ScienceBERT against several strong baselines, including BART, PEGASUS, T5, and LED, all of which have demonstrated strong performance on summarization tasks.

On the arXiv dataset, ScienceBERT achieved ROUGE-1/2/L scores of 46.8/22.4/42.9, representing improvements of 2.3/1.5/1.8 points over the best baseline (PEGASUS). On PubMed, our model achieved ROUGE-1/2/L scores of 45.3/21.2/41.8, outperforming the best baseline by 1.7/0.9/1.2 points.

The most substantial improvements were observed on our SciSum dataset, where ScienceBERT outperformed the strongest baseline by 3.1/2.4/2.7 ROUGE-1/2/L points, demonstrating its effectiveness across multiple scientific domains.

Human evaluation revealed that summaries generated by ScienceBERT were rated significantly higher in terms of scientific accuracy, completeness, and coherence compared to baseline models. Domain experts noted that ScienceBERT summaries more accurately captured specialized terminology and key scientific contributions.

Ablation studies showed that both domain-specific pre-training and scientific structure modeling contributed substantially to the final performance, with their combination yielding the best results.

Discussion:
Our results demonstrate the importance of domain-specific pre-training for scientific text summarization. The improvements observed across multiple datasets and scientific disciplines suggest that ScienceBERT effectively captures domain knowledge that general-purpose models miss.

Qualitative analysis revealed several key strengths of our approach:

1. Terminology Handling: ScienceBERT more accurately preserves domain-specific terminology and explains complex concepts.

2. Structural Awareness: The model demonstrates better awareness of scientific document structure, appropriately emphasizing methodology and results.

3. Background Knowledge: ScienceBERT shows evidence of leveraging implicit scientific knowledge not explicitly stated in the text.

One limitation of our current approach is computational efficiency - processing full scientific papers remains challenging. Future work will explore more efficient architectures for handling long scientific documents.

Conclusion:
We presented ScienceBERT, a domain-specific pre-trained language model for scientific text summarization. Our approach combines domain-adapted pre-training on a large scientific corpus with structural modeling objectives specifically designed for scientific text. Experimental results demonstrate significant improvements over strong baselines across multiple scientific disciplines.

This work highlights the importance of domain-specific adaptations for language models when applied to specialized domains like scientific literature. We believe ScienceBERT represents a significant step forward in making the growing body of scientific literature more accessible through accurate and informative summarization."""
        },
        {
            "id": "sample3",
            "title": "Citation-Aware Scientific Document Summarization",
            "text": """Citations play a crucial role in scientific literature, indicating important concepts and relationships between papers. We present a summarization framework that explicitly models citation context to identify salient information in research articles. By incorporating citation networks into our model architecture, we generate summaries that better capture the key contributions and findings as recognized by the scientific community. Evaluation on computer science and biomedical papers shows substantial improvements in summary quality.

Introduction:
Scientific literature has been growing at an exponential rate, making it increasingly difficult for researchers to keep up with developments even within their specialized fields. Automatic summarization of scientific documents presents a promising solution to this information overload problem. However, scientific papers present unique challenges for summarization systems due to their length, complexity, specialized terminology, and rich network of interconnections through citations.

Citations serve as explicit indicators of important contributions in scientific literature. When paper A cites paper B, the citation context in paper A often highlights the key contributions or findings of paper B. These citation contexts collectively provide valuable signals about which aspects of a paper the scientific community considers most significant.

Previous work on scientific document summarization has primarily focused on the content of individual papers without effectively leveraging the rich information embedded in citation networks. While some studies have explored citation-based summarization, they typically treat citation contexts in isolation without integrating them with the full content of the target paper.

In this work, we present CiteSumm, a citation-aware framework for scientific document summarization that explicitly models the relationship between a paper's content and how it is cited by other papers. Our approach generates summaries that better reflect the paper's impact and key contributions as recognized by the scientific community.

Methods:
CiteSumm consists of the following components:

1. Citation Context Extraction: For each target paper, we extract citation contexts from papers that cite it. A citation context includes the sentence containing the citation and surrounding sentences that discuss the cited work.

2. Citation Context Encoder: We encode each citation context using a SciBERT-based encoder, producing contextualized representations that capture how the paper is perceived by others.

3. Paper Content Encoder: The full text of the target paper is encoded using a Longformer encoder to handle the lengthy input.

4. Cross-Attention Fusion: We employ a cross-attention mechanism to align the paper's content with citation contexts, identifying content sections that correspond to frequently cited aspects.

5. Summary Generation: A decoder generates the final summary, attending to both the paper content representations and the citation context representations.

We train our model using a dataset of scientific papers paired with their abstracts as reference summaries, augmented with citation contexts extracted from citing papers. For papers with fewer than five citations, we use synthetic citation contexts generated from the abstract to ensure consistent training.

For evaluation, we use the ScisummNet corpus (computer science) and the CDSR dataset (biomedical systematic reviews), which provide both full papers and human-written reference summaries.

Results:
We compared CiteSumm against several strong baselines, including BART, PEGASUS, LED, and extractive methods based on citation counting.

On the ScisummNet dataset, CiteSumm achieved ROUGE-1/2/L scores of 48.2/24.5/44.1, outperforming the best baseline (LED) by 2.7/1.9/2.3 points. On the CDSR dataset, our model achieved ROUGE-1/2/L scores of 46.9/23.1/43.2, representing improvements of 2.1/1.8/1.9 points over the strongest baseline.

Human evaluation conducted with 12 domain experts showed that CiteSumm summaries were preferred over baseline summaries in 76% of cases. Experts noted that CiteSumm summaries better captured the papers' key contributions and findings that had influenced subsequent research.

Ablation studies demonstrated that both the citation contexts and the cross-attention fusion mechanism contributed significantly to performance gains. Removing either component led to substantial decreases in ROUGE scores.

We also found that the quality of summaries positively correlated with the number of available citation contexts, with papers having more citations generally receiving better summaries.

Discussion:
Our results demonstrate the effectiveness of leveraging citation information for scientific document summarization. CiteSumm summaries focus more on the aspects of papers that have influenced the scientific community, as evidenced by citations.

Qualitative analysis revealed several interesting patterns:

1. Emphasis on Impact: CiteSumm summaries give greater weight to novel contributions and findings that generated follow-up research.

2. Methodology Focus: Technical details and methodologies that are frequently cited receive more attention in the summaries.

3. Community Perspective: The summaries reflect how the paper is perceived by the scientific community, sometimes differing from the emphasis in the original abstract.

A limitation of our approach is its dependence on citation information, making it less effective for recently published papers with few citations. Future work will explore methods for handling papers with limited citation data.

Conclusion:
We presented CiteSumm, a citation-aware framework for scientific document summarization that leverages how papers are cited to identify and emphasize key contributions. By incorporating citation contexts into the summarization process, our approach generates summaries that better reflect a paper's impact on the scientific community.

Experimental results on computer science and biomedical papers demonstrate substantial improvements over strong baselines, highlighting the value of citation information for scientific summarization. This work represents a step toward more impact-focused summarization of scientific literature, helping researchers more efficiently identify influential contributions in their fields."""
        }
    ]
    return papers
