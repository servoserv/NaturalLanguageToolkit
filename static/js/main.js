document.addEventListener('DOMContentLoaded', function() {
    // Handle file option selection
    const fileOption = document.getElementById('file_option');
    const customTextArea = document.getElementById('text_input');
    const fileOptionInfo = document.getElementById('file_option_info');

    if (fileOption) {
        fileOption.addEventListener('change', function() {
            if (this.value === 'custom') {
                customTextArea.disabled = false;
                customTextArea.value = '';
                customTextArea.placeholder = 'Paste or type your scientific article text here...';
                if (fileOptionInfo) fileOptionInfo.textContent = '';
            } else {
                customTextArea.disabled = true;
                customTextArea.placeholder = 'Using sample paper...';
                
                // Show info about selected paper
                if (fileOptionInfo) {
                    if (this.value === 'sample1') {
                        fileOptionInfo.textContent = 'Selected: "Transformer-based Models for Long Document Summarization"';
                    } else if (this.value === 'sample2') {
                        fileOptionInfo.textContent = 'Selected: "Domain-Specific Pre-training for Scientific Text Summarization"';
                    } else if (this.value === 'sample3') {
                        fileOptionInfo.textContent = 'Selected: "Citation-Aware Scientific Document Summarization"';
                    }
                }
            }
        });
    }

    // Handle model type selection
    const modelType = document.getElementById('model_type');
    const modelInfo = document.getElementById('model_info');

    if (modelType && modelInfo) {
        modelType.addEventListener('change', function() {
            switch(this.value) {
                case 'extractive':
                    modelInfo.innerHTML = '<strong>Extractive:</strong> Identifies and extracts key sentences from the original text.';
                    break;
                case 'abstractive':
                    modelInfo.innerHTML = '<strong>Abstractive:</strong> Generates a summary with new sentences that capture the essence of the text.';
                    break;
                case 'hybrid':
                    modelInfo.innerHTML = '<strong>Hybrid:</strong> Combines extractive and abstractive approaches for better scientific summaries.';
                    break;
                default:
                    modelInfo.innerHTML = '';
            }
        });
        
        // Trigger change event to show initial info
        modelType.dispatchEvent(new Event('change'));
    }

    // Handle max length slider
    const lengthSlider = document.getElementById('max_length');
    const lengthValue = document.getElementById('length_value');

    if (lengthSlider && lengthValue) {
        lengthSlider.addEventListener('input', function() {
            lengthValue.textContent = this.value;
        });
        
        // Set initial value
        lengthValue.textContent = lengthSlider.value;
    }

    // Form validation
    const summarizeForm = document.getElementById('summarize_form');
    
    if (summarizeForm) {
        summarizeForm.addEventListener('submit', function(e) {
            const fileOption = document.getElementById('file_option').value;
            const textInput = document.getElementById('text_input').value;
            
            if (fileOption === 'custom' && (!textInput || textInput.trim().length < 100)) {
                e.preventDefault();
                alert('Please enter a longer text to summarize (at least 100 characters).');
            }
        });
    }

    // Handle copy summary button
    const copyButton = document.getElementById('copy_summary');
    const summaryText = document.getElementById('summary_text');
    
    if (copyButton && summaryText) {
        copyButton.addEventListener('click', function() {
            // Create a temporary textarea element
            const textarea = document.createElement('textarea');
            textarea.value = summaryText.textContent;
            document.body.appendChild(textarea);
            
            // Select and copy the text
            textarea.select();
            document.execCommand('copy');
            
            // Remove the temporary textarea
            document.body.removeChild(textarea);
            
            // Show feedback
            const originalText = this.textContent;
            this.textContent = 'Copied!';
            
            // Reset button text after 2 seconds
            setTimeout(() => {
                this.textContent = originalText;
            }, 2000);
        });
    }

    // Toggle sections on about page
    const toggleButtons = document.querySelectorAll('.toggle-section');
    
    if (toggleButtons.length > 0) {
        toggleButtons.forEach(button => {
            button.addEventListener('click', function() {
                const targetId = this.getAttribute('data-target');
                const targetSection = document.getElementById(targetId);
                
                if (targetSection) {
                    // Toggle section visibility
                    if (targetSection.classList.contains('d-none')) {
                        targetSection.classList.remove('d-none');
                        this.textContent = 'Hide Details';
                    } else {
                        targetSection.classList.add('d-none');
                        this.textContent = 'Show Details';
                    }
                }
            });
        });
    }

    // Add tooltip initialization if Bootstrap is present
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});
