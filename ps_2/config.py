import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Application configuration
class Config:
    """Base configuration class"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SESSION_SECRET', 'dev-secret-key')
    
    # Model settings
    DEFAULT_MODEL = 'hybrid'  # 'extractive', 'abstractive', or 'hybrid'
    DEFAULT_MAX_LENGTH = 250  # Default max length for summaries
    
    # Extractive model settings
    EXTRACTIVE_MODEL = 'citation'  # 'bert', 'textrank', or 'citation'
    
    # Abstractive model settings
    ABSTRACTIVE_MODEL = 'science'  # 'transformer', 'long', or 'science'
    
    # Device settings
    DEVICE = 'cuda' if os.environ.get('USE_GPU', 'False').lower() == 'true' and torch.cuda.is_available() else 'cpu'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
    # Override model settings for faster development
    EXTRACTIVE_MODEL = 'textrank'  # TextRank is faster than BERT

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # Use best models in production
    EXTRACTIVE_MODEL = 'citation'
    ABSTRACTIVE_MODEL = 'science'

# Select configuration based on environment
config_by_name = {
    'dev': DevelopmentConfig,
    'test': TestingConfig,
    'prod': ProductionConfig
}

# Get configuration
active_config = config_by_name[os.environ.get('FLASK_ENV', 'dev')]
