from app import db
from datetime import datetime

class Paper(db.Model):
    """Model for research papers"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    abstract = db.Column(db.Text, nullable=True)
    full_text = db.Column(db.Text, nullable=True)
    paper_type = db.Column(db.String(100), nullable=True)
    authors = db.Column(db.String(500), nullable=True)
    keywords = db.Column(db.String(500), nullable=True)
    domain = db.Column(db.String(100), nullable=True)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with summaries
    summaries = db.relationship('Summary', backref='paper', lazy=True)

    def __repr__(self):
        return f'<Paper {self.title}>'

class Summary(db.Model):
    """Model for paper summaries"""
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey('paper.id'), nullable=False)
    summary_text = db.Column(db.Text, nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # extractive, abstractive, or hybrid
    rouge1_score = db.Column(db.Float, nullable=True)
    rouge2_score = db.Column(db.Float, nullable=True)
    rougeL_score = db.Column(db.Float, nullable=True)
    bleu_score = db.Column(db.Float, nullable=True)
    generation_time = db.Column(db.Float, nullable=True)  # in seconds
    max_length = db.Column(db.Integer, nullable=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Summary {self.id} for Paper {self.paper_id}>'

class SummarizationRequest(db.Model):
    """Model for tracking user summarization requests"""
    id = db.Column(db.Integer, primary_key=True)
    text_input = db.Column(db.Text, nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    max_length = db.Column(db.Integer, nullable=False)
    summary_output = db.Column(db.Text, nullable=True)
    processing_time = db.Column(db.Float, nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    request_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Request {self.id}>'
