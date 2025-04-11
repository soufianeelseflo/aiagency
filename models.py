# models.py
import json
from datetime import datetime
import pytz
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Index, UniqueConstraint, func
)
from sqlalchemy.orm import relationship # Kept for potential future use, though not explicitly required by current task
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Helper function for default timestamps with UTC timezone
def utcnow():
    return datetime.now(pytz.UTC)

# --- Existing Models (Enhanced) ---

class Client(Base):
    __tablename__ = 'clients'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, index=True, nullable=True) # Added index=True, ensure nullable if applicable
    country = Column(String)
    timezone = Column(String, default="America/New_York")
    interests = Column(String)  # Comma-separated or JSON
    last_interaction = Column(DateTime(timezone=True), default=utcnow) # Ensure timezone aware default
    response_rate = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    opt_in = Column(Boolean, default=True)
    is_deliverable = Column(Boolean, default=True, nullable=False) # Added field

    # Potential future relationships (optional)
    # logs = relationship("EmailLog", back_populates="client")
    # calls = relationship("CallLog", back_populates="client")
    # knowledge_fragments = relationship("KnowledgeFragment", back_populates="related_client")

class Metric(Base): # No changes specified for Metric
    __tablename__ = 'metrics'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    metric_name = Column(String)
    value = Column(Text)

class CallLog(Base):
    __tablename__ = 'call_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True) # Added index=True
    call_sid = Column(String, unique=True, index=True, nullable=True) # Added field
    timestamp = Column(DateTime(timezone=True), default=utcnow) # Ensure timezone aware default
    conversation = Column(Text)  # JSON string
    outcome = Column(String, index=True)

    # Potential future relationship (optional)
    # client = relationship("Client", back_populates="calls")

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    call_sid = Column(String, primary_key=True) # Twilio Call SID
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True) # Verified field exists
    state = Column(String)
    conversation_log = Column(Text)  # JSON string
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow) # Verified field exists and uses UTC

class Invoice(Base):
    __tablename__ = 'invoices'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True) # Added index=True
    amount = Column(Float)
    status = Column(String, default='pending', index=True)  # Added index=True
    timestamp = Column(DateTime(timezone=True), default=utcnow) # Ensure timezone aware default
    invoice_path = Column(String, nullable=True) # Renamed from invoice_pdf_path

class Campaign(Base): # No changes specified for Campaign
    __tablename__ = 'campaigns'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    status = Column(String)
    start_date = Column(DateTime(timezone=True), default=utcnow)
    end_date = Column(DateTime(timezone=True), nullable=True)

class EmailLog(Base):
    __tablename__ = 'email_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True) # Verified field exists, ensure nullable
    recipient = Column(String, index=True) # Verified field exists and indexed
    subject = Column(String)
    content = Column(Text) # Encrypted core body
    status = Column(String, index=True)  # Verified field exists and indexed
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True) # Ensure timezone aware default and indexed
    opened_at = Column(DateTime(timezone=True), nullable=True) # Verified field exists
    responded_at = Column(DateTime(timezone=True), nullable=True) # Added field
    spam_flagged = Column(Boolean, default=False) # Verified field exists
    agent_version = Column(String, nullable=True) # Verified field exists, ensure nullable

    # Potential future relationship (optional)
    # client = relationship("Client", back_populates="logs")

class Account(Base):
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True)
    service = Column(String, index=True) # Added index=True
    email = Column(String, index=True) # Added index=True
    vault_path = Column(String, unique=True, nullable=True) # Verified field exists, ensure nullable
    api_key = Column(String, nullable=True, index=True) # Added index=True
    phone = Column(String, nullable=True)
    cookies = Column(Text, nullable=True)  # JSON-encoded cookies
    created_at = Column(DateTime(timezone=True), default=utcnow) # Ensure timezone aware default
    last_used = Column(DateTime(timezone=True), nullable=True) # Added field
    is_available = Column(Boolean, default=True, index=True) # Added index=True
    is_recurring = Column(Boolean, default=False) # Added field
    credit_status = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    # Add unique constraint for service/email if not already covered by vault_path unique
    # __table_args__ = (UniqueConstraint('service', 'email', name='uq_account_service_email'),) # Keep original if needed, vault_path unique might suffice

# --- NEW MODELS ---

class KnowledgeFragment(Base):
    """Stores atomic pieces of information gathered or generated by agents."""
    __tablename__ = 'knowledge_fragments'
    id = Column(Integer, primary_key=True)
    agent_source = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    data_type = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False) # Potentially JSON encoded
    relevance_score = Column(Float, default=0.5, nullable=False)
    tags = Column(Text, nullable=True, index=True) # JSON array string or comma-separated
    related_client_id = Column(Integer, ForeignKey('clients.id'), nullable=True, index=True)
    source_reference = Column(String, nullable=True, index=True)

    # Optional relationship back to Client
    # related_client = relationship("Client", back_populates="knowledge_fragments")

class StrategicDirective(Base):
    """Stores high-level instructions to guide agent behavior or test hypotheses."""
    __tablename__ = 'strategic_directives'
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False) # 'ThinkTool', 'Human', 'OptimizationAgent'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    target_agent = Column(String, nullable=False, index=True) # 'EmailAgent', 'All'
    directive_type = Column(String, nullable=False, index=True) # 'test_strategy', 'prioritize_target'
    content = Column(Text, nullable=False)
    priority = Column(Integer, default=5, nullable=False, index=True)
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'active', 'completed', 'failed', 'expired'
    expiry_timestamp = Column(DateTime(timezone=True), nullable=True)
    result_summary = Column(Text, nullable=True)

class LearnedPattern(Base):
    """Stores correlations, insights, and potential causal links discovered."""
    __tablename__ = 'learned_patterns'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    pattern_description = Column(Text, nullable=False)
    supporting_fragment_ids = Column(Text, nullable=False) # JSON array string or comma-separated
    confidence_score = Column(Float, default=0.5, nullable=False, index=True)
    implications = Column(Text, nullable=True) # Made nullable
    tags = Column(Text, nullable=True, index=True) # JSON array string or comma-separated

class PromptTemplate(Base):
    """Stores and manages prompts used by various agents."""
    __tablename__ = 'prompt_templates'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False, index=True)
    prompt_key = Column(String, nullable=False, index=True)
    version = Column(Integer, default=1, nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    # Unique constraint for active prompt per agent/key
    __table_args__ = (
        UniqueConstraint('agent_name', 'prompt_key', 'version', name='uq_prompt_template_version'), # Keep version unique too
        Index('ix_active_prompt_template', 'agent_name', 'prompt_key', unique=True, postgresql_where=(is_active == True)), # Partial unique index for active prompts
    )

class OSINTData(Base):
    """Stores results from OSINT gathering activities."""
    __tablename__ = 'osint_data'
    id = Column(Integer, primary_key=True)
    target = Column(String, nullable=False, index=True) # e.g., domain, email, name
    tools_used = Column(Text, nullable=True) # JSON array string or comma-separated
    raw_data = Column(Text, nullable=False) # Could be large, potentially JSON
    analysis_results = Column(Text, nullable=True) # Summary/insights from LLM analysis
    relevance = Column(String, nullable=True) # e.g., 'high', 'medium', 'low'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

class ExpenseLog(Base):
    """Tracks expenses incurred by the agency, especially API costs."""
    __tablename__ = 'expense_logs'
    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False, index=True) # e.g., 'API', 'Software', 'Infrastructure'
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    agent_source = Column(String, nullable=True) # Which agent reported/caused the expense

class GeminiResult(Base):
    """Stores raw results from Gemini API calls for logging or debugging."""
    __tablename__ = 'gemini_results'
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=True) # e.g., 'visual_analysis', 'validation'
    data = Column(Text, nullable=False) # Store the raw response data (JSON string)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)

class ServiceCredit(Base):
    """Tracks available credits for various external services."""
    __tablename__ = 'service_credits'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True) # e.g., 'OpenAI', 'Twilio', 'DeepSeek'
    credits = Column(String, nullable=False) # Store as string to handle various formats (e.g., '1000', '$5.00', 'unlimited')
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True) # Last time checked/updated
