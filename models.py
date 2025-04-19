# models.py
# Genius-Level Implementation v1.0 - Foundation for Agentic Learning & Strategy

import json
from datetime import datetime, timezone # Use timezone directly
import pytz # Keep pytz for timezone object creation where needed
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Index, UniqueConstraint, func
)
# Use modern declarative_base from sqlalchemy.orm
from sqlalchemy.orm import relationship, declarative_base

# Define Base using modern approach
Base = declarative_base()

# Helper function for default timestamps with UTC timezone
def utcnow():
    return datetime.now(timezone.utc)

# --- Existing Models (Enhanced for Genius Mandate) ---

class Client(Base):
    __tablename__ = 'clients'
    # Core Info
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, index=True, nullable=True, unique=True) # Email should ideally be unique
    phone = Column(String, index=True, nullable=True, unique=True) # Added phone, make unique if primary contact
    country = Column(String, index=True) # Index for filtering
    timezone = Column(String, default="America/New_York") # Keep default, but allow override

    # Engagement & Status
    interests = Column(Text, nullable=True)  # Use Text for potentially long JSON or comma-list
    last_interaction = Column(DateTime(timezone=True), default=utcnow, index=True) # Index for recency queries
    response_rate = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.3, index=True) # Default starting score, index for prioritization
    opt_in = Column(Boolean, default=True, nullable=False) # Ensure non-nullable
    is_deliverable = Column(Boolean, default=True, nullable=False, index=True) # Track bounces/blocks, index

    # Operational Info
    source = Column(String, nullable=True, index=True) # Track lead source (OSINT, Manual, etc.)
    assigned_agent = Column(String, nullable=True) # Track primary contact agent if needed

    def __repr__(self):
        return f"<Client(id={self.id}, name='{self.name}', email='{self.email}')>"

class Metric(Base):
    __tablename__ = 'metrics'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    metric_name = Column(String, index=True) # Index for easier querying
    value = Column(Text) # Keep as Text for flexibility (JSON, numbers, strings)
    tags = Column(Text, nullable=True) # Optional JSON tags for context (e.g., {"client_id": 123, "task_type": "email"})

    def __repr__(self):
        return f"<Metric(id={self.id}, agent='{self.agent_name}', metric='{self.metric_name}')>"

class CallLog(Base):
    __tablename__ = 'call_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True) # Allow calls not linked to client initially?
    call_sid = Column(String, unique=True, index=True, nullable=False) # Twilio SID is crucial identifier
    phone_number = Column(String, index=True) # Store the number called
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    duration_seconds = Column(Integer, nullable=True) # Store call duration
    conversation = Column(Text)  # JSON string of transcript/turns
    outcome = Column(String, index=True) # 'success', 'failed_compliance', 'failed_error', 'voicemail', 'busy', 'no_answer', 'disconnected_client', 'disconnected_error'
    recording_url = Column(String, nullable=True) # Store Twilio recording URL if enabled

    def __repr__(self):
        return f"<CallLog(id={self.id}, call_sid='{self.call_sid}', outcome='{self.outcome}')>"

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    # Use call_sid as primary key for direct lookup during active calls
    call_sid = Column(String, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True) # Link to client if available
    state = Column(String, nullable=False) # Current state in the conversation FSM
    conversation_log = Column(Text)  # Running JSON log of the conversation
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    def __repr__(self):
        return f"<ConversationState(call_sid='{self.call_sid}', state='{self.state}')>"

class Invoice(Base):
    __tablename__ = 'invoices'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True)
    amount = Column(Float, nullable=False)
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'paid', 'failed', 'cancelled'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    due_date = Column(DateTime(timezone=True), nullable=True)
    invoice_path = Column(String, nullable=True) # Path to stored PDF on VPS
    payment_link = Column(String, nullable=True) # Link to payment processor if integrated

    def __repr__(self):
        return f"<Invoice(id={self.id}, client_id={self.client_id}, amount={self.amount}, status='{self.status}')>"

class Campaign(Base):
    __tablename__ = 'campaigns'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    status = Column(String, index=True) # 'planning', 'active', 'paused', 'completed', 'archived'
    goal = Column(Text, nullable=True) # Description of the campaign goal
    start_date = Column(DateTime(timezone=True), default=utcnow)
    end_date = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<Campaign(id={self.id}, name='{self.name}', status='{self.status}')>"

class EmailLog(Base):
    __tablename__ = 'email_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True)
    recipient = Column(String, index=True, nullable=False)
    subject = Column(String)
    content_preview = Column(Text, nullable=True) # Store first N chars unencrypted for quick checks?
    status = Column(String, index=True, nullable=False) # 'sent', 'delivered', 'opened', 'clicked', 'responded', 'bounced', 'failed', 'blocked'
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True, index=True) # Index for engagement calcs
    responded_at = Column(DateTime(timezone=True), nullable=True, index=True) # Index for engagement calcs
    spam_flagged = Column(Boolean, default=False)
    agent_version = Column(String, nullable=True)
    error_message = Column(Text, nullable=True) # Store reason for failure/bounce

    def __repr__(self):
        return f"<EmailLog(id={self.id}, recipient='{self.recipient}', status='{self.status}')>"

class Account(Base):
    """Stores credentials (via Vault path) and status for external service accounts."""
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True) # e.g., 'openrouter.ai', 'twilio.com', 'deepgram.com'
    email = Column(String, index=True, nullable=True) # Login identifier
    username = Column(String, index=True, nullable=True) # Alternative login identifier
    # Store path to secrets in Vault, NOT secrets directly in DB
    vault_path = Column(String, unique=True, nullable=False) # Path in HCP Vault REQUIRED
    # api_key = Column(String, nullable=True, index=True) # DEPRECATED
    phone = Column(String, nullable=True) # Associated phone if any (stored in Vault)
    cookies = Column(Text, nullable=True)  # JSON-encoded cookies for session reuse (stored in Vault)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used = Column(DateTime(timezone=True), nullable=True, index=True) # Index for LRU logic
    is_available = Column(Boolean, default=True, nullable=False, index=True) # Availability for use
    is_recurring = Column(Boolean, default=False) # Does it have recurring credits?
    credit_status = Column(String, nullable=True) # Last known credit status (e.g., 'OK', 'Low', 'Empty')
    notes = Column(Text, nullable=True) # Any relevant notes about the account

    # Unique constraint on service+email might be useful if vault_path isn't guaranteed unique by creation logic
    __table_args__ = (UniqueConstraint('service', 'email', name='uq_account_service_email'),)

    def __repr__(self):
        return f"<Account(id={self.id}, service='{self.service}', email='{self.email}', available={self.is_available})>"

# --- NEW KNOWLEDGE BASE MODELS ---

class KnowledgeFragment(Base):
    """Stores atomic pieces of information gathered or generated by agents."""
    __tablename__ = 'knowledge_fragments'
    id = Column(Integer, primary_key=True)
    agent_source = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    data_type = Column(String, nullable=False, index=True) # Crucial for filtering
    content = Column(Text, nullable=False) # Store as JSON string for structured data
    relevance_score = Column(Float, default=0.5, nullable=False) # AI-assessed relevance
    tags = Column(Text, nullable=True, index=True) # JSON array string for flexible tagging/search
    related_client_id = Column(Integer, ForeignKey('clients.id'), nullable=True, index=True)
    source_reference = Column(String, nullable=True, index=True) # Link back to source (log ID, URL, etc.)

    def __repr__(self):
        return f"<KnowledgeFragment(id={self.id}, type='{self.data_type}', source='{self.agent_source}')>"

class StrategicDirective(Base):
    """Stores high-level instructions generated by ThinkTool to guide agents."""
    __tablename__ = 'strategic_directives'
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False) # 'ThinkToolSynthesis', 'ThinkToolCritique', 'Human', 'OptimizationAgent'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    target_agent = Column(String, nullable=False, index=True) # Specific agent name or 'All'
    directive_type = Column(String, nullable=False, index=True) # e.g., 'test_strategy', 'prioritize_target', 'update_prompt', 'evaluate_tool'
    content = Column(Text, nullable=False) # Detailed instructions, potentially JSON
    priority = Column(Integer, default=5, nullable=False, index=True) # Lower number = higher priority
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'active', 'completed', 'failed', 'expired'
    expiry_timestamp = Column(DateTime(timezone=True), nullable=True)
    result_summary = Column(Text, nullable=True) # Outcome of the directive execution

    def __repr__(self):
        return f"<StrategicDirective(id={self.id}, type='{self.directive_type}', target='{self.target_agent}', status='{self.status}')>"

class LearnedPattern(Base):
    """Stores correlations, insights, and potential causal links discovered by ThinkTool."""
    __tablename__ = 'learned_patterns'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    pattern_description = Column(Text, nullable=False)
    supporting_fragment_ids = Column(Text, nullable=False) # JSON array of KnowledgeFragment IDs
    confidence_score = Column(Float, default=0.5, nullable=False, index=True) # How reliable is this pattern?
    implications = Column(Text, nullable=True) # Strategic value or actionable consequence
    tags = Column(Text, nullable=True, index=True) # JSON array string for categorization
    status = Column(String, default='active', index=True) # 'active', 'obsolete', 'under_review'

    def __repr__(self):
        return f"<LearnedPattern(id={self.id}, confidence={self.confidence_score:.2f}, status='{self.status}')>"

class PromptTemplate(Base):
    """Stores and manages prompts used by agents, enabling dynamic updates."""
    __tablename__ = 'prompt_templates'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False, index=True)
    prompt_key = Column(String, nullable=False, index=True) # e.g., 'email_draft', 'voice_intent'
    version = Column(Integer, default=1, nullable=False)
    content = Column(Text, nullable=False) # The actual prompt template text
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    author_agent = Column(String, default="Human") # Track who created/updated ('Human', 'ThinkToolCritique')
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('agent_name', 'prompt_key', 'version', name='uq_prompt_template_version'),
        # Ensure only one prompt is active per agent/key at a time using partial index (PostgreSQL specific)
        Index('ix_active_prompt_template', 'agent_name', 'prompt_key', unique=True, postgresql_where=(is_active == True)),
    )

    def __repr__(self):
        return f"<PromptTemplate(id={self.id}, agent='{self.agent_name}', key='{self.prompt_key}', v={self.version}, active={self.is_active})>"

# --- Supporting Models (Verified/Enhanced) ---

class OSINTData(Base):
    """Stores results from OSINT gathering activities."""
    __tablename__ = 'osint_data'
    id = Column(Integer, primary_key=True)
    target = Column(String, nullable=False, index=True) # e.g., domain, email, name
    tools_used = Column(Text, nullable=True) # JSON array string
    raw_data = Column(Text, nullable=False) # Could be large, potentially JSON
    analysis_results = Column(Text, nullable=True) # JSON dump from ThinkTool analysis
    relevance = Column(String, nullable=True, index=True) # 'high', 'medium', 'low', 'pending_analysis', 'error'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<OSINTData(id={self.id}, target='{self.target}', relevance='{self.relevance}')>"

class ExpenseLog(Base):
    """Tracks expenses incurred by the agency."""
    __tablename__ = 'expense_logs' # Renamed from 'expenses'
    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False, index=True) # 'API', 'Proxy', 'Tool', 'Infrastructure', 'Concurrency'
    description = Column(Text, nullable=False) # Encrypted description
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    agent_source = Column(String, nullable=True) # Which agent triggered the expense

    def __repr__(self):
        return f"<ExpenseLog(id={self.id}, category='{self.category}', amount={self.amount})>"

class GeminiResult(Base): # Renamed for clarity
    """Stores raw results from Google Gemini API calls for logging or debugging."""
    __tablename__ = 'gemini_results'
    id = Column(Integer, primary_key=True)
    api_call_type = Column(String, nullable=False) # e.g., 'deep_search', 'visual_analysis', 'validation'
    request_data = Column(Text, nullable=True) # Store the prompt/input if needed
    response_data = Column(Text, nullable=False) # Store the raw response
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    def __repr__(self):
        return f"<GeminiResult(id={self.id}, type='{self.api_call_type}')>"

class ServiceCredit(Base):
    """Tracks available credits for external services."""
    __tablename__ = 'service_credits'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True) # e.g., 'OpenAI', 'Twilio', 'DeepSeek'
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=True, index=True) # Link to specific account
    credits_value = Column(String, nullable=False) # Store as string (e.g., '1000', '5.23', 'unlimited')
    credits_unit = Column(String, nullable=True) # e.g., 'tokens', 'USD', 'calls', 'minutes'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True) # Last check time

    # account = relationship("Account") # Optional relationship

    def __repr__(self):
        return f"<ServiceCredit(id={self.id}, service='{self.service}', credits='{self.credits_value}')>"

# --- End of models.py ---