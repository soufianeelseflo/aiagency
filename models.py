# Filename: models.py
# Description: SQLAlchemy ORM Models for the AI Agency Database (Postgres Focused).
# Version: 5.0 (Level 50+ Transmutation)

import json
import uuid as uuid_pkg
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Index,
    UniqueConstraint, func, event, DDL
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID, JSONB # Use JSONB for tags
from sqlalchemy.orm import declarative_base, relationship

# Define Base using modern approach
Base = declarative_base()

# Helper function for default timestamps with UTC timezone
def utcnow():
    return datetime.now(timezone.utc)


class Client(Base):
    __tablename__ = 'clients'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, index=True, nullable=True, unique=True) # Unique email
    phone = Column(String, index=True, nullable=True, unique=True) # Unique phone
    company = Column(String, nullable=True, index=True)
    job_title = Column(String, nullable=True, index=True) # Index job title
    country = Column(String(2), index=True, nullable=True) # Use 2-char code if standard
    timezone = Column(String, default="America/New_York")
    interests = Column(JSONB, nullable=True, index=True, postgresql_using='gin') # Use JSONB for interests/tags
    last_interaction = Column(DateTime(timezone=True), default=utcnow, index=True)
    last_contacted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_opened_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_replied_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_enriched_at = Column(DateTime(timezone=True), nullable=True, index=True) # Track enrichment time
    engagement_score = Column(Float, default=0.1, nullable=False, index=True)
    opt_in = Column(Boolean, default=True, nullable=False, index=True)
    is_deliverable = Column(Boolean, default=True, nullable=False, index=True)
    source = Column(String, nullable=True, index=True)
    source_reference = Column(Text, nullable=True, index=True) # Changed to Text for longer URLs
    assigned_agent = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    industry = Column(String, nullable=True, index=True) # Added industry
    location = Column(String, nullable=True) # Added location

    # Relationships
    email_logs = relationship("EmailLog", back_populates="client", cascade="all, delete-orphan")
    call_logs = relationship("CallLog", back_populates="client", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="client", cascade="all, delete-orphan")
    knowledge_fragments = relationship("KnowledgeFragment", back_populates="client") # Don't delete KFs if client deleted
    conversation_states = relationship("ConversationState", back_populates="client", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(id={self.id}, name='{self.name}', email='{self.email}', opt_in={self.opt_in})>"

class EmailLog(Base):
    __tablename__ = 'email_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='SET NULL'), index=True, nullable=True) # Set null on client delete
    recipient = Column(String, index=True, nullable=False)
    subject = Column(Text, nullable=True)
    content_preview = Column(Text, nullable=True)
    status = Column(String, index=True, nullable=False) # 'sent', 'delivered', 'opened', 'responded', 'bounced', 'failed_send', 'blocked_compliance', 'flagged_spam_risk', 'failed_verification', 'error_internal'
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True, index=True)
    responded_at = Column(DateTime(timezone=True), nullable=True, index=True)
    agent_version = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    sender_account = Column(String, nullable=True, index=True)
    message_id = Column(Text, unique=True, index=True, nullable=True) # Provider Message-ID
    tracking_pixel_id = Column(UUID(as_uuid=True), unique=True, index=True, nullable=True, default=uuid_pkg.uuid4)

    # Relationship
    client = relationship("Client", back_populates="email_logs")
    composition = relationship("EmailComposition", back_populates="email_log", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<EmailLog(id={self.id}, recipient='{self.recipient}', status='{self.status}')>"

class CallLog(Base):
    __tablename__ = 'call_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='SET NULL'), index=True, nullable=True) # Set null on client delete
    call_sid = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    duration_seconds = Column(Integer, nullable=True)
    transcript = Column(Text, nullable=True) # JSON string of conversation turns
    outcome = Column(String, index=True, nullable=True)
    recording_url = Column(Text, nullable=True) # Changed to Text
    final_twilio_status = Column(String, nullable=True, index=True)
    agent_version = Column(String, nullable=True) # Added agent version

    # Relationship
    client = relationship("Client", back_populates="call_logs")

    def __repr__(self):
        return f"<CallLog(id={self.id}, call_sid='{self.call_sid}', outcome='{self.outcome}')>"

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    call_sid = Column(String, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='SET NULL'), index=True, nullable=True) # Set null on client delete
    state = Column(String, nullable=False)
    conversation_log = Column(Text) # JSON string
    discovered_needs_log = Column(Text, nullable=True) # JSON string of discovered needs
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationship
    client = relationship("Client", back_populates="conversation_states")

    def __repr__(self):
        return f"<ConversationState(call_sid='{self.call_sid}', state='{self.state}')>"

class Invoice(Base):
    __tablename__ = 'invoices'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=False) # Required client
    amount = Column(Float, nullable=False)
    status = Column(String, default='pending', nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    due_date = Column(DateTime(timezone=True), nullable=True)
    invoice_path = Column(Text, nullable=True) # Path to stored PDF on VPS
    payment_link = Column(Text, nullable=True) # Link to payment processor
    source_reference = Column(String, nullable=True, index=True) # e.g., CallLog.call_sid
    invoice_number = Column(String, unique=True, index=True, nullable=True) # Optional invoice number
    notes = Column(Text, nullable=True) # Additional notes

    # Relationship
    client = relationship("Client", back_populates="invoices")

    def __repr__(self):
        return f"<Invoice(id={self.id}, client_id={self.client_id}, amount={self.amount}, status='{self.status}')>"

class AccountCredentials(Base):
    """Stores credentials for various service accounts."""
    __tablename__ = 'account_credentials'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True)
    account_identifier = Column(String, nullable=False, index=True) # Email, username, etc.
    api_key = Column(Text, nullable=True) # Encrypted
    password = Column(Text, nullable=True) # Encrypted
    proxy_used = Column(Text, nullable=True) # Proxy associated with this account
    status = Column(String, default='active', nullable=False, index=True) # 'active', 'limited', 'banned', 'expired', 'needs_review', 'unknown'
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used = Column(DateTime(timezone=True), nullable=True, index=True)
    last_status_update_ts = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow) # Track when status changed
    notes = Column(Text, nullable=True) # e.g., {"trial_expiry": "...", "usage_limits": "...", "2fa_backup_codes": [...]}
    metadata = Column(JSONB, nullable=True) # Store structured metadata if needed

    __table_args__ = (UniqueConstraint('service', 'account_identifier', name='uq_account_cred_service_identifier'),)

    def __repr__(self):
        return f"<AccountCredentials(id={self.id}, service='{self.service}', identifier='{self.account_identifier}', status='{self.status}')>"

class ExpenseLog(Base):
    """Tracks operational expenses."""
    __tablename__ = 'expense_logs'
    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False, index=True) # 'LLM', 'API_Clay', 'API_Twilio', 'API_Deepgram', 'Proxy', 'Resource', 'ConcurrencyAdjustment'
    description = Column(Text, nullable=False) # Details about the expense
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    agent_source = Column(String, nullable=True, index=True) # Which agent reported the expense

    def __repr__(self):
        return f"<ExpenseLog(id={self.id}, category='{self.category}', amount={self.amount})>"

class MigrationStatus(Base):
    """Tracks the completion status of data migrations."""
    __tablename__ = 'migration_status'
    migration_name = Column(String, primary_key=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        status = "Completed" if self.completed_at else "Pending"
        return f"<MigrationStatus(name='{self.migration_name}', status='{status}')>"

# --- Knowledge Base & Strategy Models ---

class KnowledgeFragment(Base):
    """Stores atomic pieces of information gathered or generated by agents."""
    __tablename__ = 'knowledge_fragments'
    id = Column(Integer, primary_key=True)
    agent_source = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    last_accessed_ts = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    data_type = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False) # JSON string for structured data
    item_hash = Column(String(64), unique=True, index=True, nullable=False) # SHA-256 hash REQUIRED
    relevance_score = Column(Float, default=0.5, nullable=False, index=True)
    tags = Column(JSONB, nullable=True, index=True, postgresql_using='gin') # Store as actual JSONB array for better querying
    related_client_id = Column(Integer, ForeignKey('clients.id', ondelete='SET NULL'), nullable=True, index=True)
    source_reference = Column(Text, nullable=True, index=True)
    related_directive_id = Column(Integer, ForeignKey('strategic_directives.id', ondelete='SET NULL'), nullable=True, index=True) # Link to directive

    # Relationships
    client = relationship("Client", back_populates="knowledge_fragments")
    directive = relationship("StrategicDirective", back_populates="knowledge_fragments") # Added relationship

    def __repr__(self):
        return f"<KnowledgeFragment(id={self.id}, type='{self.data_type}', source='{self.agent_source}', hash='{self.item_hash[:8]}...')>"

class LearnedPattern(Base):
    """Stores correlations, insights, and potential causal links discovered by ThinkTool."""
    __tablename__ = 'learned_patterns'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    pattern_description = Column(Text, nullable=False)
    supporting_fragment_ids = Column(JSONB, nullable=False) # Store as JSON array of int IDs
    confidence_score = Column(Float, default=0.5, nullable=False, index=True)
    implications = Column(Text, nullable=True)
    tags = Column(JSONB, nullable=True, index=True, postgresql_using='gin') # Use JSONB
    status = Column(String, default='active', index=True)
    pattern_type = Column(String, default='observational', index=True, nullable=False) # 'observational', 'causal', 'exploit_hypothesis'
    potential_exploit_details = Column(Text, nullable=True) # Specifics if it's an exploit hypothesis

    def __repr__(self):
        return f"<LearnedPattern(id={self.id}, type='{self.pattern_type}', confidence={self.confidence_score:.2f}, status='{self.status}')>"

class StrategicDirective(Base):
    """Stores high-level instructions generated by ThinkTool to guide agents."""
    __tablename__ = 'strategic_directives'
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False, index=True) # Added index
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    target_agent = Column(String, nullable=False, index=True)
    directive_type = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False) # JSON string
    priority = Column(Integer, default=5, nullable=False, index=True)
    status = Column(String, default='pending', nullable=False, index=True)
    expiry_timestamp = Column(DateTime(timezone=True), nullable=True)
    result_summary = Column(Text, nullable=True) # Result message from agent
    notes = Column(Text, nullable=True) # Additional notes, e.g., ROI/Risk from ThinkTool

    # Relationship
    knowledge_fragments = relationship("KnowledgeFragment", back_populates="directive")

    def __repr__(self):
        return f"<StrategicDirective(id={self.id}, type='{self.directive_type}', target='{self.target_agent}', status='{self.status}')>"

class PromptTemplate(Base):
    """Stores and manages prompts used by agents, enabling dynamic updates."""
    __tablename__ = 'prompt_templates'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False, index=True)
    prompt_key = Column(String, nullable=False, index=True)
    version = Column(Integer, default=1, nullable=False)
    content = Column(Text, nullable=False) # The actual prompt template text
    is_active = Column(Boolean, default=True, nullable=False)
    author_agent = Column(String, default="Human", nullable=False) # Added nullable=False
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    notes = Column(Text, nullable=True) # e.g., Critique summary leading to this version

    __table_args__ = (
        UniqueConstraint('agent_name', 'prompt_key', 'version', name='uq_prompt_template_version'),
        Index('ix_active_prompt_template', 'agent_name', 'prompt_key', unique=True, postgresql_where=(is_active == True)),
    )

    def __repr__(self):
        return f"<PromptTemplate(id={self.id}, agent='{self.agent_name}', key='{self.prompt_key}', v={self.version}, active={self.is_active})>"

class EmailStyles(Base):
    """Stores successful email styles for learning."""
    __tablename__ = 'email_styles'
    id = Column(Integer, primary_key=True)
    style_name = Column(String, unique=True, index=True) # e.g., 'Hormozi_Value_Stack', 'Concise_Challenger'
    content_hash = Column(String(64), index=True, nullable=False) # SHA256 hash of combined subject+body template
    body_template = Column(Text, nullable=True)
    subject_template = Column(Text, nullable=True)
    performance_score = Column(Float, default=0.5, index=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True)
    tags = Column(JSONB, nullable=True, index=True, postgresql_using='gin')

    def __repr__(self):
        return f"<EmailStyle(id={self.id}, name='{self.style_name}', score={self.performance_score:.2f})>"

class EmailComposition(Base):
    """Links a sent email (EmailLog) to the specific knowledge fragments and style used."""
    __tablename__ = 'email_composition'
    id = Column(Integer, primary_key=True)
    email_log_id = Column(Integer, ForeignKey('email_logs.id', ondelete='CASCADE'), unique=True, nullable=False, index=True) # Cascade delete
    subject_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id', ondelete='SET NULL'), nullable=True)
    hook_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id', ondelete='SET NULL'), nullable=True)
    body_snippets_kf_ids = Column(ARRAY(Integer), nullable=True)
    cta_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id', ondelete='SET NULL'), nullable=True)
    style_id = Column(Integer, ForeignKey('email_styles.id', ondelete='SET NULL'), nullable=True, index=True)
    llm_generation_metadata = Column(JSONB, nullable=True) # Store model used, temp, tokens, etc.
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    email_log = relationship("EmailLog", back_populates="composition")
    style = relationship("EmailStyles")
    # Define relationships to KnowledgeFragment if needed, using primaryjoin for disambiguation if necessary

    def __repr__(self):
        return f"<EmailComposition(id={self.id}, email_log_id={self.email_log_id})>"

class KVStore(Base):
    """Generic Key-Value store for caching or temporary data."""
    __tablename__ = 'kv_store'
    key = Column(Text, primary_key=True)
    value = Column(Text, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)

    def __repr__(self):
        return f"<KVStore(key='{self.key}', expires_at='{self.expires_at}')>"

# --- End of models.py ---