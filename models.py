# Filename: models.py
# Description: SQLAlchemy ORM Models for the AI Agency Database.
# Version: 3.0 (Genius Agentic - Postgres Optimized)

import json
import uuid as uuid_pkg # Alias to avoid conflict with Column name
from datetime import datetime, timezone, timedelta
import pytz

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Index,
    UniqueConstraint, func, event, DDL # Keep func, event, DDL
)
# Import UUID type specifically for PostgreSQL
from sqlalchemy.dialects.postgresql import ARRAY, UUID, JSONB # Use JSONB for better indexing if needed
# Use modern declarative_base from sqlalchemy.orm
from sqlalchemy.orm import relationship, declarative_base

# Define Base using modern approach
Base = declarative_base()

# Helper function for default timestamps with UTC timezone
def utcnow():
    return datetime.now(timezone.utc)

# --- Core Operational Models ---

class Client(Base):
    __tablename__ = 'clients'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, index=True, nullable=True, unique=True) # Unique email
    phone = Column(String, index=True, nullable=True, unique=True) # Unique phone
    country = Column(String, index=True)
    timezone = Column(String, default="America/New_York")
    interests = Column(Text, nullable=True) # Store as JSON string or comma-separated
    last_interaction = Column(DateTime(timezone=True), default=utcnow, index=True)
    last_contacted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_opened_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_replied_at = Column(DateTime(timezone=True), nullable=True, index=True)
    response_rate = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.1, index=True) # ### Phase 1 Plan Ref: 3.2 (Verify/Add score)
    opt_in = Column(Boolean, default=True, nullable=False, index=True) # ### Phase 1 Plan Ref: 3.1 (Add opt_in)
    is_deliverable = Column(Boolean, default=True, nullable=False, index=True)
    source = Column(String, nullable=True, index=True)
    assigned_agent = Column(String, nullable=True)

    def __repr__(self):
        return f"<Client(id={self.id}, name='{self.name}', email='{self.email}')>"

class EmailLog(Base):
    __tablename__ = 'email_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True)
    recipient = Column(String, index=True, nullable=False)
    subject = Column(String)
    content_preview = Column(Text, nullable=True)
    status = Column(String, index=True, nullable=False) # 'sent', 'delivered', 'opened', 'clicked', 'responded', 'bounced', 'failed', 'blocked'
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True, index=True)
    responded_at = Column(DateTime(timezone=True), nullable=True, index=True)
    spam_flagged = Column(Boolean, default=False)
    agent_version = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    sender_account = Column(String, nullable=True, index=True)
    message_id = Column(Text, unique=True, index=True, nullable=True) # ### Phase 1 Plan Ref: 3.2 (Add message_id)
    tracking_pixel_id = Column(UUID(as_uuid=True), unique=True, index=True, nullable=True, default=uuid_pkg.uuid4) # ### Phase 1 Plan Ref: 3.2 (Add tracking_pixel_id)

    def __repr__(self):
        return f"<EmailLog(id={self.id}, recipient='{self.recipient}', status='{self.status}')>"

class CallLog(Base):
    __tablename__ = 'call_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True)
    call_sid = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    duration_seconds = Column(Integer, nullable=True)
    transcript = Column(Text, nullable=True) # ### Phase 1 Plan Ref: 3.3 (Enhance CallLog)
    outcome = Column(Text, nullable=True, index=True) # ### Phase 1 Plan Ref: 3.3 (Changed outcome to Text for detail)
    recording_url = Column(String, nullable=True)
    final_twilio_status = Column(String, nullable=True)

    def __repr__(self):
        return f"<CallLog(id={self.id}, call_sid='{self.call_sid}', outcome='{self.outcome}')>"

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    call_sid = Column(String, primary_key=True) # Use call_sid as PK
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True)
    state = Column(String, nullable=False)
    conversation_log = Column(Text) # JSON string log
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    def __repr__(self):
        return f"<ConversationState(call_sid='{self.call_sid}', state='{self.state}')>"

class Invoice(Base):
    __tablename__ = 'invoices'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'paid', 'failed', 'cancelled'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    due_date = Column(DateTime(timezone=True), nullable=True)
    invoice_path = Column(String, nullable=True)
    payment_link = Column(String, nullable=True)
    # Added W8/Bank reference for payout tracking if needed
    payout_reference = Column(String, nullable=True) # e.g., Transaction ID from payment processor
    w8_status = Column(String, nullable=True) # e.g., 'Provided', 'Missing', 'Requested'

    def __repr__(self):
        return f"<Invoice(id={self.id}, client_id={self.client_id}, amount={self.amount}, status='{self.status}')>"

class AccountCredentials(Base):
    """Stores credentials for rotated free trial or acquired accounts."""
    # ### Phase 1 Plan Ref: 3.6 (ADD AccountCredentials table)
    __tablename__ = 'account_credentials'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True) # e.g., 'clay.com', 'heygen.com', 'facebook.com'
    account_identifier = Column(String, nullable=False, index=True) # e.g., email, username used for login
    api_key = Column(Text, nullable=True) # Encrypted
    password = Column(Text, nullable=True) # Encrypted
    proxy_used = Column(String, nullable=True) # Proxy associated with this account's creation/use
    status = Column(String, default='active', nullable=False, index=True) # 'active', 'limited', 'banned', 'expired'
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used = Column(DateTime(timezone=True), nullable=True, index=True)
    notes = Column(Text, nullable=True) # e.g., Free trial limits, purpose

    __table_args__ = (UniqueConstraint('service', 'account_identifier', name='uq_account_cred_service_identifier'),)

    def __repr__(self):
        return f"<AccountCredentials(id={self.id}, service='{self.service}', identifier='{self.account_identifier}', status='{self.status}')>"

class ExpenseLog(Base):
    """Tracks expenses incurred by the agency."""
    __tablename__ = 'expense_logs'
    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False, index=True) # 'API', 'Proxy', 'Tool', 'Infrastructure', 'Concurrency'
    description = Column(Text, nullable=False) # Encrypted description
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    agent_source = Column(String, nullable=True, index=True)

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
    # ### Phase 1 Plan Ref: 3.8 (Verify KB tables)
    __tablename__ = 'knowledge_fragments'
    id = Column(Integer, primary_key=True)
    agent_source = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    last_accessed_ts = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False, index=True)
    data_type = Column(String, nullable=False, index=True) # e.g., 'email_subject', 'legal_interpretation', 'osint_summary', 'learning_material_summary', 'clay_com_lead'
    content = Column(Text, nullable=False) # The actual knowledge content (string or JSON string)
    item_hash = Column(String(64), unique=True, index=True, nullable=False) # SHA-256 hash REQUIRED
    relevance_score = Column(Float, default=0.5, nullable=False)
    tags = Column(Text, nullable=True) # Store as JSON string '["tag1", "tag2"]'
    related_client_id = Column(Integer, ForeignKey('clients.id'), nullable=True, index=True)
    source_reference = Column(String, nullable=True, index=True) # Link back to source (log ID, URL, task ID etc.)

    # Consider GIN index for tags if using JSONB and heavy tag queries in Postgres
    # __table_args__ = (Index('ix_knowledge_fragments_tags_gin', 'tags', postgresql_using='gin'),)

    def __repr__(self):
        return f"<KnowledgeFragment(id={self.id}, type='{self.data_type}', source='{self.agent_source}', hash='{self.item_hash[:8]}...')>"

class EmailStyles(Base):
    """Stores successful email styles for learning."""
    # ### Phase 1 Plan Ref: 3.4 (Add EmailStyles table)
    __tablename__ = 'email_styles'
    id = Column(Integer, primary_key=True)
    content_hash = Column(String(64), unique=True, index=True, nullable=False) # SHA256 hash of body/subject combo?
    body_template = Column(Text, nullable=True)
    subject_template = Column(Text, nullable=True)
    performance_score = Column(Float, default=0.5, index=True) # Based on open/reply rates
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    tags = Column(Text, nullable=True) # JSON string for style tags (e.g., 'formal', 'short', 'value_prop')

    def __repr__(self):
        return f"<EmailStyle(id={self.id}, score={self.performance_score:.2f}, hash='{self.content_hash[:8]}...')>"

class StrategicDirective(Base):
    """Stores high-level instructions generated by ThinkTool to guide agents."""
    # ### Phase 1 Plan Ref: 3.8 (Verify KB tables)
    __tablename__ = 'strategic_directives'
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False) # 'ThinkToolSynthesis', 'ThinkToolCritique', 'Human', 'OptimizationAgent'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    target_agent = Column(String, nullable=False, index=True) # Specific agent name or 'All'
    directive_type = Column(String, nullable=False, index=True) # e.g., 'test_strategy', 'prioritize_target', 'update_prompt', 'evaluate_tool', 'create_account'
    content = Column(Text, nullable=False) # Detailed instructions, potentially JSON
    priority = Column(Integer, default=5, nullable=False, index=True) # Lower number = higher priority
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'active', 'completed', 'failed', 'expired', 'cancelled'
    expiry_timestamp = Column(DateTime(timezone=True), nullable=True)
    result_summary = Column(Text, nullable=True) # Outcome of the directive execution

    def __repr__(self):
        return f"<StrategicDirective(id={self.id}, type='{self.directive_type}', target='{self.target_agent}', status='{self.status}')>"

class LearnedPattern(Base):
    """Stores correlations, insights, and potential causal links discovered by ThinkTool."""
    # ### Phase 1 Plan Ref: 3.8 (Verify KB tables)
    __tablename__ = 'learned_patterns'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    pattern_description = Column(Text, nullable=False)
    supporting_fragment_ids = Column(Text, nullable=False) # JSON array string of KnowledgeFragment IDs
    confidence_score = Column(Float, default=0.5, nullable=False, index=True)
    implications = Column(Text, nullable=True) # Strategic value or actionable consequence
    tags = Column(Text, nullable=True) # Store as JSON string '["tag1", "tag2"]'
    status = Column(String, default='active', index=True) # 'active', 'obsolete', 'under_review'

    # Consider GIN index for tags if using JSONB and heavy tag queries in Postgres
    # __table_args__ = (Index('ix_learned_patterns_tags_gin', 'tags', postgresql_using='gin'),)

    def __repr__(self):
        return f"<LearnedPattern(id={self.id}, confidence={self.confidence_score:.2f}, status='{self.status}')>"

class PromptTemplate(Base):
    """Stores and manages prompts used by agents, enabling dynamic updates."""
    __tablename__ = 'prompt_templates'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False, index=True)
    prompt_key = Column(String, nullable=False, index=True) # e.g., 'email_draft', 'voice_intent'
    version = Column(Integer, default=1, nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    author_agent = Column(String, default="Human") # Track who created/updated ('Human', 'ThinkToolCritique')
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('agent_name', 'prompt_key', 'version', name='uq_prompt_template_version'),
        # Ensure only one prompt is active per agent/key at a time using partial index (PostgreSQL specific)
        Index('ix_active_prompt_template', 'agent_name', 'prompt_key', unique=True, postgresql_where=(is_active == True)),
    )

    def __repr__(self):
        return f"<PromptTemplate(id={self.id}, agent='{self.agent_name}', key='{self.prompt_key}', v={self.version}, active={self.is_active})>"

class KVStore(Base):
    """Generic Key-Value store for caching or temporary data."""
    # ### Phase 1 Plan Ref: 3.9 (Add KVStore table)
    __tablename__ = 'kv_store'
    cache_key = Column(Text, primary_key=True)
    cache_value = Column(Text, nullable=False) # Store serialized data (JSON string)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)

    def __repr__(self):
        return f"<KVStore(key='{self.cache_key}', expires_at='{self.expires_at}')>"

# --- End of models.py ---