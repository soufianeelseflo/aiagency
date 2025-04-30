# Filename: models.py
# Description: SQLAlchemy ORM Models for the AI Agency Database.
# Version: 2.1 (Production Ready - Enhanced Schema, Indexes, Constraints)

import json
import uuid as uuid_pkg # Alias to avoid conflict with Column name
from datetime import datetime, timezone, timedelta # Use timezone directly
import pytz # Keep pytz for timezone object creation where needed

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Index,
    UniqueConstraint, func, event, DDL # Import func, event, DDL
)
# Import UUID type specifically for PostgreSQL
from sqlalchemy.dialects.postgresql import ARRAY, UUID
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
    last_contacted_at = Column(DateTime(timezone=True), nullable=True, index=True) # Added
    last_opened_at = Column(DateTime(timezone=True), nullable=True, index=True) # Added
    last_replied_at = Column(DateTime(timezone=True), nullable=True, index=True) # Added
    response_rate = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.1, index=True)
    opt_in = Column(Boolean, default=True, nullable=False)
    is_deliverable = Column(Boolean, default=True, nullable=False, index=True)
    source = Column(String, nullable=True, index=True)
    assigned_agent = Column(String, nullable=True)

    # Relationships (Optional, define if needed for ORM queries)
    # email_logs = relationship("EmailLog", back_populates="client")
    # call_logs = relationship("CallLog", back_populates="client")
    # invoices = relationship("Invoice", back_populates="client")

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
    opened_at = Column(DateTime(timezone=True), nullable=True, index=True) # Added index
    responded_at = Column(DateTime(timezone=True), nullable=True, index=True) # Added index
    spam_flagged = Column(Boolean, default=False)
    agent_version = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    sender_account = Column(String, nullable=True, index=True) # Added index
    message_id = Column(Text, unique=True, index=True, nullable=True) # Added Text, unique, index
    tracking_pixel_id = Column(UUID(as_uuid=True), unique=True, index=True, nullable=True, default=uuid_pkg.uuid4) # Added UUID

    # client = relationship("Client", back_populates="email_logs") # Optional relationship

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
    transcript = Column(Text, nullable=True) # Changed from 'conversation' for clarity
    outcome = Column(Boolean, nullable=True, index=True) # Changed to Boolean for success/fail
    outcome_details = Column(Text, nullable=True) # Added for context
    recording_url = Column(String, nullable=True)
    final_twilio_status = Column(String, nullable=True) # Added

    # client = relationship("Client", back_populates="call_logs") # Optional relationship

    def __repr__(self):
        outcome_str = 'Success' if self.outcome else ('Failure' if self.outcome is False else 'Unknown')
        return f"<CallLog(id={self.id}, call_sid='{self.call_sid}', outcome='{outcome_str}')>"

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
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=False) # Ensure non-nullable
    amount = Column(Float, nullable=False)
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'paid', 'failed', 'cancelled'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    due_date = Column(DateTime(timezone=True), nullable=True)
    invoice_path = Column(String, nullable=True)
    payment_link = Column(String, nullable=True)

    # client = relationship("Client", back_populates="invoices") # Optional relationship

    def __repr__(self):
        return f"<Invoice(id={self.id}, client_id={self.client_id}, amount={self.amount}, status='{self.status}')>"

class Account(Base):
    """Stores credentials (via Vault path) and status for external service accounts."""
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True)
    email = Column(String, index=True, nullable=True)
    username = Column(String, index=True, nullable=True)
    vault_path = Column(String, unique=True, nullable=False) # REQUIRED, unique path in Vault
    phone = Column(String, nullable=True) # Associated phone if any (actual number in Vault)
    # cookies = Column(Text, nullable=True) # DEPRECATED - Store cookies in Vault via vault_path
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used = Column(DateTime(timezone=True), nullable=True, index=True)
    is_available = Column(Boolean, default=True, nullable=False, index=True)
    is_recurring = Column(Boolean, default=False)
    credit_status = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    last_credit_check_ts = Column(DateTime(timezone=True), nullable=True) # Added
    proxy_assigned = Column(String, nullable=True) # Added
    user_agent_override = Column(String, nullable=True) # Added

    __table_args__ = (UniqueConstraint('service', 'email', name='uq_account_service_email'),)

    def __repr__(self):
        return f"<Account(id={self.id}, service='{self.service}', email='{self.email}', available={self.is_available})>"

class OSINTData(Base):
    """Stores results from OSINT gathering activities."""
    __tablename__ = 'osint_data'
    id = Column(Integer, primary_key=True)
    target = Column(String, nullable=False, index=True)
    tools_used = Column(Text, nullable=True) # JSON array string
    raw_data = Column(Text, nullable=False) # Could be large JSON
    analysis_results = Column(Text, nullable=True) # JSON dump from analysis
    relevance = Column(String, nullable=True, index=True) # 'high', 'medium', 'low', 'pending_analysis', 'error'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<OSINTData(id={self.id}, target='{self.target}', relevance='{self.relevance}')>"

class ExpenseLog(Base):
    """Tracks expenses incurred by the agency."""
    __tablename__ = 'expense_logs'
    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False, index=True) # 'API', 'Proxy', 'Tool', 'Infrastructure', 'Concurrency'
    description = Column(Text, nullable=False) # Encrypted description
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    agent_source = Column(String, nullable=True, index=True) # Added index

    def __repr__(self):
        return f"<ExpenseLog(id={self.id}, category='{self.category}', amount={self.amount})>"

class ServiceCredit(Base):
    """Tracks available credits for external services."""
    __tablename__ = 'service_credits'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=True, index=True)
    credits_value = Column(String, nullable=False) # e.g., '1000', '5.23', 'unlimited'
    credits_unit = Column(String, nullable=True) # e.g., 'tokens', 'USD', 'calls', 'minutes'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    # account = relationship("Account") # Optional relationship

    def __repr__(self):
        return f"<ServiceCredit(id={self.id}, service='{self.service}', credits='{self.credits_value}')>"

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
    last_accessed_ts = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False, index=True)
    data_type = Column(String, nullable=False, index=True) # e.g., 'email_subject', 'legal_interpretation', 'osint_summary'
    content = Column(Text, nullable=False) # The actual knowledge content (string or JSON string)
    item_hash = Column(String(64), unique=True, index=True, nullable=False) # SHA-256 hash REQUIRED for deduplication
    relevance_score = Column(Float, default=0.5, nullable=False)
    tags = Column(Text, nullable=True) # Store as JSON string, consider GIN index if using JSONB operators heavily
    related_client_id = Column(Integer, ForeignKey('clients.id'), nullable=True, index=True)
    source_reference = Column(String, nullable=True, index=True) # Link back to source (log ID, URL, task ID etc.)

    # Add GIN index for tags if using PostgreSQL JSONB functions frequently
    # __table_args__ = (Index('ix_knowledge_fragments_tags_gin', 'tags', postgresql_using='gin'),)

    def __repr__(self):
        return f"<KnowledgeFragment(id={self.id}, type='{self.data_type}', source='{self.agent_source}', hash='{self.item_hash[:8]}...')>"

class KnowledgePerformance(Base):
    """Links knowledge fragments to performance metrics."""
    __tablename__ = 'knowledge_performance'
    id = Column(Integer, primary_key=True)
    knowledge_fragment_id = Column(Integer, ForeignKey('knowledge_fragments.id'), index=True, nullable=False)
    metric_type = Column(String, index=True, nullable=False) # e.g., 'open_rate', 'click_rate', 'conversion_rate'
    metric_value = Column(Float, nullable=False)
    context = Column(Text, nullable=True) # JSON string for context (campaign_id, sample_size, etc.)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    # knowledge_fragment = relationship("KnowledgeFragment") # Optional relationship

    def __repr__(self):
        return f"<KnowledgePerformance(id={self.id}, fragment_id={self.knowledge_fragment_id}, metric='{self.metric_type}', value={self.metric_value})>"

class EmailComposition(Base):
    """Links a sent email (EmailLog) to the specific knowledge fragments used."""
    __tablename__ = 'email_composition'
    id = Column(Integer, primary_key=True)
    email_log_id = Column(Integer, ForeignKey('email_logs.id'), unique=True, nullable=False, index=True)
    subject_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id'), nullable=True, index=True) # Added index
    hook_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id'), nullable=True, index=True) # Added index
    body_snippets_kf_ids = Column(ARRAY(Integer), nullable=True) # PostgreSQL specific ARRAY type
    cta_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id'), nullable=True, index=True) # Added index
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Add GIN index for body_snippets_kf_ids if querying array contents frequently
    # __table_args__ = (Index('ix_email_composition_body_snippets_gin', 'body_snippets_kf_ids', postgresql_using='gin'),)

    def __repr__(self):
        return f"<EmailComposition(id={self.id}, email_log_id={self.email_log_id})>"

class StrategicDirective(Base):
    """Stores high-level instructions generated by ThinkTool to guide agents."""
    __tablename__ = 'strategic_directives'
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False) # 'ThinkToolSynthesis', 'ThinkToolCritique', 'Human', 'OptimizationAgent'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True) # Added index
    target_agent = Column(String, nullable=False, index=True)
    directive_type = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False) # Detailed instructions, potentially JSON
    priority = Column(Integer, default=5, nullable=False, index=True)
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'active', 'completed', 'failed', 'expired', 'cancelled'
    expiry_timestamp = Column(DateTime(timezone=True), nullable=True)
    result_summary = Column(Text, nullable=True)

    def __repr__(self):
        return f"<StrategicDirective(id={self.id}, type='{self.directive_type}', target='{self.target_agent}', status='{self.status}')>"

class LearnedPattern(Base):
    """Stores correlations, insights, and potential causal links discovered by ThinkTool."""
    __tablename__ = 'learned_patterns'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True) # Added index
    pattern_description = Column(Text, nullable=False)
    supporting_fragment_ids = Column(Text, nullable=False) # JSON array string of KnowledgeFragment IDs
    confidence_score = Column(Float, default=0.5, nullable=False, index=True)
    implications = Column(Text, nullable=True)
    tags = Column(Text, nullable=True) # Store as JSON string, consider GIN index
    status = Column(String, default='active', index=True) # 'active', 'obsolete', 'under_review'

    # Add GIN index for tags if using PostgreSQL JSONB functions frequently
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
    is_active = Column(Boolean, default=True, nullable=False) # Removed index here, handled by partial index below
    author_agent = Column(String, default="Human")
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('agent_name', 'prompt_key', 'version', name='uq_prompt_template_version'),
        # Ensure only one prompt is active per agent/key at a time using partial index (PostgreSQL specific)
        Index('ix_active_prompt_template', 'agent_name', 'prompt_key', unique=True, postgresql_where=(is_active == True)),
    )

    def __repr__(self):
        return f"<PromptTemplate(id={self.id}, agent='{self.agent_name}', key='{self.prompt_key}', v={self.version}, active={self.is_active})>"

# --- Optional Monitoring/Workflow Models ---

class AgentStatusLog(Base):
    """(Optional) Logs agent status changes over time for monitoring."""
    __tablename__ = 'agent_status_log'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    old_status = Column(String, nullable=True)
    new_status = Column(String, nullable=False, index=True)
    details = Column(Text, nullable=True) # Optional context (e.g., task ID, error message)

    def __repr__(self):
        return f"<AgentStatusLog(id={self.id}, agent='{self.agent_name}', status='{self.new_status}')>"

class WorkflowState(Base):
    """(Optional) Persists state for long-running, multi-step workflows."""
    __tablename__ = 'workflow_state'
    workflow_id = Column(String, primary_key=True) # Unique ID for the workflow instance
    workflow_type = Column(String, nullable=False, index=True) # e.g., 'UGC_Generation', 'ClientOnboarding'
    current_step = Column(String, nullable=False, index=True) # Identifier for the current step
    state_data = Column(Text, nullable=False) # JSON string containing all necessary state
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, index=True)
    status = Column(String, default='running', nullable=False, index=True) # 'running', 'completed', 'failed', 'paused'

    def __repr__(self):
        return f"<WorkflowState(id='{self.workflow_id}', type='{self.workflow_type}', step='{self.current_step}', status='{self.status}')>"

# --- Utility Models ---

class EmailStyles(Base):
    """Stores successful email styles for learning."""
    __tablename__ = 'email_styles'
    id = Column(Integer, primary_key=True)
    content_hash = Column(String(64), unique=True, index=True, nullable=False) # SHA256 hash of body
    body_template = Column(Text, nullable=True)
    subject_template = Column(Text, nullable=True)
    performance_score = Column(Float, default=0.5, index=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    def __repr__(self):
        return f"<EmailStyle(id={self.id}, score={self.performance_score:.2f}, hash='{self.content_hash[:8]}...')>"

class EmailCache(Base):
    """Postgres-based cache table."""
    __tablename__ = 'email_cache'
    cache_key = Column(Text, primary_key=True)
    cache_value = Column(Text, nullable=False) # Store serialized data (JSON string)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)

    def __repr__(self):
        return f"<EmailCache(key='{self.cache_key}', expires_at='{self.expires_at}')>"

# --- End of models.py ---