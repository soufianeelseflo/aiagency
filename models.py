# Filename: models.py
# Description: SQLAlchemy ORM Models for the AI Agency Database (Postgres Focused).
# Version: 4.1 (Added discovered_needs_log to ConversationState)

import json
import uuid as uuid_pkg
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Index,
    UniqueConstraint, func, event, DDL
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID, JSONB # Use JSONB for tags if possible
from sqlalchemy.orm import declarative_base, relationship # Added relationship

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
    company = Column(String, nullable=True, index=True) # Added company
    job_title = Column(String, nullable=True) # Added job title
    country = Column(String, index=True)
    timezone = Column(String, default="America/New_York")
    interests = Column(Text, nullable=True) # Store as JSON string or comma-separated
    last_interaction = Column(DateTime(timezone=True), default=utcnow, index=True)
    last_contacted_at = Column(DateTime(timezone=True), nullable=True, index=True) # Track last outreach attempt
    last_opened_at = Column(DateTime(timezone=True), nullable=True, index=True) # Track last email open
    last_replied_at = Column(DateTime(timezone=True), nullable=True, index=True) # Track last email reply
    engagement_score = Column(Float, default=0.1, index=True, nullable=False) # Lead score (Added nullable=False)
    opt_in = Column(Boolean, default=True, nullable=False, index=True) # GDPR/CASL/STOP tracking (Verified)
    is_deliverable = Column(Boolean, default=True, nullable=False, index=True) # Email deliverability
    source = Column(String, nullable=True, index=True) # e.g., 'Clay.com', 'Manual', 'WebScrape'
    source_reference = Column(String, nullable=True, index=True) # Added source reference (e.g., LinkedIn URL)
    assigned_agent = Column(String, nullable=True) # Primary agent if needed

    # Relationships (Optional but useful)
    email_logs = relationship("EmailLog", back_populates="client")
    call_logs = relationship("CallLog", back_populates="client")
    invoices = relationship("Invoice", back_populates="client")
    knowledge_fragments = relationship("KnowledgeFragment", back_populates="client")
    conversation_states = relationship("ConversationState", back_populates="client") # Added relationship

    def __repr__(self):
        return f"<Client(id={self.id}, name='{self.name}', email='{self.email}', opt_in={self.opt_in})>"

class EmailLog(Base):
    __tablename__ = 'email_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True)
    recipient = Column(String, index=True, nullable=False)
    subject = Column(Text, nullable=True)
    content_preview = Column(Text, nullable=True) # Store first ~250 chars
    status = Column(String, index=True, nullable=False) # 'sent', 'delivered', 'opened', 'responded', 'bounced', 'failed_send', 'blocked_compliance', 'error_internal'
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True, index=True)
    responded_at = Column(DateTime(timezone=True), nullable=True, index=True)
    agent_version = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    sender_account = Column(String, nullable=True, index=True)
    message_id = Column(Text, unique=True, index=True, nullable=True) # SMTP Message-ID
    tracking_pixel_id = Column(UUID(as_uuid=True), unique=True, index=True, nullable=True, default=uuid_pkg.uuid4) # For open tracking

    # Relationship
    client = relationship("Client", back_populates="email_logs")
    composition = relationship("EmailComposition", back_populates="email_log", uselist=False) # One-to-one

    def __repr__(self):
        return f"<EmailLog(id={self.id}, recipient='{self.recipient}', status='{self.status}')>"

class CallLog(Base):
    __tablename__ = 'call_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True)
    call_sid = Column(String, unique=True, index=True, nullable=False) # Twilio SID
    phone_number = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    duration_seconds = Column(Integer, nullable=True)
    transcript = Column(Text, nullable=True) # JSON string of conversation turns
    outcome = Column(String, index=True, nullable=True) # 'success_sale', 'success_followup', 'failed_hangup', 'failed_voicemail', 'failed_error', 'failed_compliance', 'disconnected_client_request', etc.
    recording_url = Column(String, nullable=True)
    final_twilio_status = Column(String, nullable=True) # e.g., 'completed', 'no-answer', 'busy'

    # Relationship
    client = relationship("Client", back_populates="call_logs")

    def __repr__(self):
        return f"<CallLog(id={self.id}, call_sid='{self.call_sid}', outcome='{self.outcome}')>"

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    call_sid = Column(String, primary_key=True) # Use call_sid as PK
    client_id = Column(Integer, ForeignKey('clients.id'), index=True, nullable=True)
    state = Column(String, nullable=False) # Current state in the voice agent FSM
    conversation_log = Column(Text) # Running JSON log of the conversation turns
    discovered_needs_log = Column(Text, nullable=True) # NEW: JSON log of discovered needs
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationship
    client = relationship("Client", back_populates="conversation_states") # Added relationship

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
    invoice_path = Column(String, nullable=True) # Path to stored PDF on VPS
    payment_link = Column(String, nullable=True) # Link to payment processor
    source_reference = Column(String, nullable=True) # e.g., CallLog.call_sid

    # Relationship
    client = relationship("Client", back_populates="invoices")

    def __repr__(self):
        return f"<Invoice(id={self.id}, client_id={self.client_id}, amount={self.amount}, status='{self.status}')>"

class AccountCredentials(Base):
    """Stores credentials for rotated free trial or acquired accounts."""
    __tablename__ = 'account_credentials'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True) # e.g., 'clay.com', 'heygen.com', 'facebook.com', 'twilio.com'
    account_identifier = Column(String, nullable=False, index=True) # e.g., email, username used for login
    api_key = Column(Text, nullable=True) # Encrypted API key
    password = Column(Text, nullable=True) # Encrypted password
    proxy_used = Column(String, nullable=True) # Proxy associated with this account's creation/use
    status = Column(String, default='active', nullable=False, index=True) # 'active', 'limited', 'banned', 'expired'
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used = Column(DateTime(timezone=True), nullable=True, index=True)
    notes = Column(Text, nullable=True) # e.g., Trial expiry date, usage limits

    __table_args__ = (UniqueConstraint('service', 'account_identifier', name='uq_account_cred_service_identifier'),)

    def __repr__(self):
        return f"<AccountCredentials(id={self.id}, service='{self.service}', identifier='{self.account_identifier}', status='{self.status}')>"

class ExpenseLog(Base):
    """Tracks operational expenses."""
    __tablename__ = 'expense_logs'
    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False, index=True) # 'LLM', 'API_Clay', 'API_Twilio', 'API_Deepgram', 'Proxy', 'Resource', 'ConcurrencyAdjustment'
    description = Column(Text, nullable=False) # Encrypted description
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

# --- Knowledge Base & Strategy Models (Managed by ThinkTool via Postgres) ---

class KnowledgeFragment(Base):
    """Stores atomic pieces of information gathered or generated by agents."""
    __tablename__ = 'knowledge_fragments'
    id = Column(Integer, primary_key=True)
    agent_source = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    last_accessed_ts = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True) # Updated on query
    data_type = Column(String, nullable=False, index=True) # e.g., 'email_subject', 'legal_interpretation', 'osint_summary', 'learning_material_summary', 'competitor_strategy', 'market_need', 'client_insight', 'successful_call_script'
    content = Column(Text, nullable=False) # The actual knowledge content (string or JSON string)
    item_hash = Column(String(64), unique=True, index=True, nullable=False) # SHA-256 hash REQUIRED
    relevance_score = Column(Float, default=0.5, nullable=False)
    tags = Column(Text, nullable=True) # Store as JSON string: '["tag1", "tag2"]'
    related_client_id = Column(Integer, ForeignKey('clients.id'), nullable=True, index=True)
    source_reference = Column(String, nullable=True, index=True) # Link back to source (log ID, URL, task ID etc.)

    # Relationship
    client = relationship("Client", back_populates="knowledge_fragments")

    # Add GIN index for tags if using PostgreSQL JSONB functions frequently and storing tags as JSONB
    # __table_args__ = (Index('ix_knowledge_fragments_tags_gin', 'tags', postgresql_using='gin'),)

    def __repr__(self):
        return f"<KnowledgeFragment(id={self.id}, type='{self.data_type}', source='{self.agent_source}', hash='{self.item_hash[:8]}...')>"

class LearnedPattern(Base):
    """Stores correlations, insights, and potential causal links discovered by ThinkTool."""
    __tablename__ = 'learned_patterns'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    pattern_description = Column(Text, nullable=False)
    supporting_fragment_ids = Column(Text, nullable=False) # JSON array string of KnowledgeFragment IDs
    confidence_score = Column(Float, default=0.5, nullable=False, index=True)
    implications = Column(Text, nullable=True) # Strategic value or actionable consequence
    tags = Column(Text, nullable=True) # Store as JSON string: '["tag1", "tag2"]'
    status = Column(String, default='active', index=True) # 'active', 'obsolete', 'under_review'

    # Add GIN index for tags if using PostgreSQL JSONB functions frequently
    # __table_args__ = (Index('ix_learned_patterns_tags_gin', 'tags', postgresql_using='gin'),)

    def __repr__(self):
        return f"<LearnedPattern(id={self.id}, confidence={self.confidence_score:.2f}, status='{self.status}')>"

class StrategicDirective(Base):
    """Stores high-level instructions generated by ThinkTool to guide agents."""
    __tablename__ = 'strategic_directives'
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False) # 'ThinkToolSynthesis', 'ThinkToolCritique', 'Human', 'OptimizationLogic'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    target_agent = Column(String, nullable=False, index=True) # Specific agent name or 'All' or 'Orchestrator'
    directive_type = Column(String, nullable=False, index=True) # e.g., 'test_strategy', 'prioritize_target', 'update_prompt', 'evaluate_tool', 'create_trial_account', 'execute_clay_call'
    content = Column(Text, nullable=False) # Detailed instructions, potentially JSON
    priority = Column(Integer, default=5, nullable=False, index=True) # Lower number = higher priority
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'active', 'completed', 'failed', 'expired', 'cancelled'
    expiry_timestamp = Column(DateTime(timezone=True), nullable=True)
    result_summary = Column(Text, nullable=True) # Outcome of the directive execution

    def __repr__(self):
        return f"<StrategicDirective(id={self.id}, type='{self.directive_type}', target='{self.target_agent}', status='{self.status}')>"

class PromptTemplate(Base):
    """Stores and manages prompts used by agents, enabling dynamic updates."""
    __tablename__ = 'prompt_templates'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False, index=True)
    prompt_key = Column(String, nullable=False, index=True) # e.g., 'email_draft', 'voice_intent', 'think_synthesize'
    version = Column(Integer, default=1, nullable=False)
    content = Column(Text, nullable=False) # The actual prompt template text
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

class EmailStyles(Base):
    """Stores successful email styles for learning."""
    __tablename__ = 'email_styles'
    id = Column(Integer, primary_key=True)
    content_hash = Column(String(64), unique=True, index=True, nullable=False) # SHA256 hash of body
    body_template = Column(Text, nullable=True) # Store the successful HTML body
    subject_template = Column(Text, nullable=True) # Store the successful subject
    performance_score = Column(Float, default=0.5, index=True) # Based on open/reply rates associated with this style
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True)

    def __repr__(self):
        return f"<EmailStyle(id={self.id}, score={self.performance_score:.2f}, hash='{self.content_hash[:8]}...')>"

class EmailComposition(Base):
    """Links a sent email (EmailLog) to the specific knowledge fragments used."""
    __tablename__ = 'email_composition'
    id = Column(Integer, primary_key=True)
    email_log_id = Column(Integer, ForeignKey('email_logs.id'), unique=True, nullable=False, index=True)
    subject_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id'), nullable=True)
    hook_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id'), nullable=True)
    body_snippets_kf_ids = Column(ARRAY(Integer), nullable=True) # Use ARRAY for list of IDs (PostgreSQL specific)
    cta_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id'), nullable=True)
    style_id = Column(Integer, ForeignKey('email_styles.id'), nullable=True) # Link to EmailStyles if applicable
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    email_log = relationship("EmailLog", back_populates="composition")
    # Define relationships to KnowledgeFragment if needed (use foreign_keys for multiple FKs to same table)
    # subject_fragment = relationship("KnowledgeFragment", foreign_keys=[subject_kf_id])
    # style = relationship("EmailStyles")

    def __repr__(self):
        return f"<EmailComposition(id={self.id}, email_log_id={self.email_log_id})>"

class KVStore(Base):
    """Generic Key-Value store for caching or temporary data (e.g., tracking pixel mapping)."""
    __tablename__ = 'kv_store'
    key = Column(Text, primary_key=True) # The key (e.g., tracking_pixel_uuid)
    value = Column(Text, nullable=False) # The value (e.g., email_log_id as string)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True) # Null means no expiry

    def __repr__(self):
        return f"<KVStore(key='{self.key}', expires_at='{self.expires_at}')>"

# --- Optional: Add Trigger for last_accessed_ts on KnowledgeFragment ---
# This ensures last_accessed_ts is updated automatically on SELECT/UPDATE in Postgres
# Note: Requires superuser privileges or trust settings in Postgres to create functions.
# Consider if application-level updates (in query_knowledge_base) are sufficient.

# update_last_accessed_trigger_func = DDL("""
# CREATE OR REPLACE FUNCTION update_last_accessed_ts_func()
# RETURNS TRIGGER AS $
# BEGIN
#     -- Check if the trigger is fired by an UPDATE statement
#     -- Only update last_accessed_ts if other columns are being updated,
#     -- or handle SELECT updates separately in application logic.
#     -- This basic version updates on ANY update.
#     NEW.last_accessed_ts = now() AT TIME ZONE 'utc';
#     RETURN NEW;
# END;
# $ LANGUAGE plpgsql;
# """)

# update_last_accessed_trigger = DDL("""
# DROP TRIGGER IF EXISTS update_kf_last_accessed ON knowledge_fragments; -- Drop existing trigger first
# CREATE TRIGGER update_kf_last_accessed
# BEFORE UPDATE ON knowledge_fragments
# FOR EACH ROW
# EXECUTE FUNCTION update_last_accessed_ts_func();
# """)

# event.listen(KnowledgeFragment.__table__, 'after_create', update_last_accessed_trigger_func.execute_if(dialect='postgresql'))
# event.listen(KnowledgeFragment.__table__, 'after_create', update_last_accessed_trigger.execute_if(dialect='postgresql'))


# --- End of models.py ---