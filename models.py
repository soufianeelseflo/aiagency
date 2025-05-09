# Filename: models.py
# Description: SQLAlchemy ORM Models for the AI Agency Database (Postgres Focused).
# Version: 4.2 (Level 30+ Transmutation - Added last_enriched_at, refined relationships, task_id to KF)

import json
import uuid as uuid_pkg
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Index,
    UniqueConstraint, func, event, DDL
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def utcnow():
    return datetime.now(timezone.utc)

class Client(Base):
    __tablename__ = 'clients'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, index=True, nullable=True, unique=True)
    phone = Column(String, index=True, nullable=True, unique=True)
    company = Column(String, nullable=True, index=True)
    job_title = Column(String, nullable=True)
    country = Column(String, index=True, nullable=True) # Made nullable
    timezone = Column(String, default="America/New_York", nullable=True) # Made nullable
    industry = Column(String, nullable=True, index=True) # Added industry
    interests = Column(Text, nullable=True) # JSON string or comma-separated
    last_interaction = Column(DateTime(timezone=True), default=utcnow, index=True)
    last_contacted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_opened_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_replied_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_enriched_at = Column(DateTime(timezone=True), nullable=True, index=True) # Added for Clay webhook processing (MRE)
    engagement_score = Column(Float, default=0.1, index=True, nullable=False)
    opt_in = Column(Boolean, default=True, nullable=False, index=True)
    is_deliverable = Column(Boolean, default=True, nullable=False, index=True)
    source = Column(String, nullable=True, index=True)
    source_reference = Column(String, nullable=True, index=True) # e.g., LinkedIn URL, Clay correlation_id
    assigned_agent = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False) # Added for PLEA

    email_logs = relationship("EmailLog", back_populates="client", cascade="all, delete-orphan")
    call_logs = relationship("CallLog", back_populates="client", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="client", cascade="all, delete-orphan")
    knowledge_fragments = relationship("KnowledgeFragment", back_populates="client") # No cascade delete here, KF might be general
    conversation_states = relationship("ConversationState", back_populates="client", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(id={self.id}, name='{self.name}', email='{self.email}', opt_in={self.opt_in})>"

class EmailLog(Base):
    __tablename__ = 'email_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='SET NULL'), index=True, nullable=True) # ondelete SET NULL
    recipient = Column(String, index=True, nullable=False)
    subject = Column(Text, nullable=True)
    content_preview = Column(Text, nullable=True)
    status = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True, index=True)
    responded_at = Column(DateTime(timezone=True), nullable=True, index=True)
    agent_version = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    sender_account = Column(String, nullable=True, index=True)
    message_id = Column(Text, unique=True, index=True, nullable=True)
    tracking_pixel_id = Column(UUID(as_uuid=True), unique=True, index=True, nullable=True, default=uuid_pkg.uuid4)
    task_id_source = Column(String, nullable=True, index=True) # PLEA: Link to task ID that generated this

    client = relationship("Client", back_populates="email_logs")
    composition = relationship("EmailComposition", back_populates="email_log", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<EmailLog(id={self.id}, recipient='{self.recipient}', status='{self.status}')>"

class CallLog(Base):
    __tablename__ = 'call_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='SET NULL'), index=True, nullable=True) # ondelete SET NULL
    call_sid = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)
    duration_seconds = Column(Integer, nullable=True)
    transcript = Column(Text, nullable=True) # JSON string
    outcome = Column(String, index=True, nullable=True)
    recording_url = Column(String, nullable=True)
    final_twilio_status = Column(String, nullable=True)
    task_id_source = Column(String, nullable=True, index=True) # PLEA

    client = relationship("Client", back_populates="call_logs")

    def __repr__(self):
        return f"<CallLog(id={self.id}, call_sid='{self.call_sid}', outcome='{self.outcome}')>"

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    call_sid = Column(String, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='SET NULL'), index=True, nullable=True) # ondelete SET NULL
    state = Column(String, nullable=False)
    conversation_log = Column(Text) # JSON
    discovered_needs_log = Column(Text, nullable=True) # JSON
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    client = relationship("Client", back_populates="conversation_states")

    def __repr__(self):
        return f"<ConversationState(call_sid='{self.call_sid}', state='{self.state}')>"

class Invoice(Base):
    __tablename__ = 'invoices'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id', ondelete='RESTRICT'), index=True, nullable=False) # RESTRICT delete if invoices exist
    amount = Column(Float, nullable=False)
    status = Column(String, default='pending', nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    due_date = Column(DateTime(timezone=True), nullable=True)
    invoice_path = Column(String, nullable=True)
    payment_link = Column(String, nullable=True)
    source_reference = Column(String, nullable=True) # e.g., CallLog.call_sid, task_id
    task_id_source = Column(String, nullable=True, index=True) # PLEA

    client = relationship("Client", back_populates="invoices")

    def __repr__(self):
        return f"<Invoice(id={self.id}, client_id={self.client_id}, amount={self.amount}, status='{self.status}')>"

class AccountCredentials(Base): # For MRE
    __tablename__ = 'account_credentials'
    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False, index=True) # e.g., 'clay.com', 'google.com', 'temp_mail_service_x'
    account_identifier = Column(String, nullable=False, index=True) # username/email
    api_key = Column(Text, nullable=True) # Encrypted
    password = Column(Text, nullable=True) # Encrypted
    proxy_used_on_creation = Column(String, nullable=True) # MRE
    status = Column(String, default='unknown', nullable=False, index=True) # 'active', 'limited', 'banned', 'expired', 'needs_review', 'unknown'
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used_ts = Column(DateTime(timezone=True), nullable=True, index=True) # PLEA
    last_successful_use_ts = Column(DateTime(timezone=True), nullable=True, index=True) # PLEA
    last_status_update_ts = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow) # PLEA
    notes = Column(Text, nullable=True) # Trial expiry, usage limits, UI interaction hints
    task_id_source = Column(String, nullable=True, index=True) # PLEA: Task that created this account

    __table_args__ = (UniqueConstraint('service', 'account_identifier', name='uq_account_cred_service_identifier'),)

    def __repr__(self):
        return f"<AccountCredentials(id={self.id}, service='{self.service}', identifier='{self.account_identifier}', status='{self.status}')>"

class ExpenseLog(Base): # For MRE
    __tablename__ = 'expense_logs'
    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    currency = Column(String, default='USD', nullable=False) # Added currency
    category = Column(String, nullable=False, index=True) # 'LLM', 'API_Clay', 'API_Twilio', 'Proxy', 'Resource_Acquisition', 'Manual_Expense'
    description = Column(Text, nullable=False) # Can be encrypted if sensitive
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    agent_source = Column(String, nullable=True, index=True)
    task_id_reference = Column(String, nullable=True, index=True) # PLEA

    def __repr__(self):
        return f"<ExpenseLog(id={self.id}, category='{self.category}', amount={self.amount} {self.currency})>"

class MigrationStatus(Base):
    __tablename__ = 'migration_status'
    migration_name = Column(String, primary_key=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        status = "Completed" if self.completed_at else "Pending"
        return f"<MigrationStatus(name='{self.migration_name}', status='{status}')>"

class KnowledgeFragment(Base): # Central to PLEA
    __tablename__ = 'knowledge_fragments'
    id = Column(Integer, primary_key=True)
    agent_source = Column(String, nullable=False, index=True) # Agent that logged this
    task_id_source = Column(String, nullable=True, index=True) # Task that generated/discovered this
    directive_id_source = Column(Integer, ForeignKey('strategic_directives.id', ondelete='SET NULL'), nullable=True, index=True) # Directive that led to this
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    last_accessed_ts = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    data_type = Column(String, nullable=False, index=True) # e.g., 'email_subject_idea', 'competitor_pricing_model', 'ui_interaction_log', 'successful_temp_mail_script'
    content = Column(Text, nullable=False) # String or JSON string
    content_vector = Column(JSONB, nullable=True) # Placeholder for future embedding storage for semantic search
    item_hash = Column(String(64), unique=True, index=True, nullable=False) # SHA-256 of content
    relevance_score = Column(Float, default=0.5, nullable=False, index=True) # Contextual relevance
    confidence_score = Column(Float, default=1.0, nullable=False, index=True) # Confidence in the data itself
    tags = Column(Text, nullable=True) # JSON string: '["tag1", "tag2"]'
    related_client_id = Column(Integer, ForeignKey('clients.id', ondelete='SET NULL'), nullable=True, index=True)
    source_reference = Column(String, nullable=True, index=True) # URL, Log ID, external system ID
    # For UI automation logs (MRE, PLEA)
    ui_interaction_screenshot_ref = Column(String, nullable=True) # Path/URL to screenshot if applicable
    ui_interaction_success = Column(Boolean, nullable=True) # If this KF logs a UI step outcome

    client = relationship("Client", back_populates="knowledge_fragments")
    directive_source = relationship("StrategicDirective", back_populates="related_knowledge_fragments")

    __table_args__ = (Index('ix_kf_tags_gin', 'tags', postgresql_using='gin', postgresql_ops={'tags': 'jsonb_path_ops'}),) # For JSONB operators

    def __repr__(self):
        return f"<KnowledgeFragment(id={self.id}, type='{self.data_type}', hash='{self.item_hash[:8]}...')>"

class LearnedPattern(Base): # For ARAA, PLEA
    __tablename__ = 'learned_patterns'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    pattern_description = Column(Text, nullable=False)
    supporting_fragment_ids = Column(Text, nullable=False) # JSON array of KF IDs
    confidence_score = Column(Float, default=0.5, nullable=False, index=True)
    implications = Column(Text, nullable=True) # Strategic value or actionable consequence
    pattern_type = Column(String, default="observational", index=True) # e.g., 'observational', 'causal_hypothesis', 'exploit_hypothesis' (MRE, AMAC)
    potential_exploit_details = Column(Text, nullable=True) # Specifics if 'exploit_hypothesis'
    tags = Column(Text, nullable=True) # JSON string
    status = Column(String, default='active', index=True) # 'active', 'obsolete', 'under_review', 'validated_exploit'

    __table_args__ = (Index('ix_lp_tags_gin', 'tags', postgresql_using='gin', postgresql_ops={'tags': 'jsonb_path_ops'}),)

    def __repr__(self):
        return f"<LearnedPattern(id={self.id}, type='{self.pattern_type}', conf={self.confidence_score:.2f})>"

class StrategicDirective(Base): # For ARAA, PCOF
    __tablename__ = 'strategic_directives'
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False) # 'ThinkToolSynthesis', 'HumanOperator', 'SelfCritique'
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    target_agent = Column(String, nullable=False, index=True) # Specific agent or 'All' or 'Orchestrator'
    directive_type = Column(String, nullable=False, index=True) # e.g., 'test_strategy', 'update_prompt_template', 'acquire_resource_gmail', 'execute_clay_enrichment_workflow'
    content = Column(Text, nullable=False) # Detailed instructions, potentially JSON
    priority = Column(Integer, default=5, nullable=False, index=True) # 1=highest
    status = Column(String, default='pending', nullable=False, index=True) # 'pending', 'active', 'completed', 'failed', 'expired', 'cancelled', 'halted_by_reflection'
    expiry_timestamp = Column(DateTime(timezone=True), nullable=True)
    result_summary = Column(Text, nullable=True) # Outcome of execution
    estimated_roi_or_impact = Column(String, nullable=True) # PCOF
    risk_assessment_summary = Column(Text, nullable=True) # ARAA

    related_knowledge_fragments = relationship("KnowledgeFragment", back_populates="directive_source")

    def __repr__(self):
        return f"<StrategicDirective(id={self.id}, type='{self.directive_type}', target='{self.target_agent}', status='{self.status}')>"

class PromptTemplate(Base): # For ARAA, PLEA
    __tablename__ = 'prompt_templates'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False, index=True)
    prompt_key = Column(String, nullable=False, index=True)
    version = Column(Integer, default=1, nullable=False)
    content = Column(Text, nullable=False) # The prompt template itself
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    author_agent = Column(String, default="Human") # 'Human', 'ThinkToolCritique', 'SelfAdapted'
    last_updated = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    performance_score = Column(Float, nullable=True, index=True) # PLEA
    usage_count = Column(Integer, default=0, nullable=False) # PLEA
    notes = Column(Text, nullable=True) # e.g., critique summary leading to this version

    __table_args__ = (
        UniqueConstraint('agent_name', 'prompt_key', 'version', name='uq_prompt_template_version'),
        Index('ix_active_prompt_template', 'agent_name', 'prompt_key', unique=True, postgresql_where=(is_active == True)),
    )

    def __repr__(self):
        return f"<PromptTemplate(id={self.id}, agent='{self.agent_name}', key='{self.prompt_key}', v={self.version}, active={self.is_active})>"

class EmailStyles(Base): # For AMAC, PLEA
    __tablename__ = 'email_styles'
    id = Column(Integer, primary_key=True)
    content_hash = Column(String(64), unique=True, index=True, nullable=False)
    body_template = Column(Text, nullable=True)
    subject_template = Column(Text, nullable=True)
    performance_score = Column(Float, default=0.5, index=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True)
    tags = Column(Text, nullable=True) # JSON string: for style attributes e.g., 'aggressive_cta', 'short_form'

    def __repr__(self):
        return f"<EmailStyle(id={self.id}, score={self.performance_score:.2f}, hash='{self.content_hash[:8]}...')>"

class EmailComposition(Base): # For PLEA
    __tablename__ = 'email_composition'
    id = Column(Integer, primary_key=True)
    email_log_id = Column(Integer, ForeignKey('email_logs.id', ondelete='CASCADE'), unique=True, nullable=False, index=True) # Cascade delete
    subject_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id', ondelete='SET NULL'), nullable=True)
    hook_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id', ondelete='SET NULL'), nullable=True)
    body_snippets_kf_ids = Column(ARRAY(Integer), nullable=True)
    cta_kf_id = Column(Integer, ForeignKey('knowledge_fragments.id', ondelete='SET NULL'), nullable=True)
    style_id = Column(Integer, ForeignKey('email_styles.id', ondelete='SET NULL'), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    email_log = relationship("EmailLog", back_populates="composition")

    def __repr__(self):
        return f"<EmailComposition(id={self.id}, email_log_id={self.email_log_id})>"

class KVStore(Base): # For general purpose storage
    __tablename__ = 'kv_store'
    key = Column(String, primary_key=True) # String for better compatibility
    value = Column(Text, nullable=False) # JSON string if complex value needed
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)

    def __repr__(self):
        return f"<KVStore(key='{self.key}', expires_at='{self.expires_at}')>"

# --- End of models.py ---