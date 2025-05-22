# Filename: models.py
# Description: SQLAlchemy Models for the AI Agency Database.
# Version: 3.2 (Ensured LearnedPattern and all typing imports are present)

import uuid
import enum
import logging # Added for fallback logger
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Union # ENSURE ALL ARE PRESENT

from sqlalchemy import (
    create_engine, Column, String, DateTime, ForeignKey, Text, Boolean,
    Integer, Float, Enum as SAEnum, UniqueConstraint, Index, CheckConstraint, LargeBinary
)
from sqlalchemy.orm import relationship, validates, sessionmaker, declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import UUID, JSONB

# --- Settings and Utilities Import ---
try:
    from utils.database import encrypt_data, decrypt_data, Base as CustomBaseFromUtil
    from config.settings import settings as app_settings
    UTILS_AND_SETTINGS_AVAILABLE = True
except ImportError as e_import_utils:
    # Fallback logger setup
    _models_logger = logging.getLogger(__name__)
    if not _models_logger.hasHandlers(): # Avoid adding handlers multiple times
        _models_logger.setLevel(logging.CRITICAL)
        _ch = logging.StreamHandler()
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _ch.setFormatter(_formatter)
        _models_logger.addHandler(_ch)

    _models_logger.critical(
        f"CRITICAL FAILURE in models.py: Could not import 'utils.database' or 'config.settings'. "
        f"Cause: {e_import_utils}. Database encryption and potentially other model functionalities will be broken. "
        "This indicates a severe problem in project structure or dependencies that needs immediate attention."
    )
    CustomBaseFromUtil = declarative_base() # type: ignore
    def encrypt_data(data: Optional[str]) -> Optional[str]:
        _models_logger.error("Dummy encrypt_data called: Encryption unavailable due to import failure.")
        return data
    def decrypt_data(data: Optional[str]) -> Optional[str]:
        _models_logger.error("Dummy decrypt_data called: Decryption unavailable due to import failure.")
        return data
    class DummyAppSettings:
        DATABASE_ENCRYPTION_KEY: Optional[str] = None
        DATABASE_URL: Optional[str] = None # Added for __main__ block
    app_settings = DummyAppSettings() # type: ignore
    UTILS_AND_SETTINGS_AVAILABLE = False

Base = CustomBaseFromUtil
logger = logging.getLogger(__name__) # General logger for this module

# --- Enum Definitions (Standard Python Enums) ---
class TaskStatus(enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    LEARNING_PHASE = "learning_phase"
    ACTION_PHASE = "action_phase"
    REFLECTION_PHASE = "reflection_phase"
    HALTED_BY_REFLECTION = "halted_by_reflection"
    SKIPPED = "skipped"
    COMPLETED_NO_DATA = "completed_no_data" # Added for ThinkTool Clay processing

class AgentName(enum.Enum):
    ORCHESTRATOR = "Orchestrator"
    THINK_TOOL = "ThinkTool"
    EMAIL_AGENT = "EmailAgent"
    VOICE_SALES_AGENT = "VoiceSalesAgent"
    SOCIAL_MEDIA_MANAGER = "SocialMediaManager"
    LEGAL_AGENT = "LegalAgent"
    GMAIL_CREATOR_AGENT = "GmailCreatorAgent"
    Browse_AGENT = "BrowseAgent"
    VIDEO_CREATION_AGENT = "VideoCreationAgent"
    CLIENT_RESEARCH_AGENT = "ClientResearchAgent"
    PAYMENT_PROCESSOR_AGENT = "PaymentProcessorAgent"
    UNKNOWN = "Unknown"
    PROGRAMMER_AGENT = "ProgrammerAgent" # Added from Orchestrator

class ClientStatus(enum.Enum): # As defined in your models.py v3.0
    LEAD = "lead"
    CONTACTED = "contacted"
    ENGAGED = "engaged"
    QUALIFIED = "qualified"
    PROPOSAL_SENT = "proposal_sent"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    ONBOARDING = "onboarding"
    ACTIVE_SERVICE = "active_service"
    DORMANT = "dormant"
    DO_NOT_CONTACT = "do_not_contact"

class InteractionType(enum.Enum): # As defined in your models.py v3.0
    EMAIL_SENT = "email_sent"
    EMAIL_OPENED = "email_opened"
    EMAIL_CLICKED = "email_clicked"
    EMAIL_REPLIED = "email_replied"
    EMAIL_BOUNCED = "email_bounced"
    EMAIL_UNSUBSCRIBED = "email_unsubscribed"
    CALL_OUTBOUND_INITIATED = "call_outbound_initiated"
    CALL_ANSWERED = "call_answered"
    CALL_COMPLETED = "call_completed"
    CALL_VOICEMAIL_LEFT = "call_voicemail_left"
    CALL_FAILED = "call_failed"
    CALL_SCHEDULED = "call_scheduled"
    SOCIAL_MEDIA_POST = "social_media_post"
    SOCIAL_MEDIA_DM = "social_media_dm"
    WEBSITE_VISIT_TRACKED = "website_visit_tracked"
    FORM_SUBMISSION = "form_submission"
    MEETING_SCHEDULED = "meeting_scheduled"
    MEETING_COMPLETED = "meeting_completed"
    INVOICE_SENT = "invoice_sent"
    PAYMENT_RECEIVED = "payment_received"
    NOTE_ADDED = "note_added"
    TASK_COMPLETED_FOR_CLIENT = "task_completed_for_client"

class PaymentStatus(enum.Enum): # As defined in your models.py v3.0
    PENDING = "pending"
    PAID = "paid"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_PAID = "partially_paid"
    PARTIALLY_REFUNDED = "partially_refunded"
    VOIDED = "voided"
    OVERDUE = "overdue"

class AccountStatus(enum.Enum): # As defined in your models.py v3.0
    ACTIVE = "active"
    NEEDS_REVIEW = "needs_review"
    BANNED = "banned"
    NEEDS_PASSWORD_RESET = "needs_password_reset"
    LOCKED = "locked"
    DISABLED = "disabled"

# --- Model Definitions ---

class Client(Base): # type: ignore
    __tablename__ = "clients"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    company_name = Column(String(255), nullable=False, index=True)
    website = Column(String(512), nullable=True)
    primary_contact_name = Column(String(255), nullable=True)
    primary_contact_email = Column(String(255), nullable=True, index=True)
    primary_contact_phone = Column(String(50), nullable=True)
    industry = Column(String(100), nullable=True)
    company_size = Column(String(50), nullable=True)
    country = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    
    client_status = Column(SAEnum(ClientStatus), default=ClientStatus.LEAD, nullable=False)
    opt_in_status = Column(String(50), default="pending", index=True)
    client_score = Column(Float, default=0.0, nullable=False)
    source = Column(String(100), nullable=True)
    
    service_tier = Column(String(100), nullable=True)
    contract_start_date = Column(DateTime(timezone=True), nullable=True)
    contract_end_date = Column(DateTime(timezone=True), nullable=True)
    last_interaction = Column(DateTime(timezone=True), nullable=True) # Added from VoiceSalesAgent
    last_enriched_at = Column(DateTime(timezone=True), nullable=True) # Added from ThinkTool
    last_contacted_at = Column(DateTime(timezone=True), nullable=True) # Added from EmailAgent
    opt_in = Column(Boolean, default=True) # Added from EmailAgent/VoiceSalesAgent
    is_deliverable = Column(Boolean, default=True) # Added from EmailAgent/VoiceSalesAgent
    email = Column(String(255), nullable=True, index=True) # Added from EmailAgent/VoiceSalesAgent (ensure consistent with primary_contact_email)
    name = Column(String(255), nullable=True) # Added from EmailAgent/VoiceSalesAgent (ensure consistent with primary_contact_name)
    phone = Column(String(50), nullable=True) # Added from VoiceSalesAgent (ensure consistent with primary_contact_phone)
    timezone = Column(String(100), nullable=True) # Added from VoiceSalesAgent
    interests = Column(Text, nullable=True) # Added from EmailAgent/VoiceSalesAgent (JSON string)
    job_title = Column(String(255), nullable=True) # Added from EmailAgent/ThinkTool
    company = Column(String(255), nullable=True) # Added from EmailAgent/ThinkTool (ensure consistent with company_name)
    location = Column(String(255), nullable=True) # Added from ThinkTool
    source_reference = Column(Text, nullable=True) # Added from ThinkTool (e.g. LinkedIn URL)
    engagement_score = Column(Float, default=0.0) # Added from EmailAgent (ensure consistent with client_score)


    json_data = Column(JSONB, nullable=True, comment="Flexible JSON field for additional structured client data.")

    tasks = relationship("Task", back_populates="client", cascade="all, delete-orphan", lazy="selectin")
    interactions = relationship("InteractionLog", back_populates="client", cascade="all, delete-orphan", lazy="selectin")
    invoices = relationship("Invoice", back_populates="client", cascade="all, delete-orphan", lazy="selectin")
    notes = relationship("Note", back_populates="client", cascade="all, delete-orphan", lazy="selectin")
    
    __table_args__ = (
        Index('idx_client_company_email_unique', 'company_name', 'primary_contact_email', unique=True, postgresql_where=(primary_contact_email.isnot(None))), # type: ignore
        Index('idx_client_status_opt_in', 'client_status', 'opt_in_status'),
    )
    def __repr__(self) -> str:
        return f"<Client(id={self.id}, company_name='{self.company_name}', email='{self.primary_contact_email}')>"

class Task(Base): # type: ignore
    __tablename__ = "tasks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    agent_name = Column(SAEnum(AgentName), nullable=False, index=True)
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=True, index=True)
    client = relationship("Client", back_populates="tasks", lazy="joined")

    priority = Column(Integer, default=5, nullable=False)
    payload = Column(JSONB, comment="Task-specific input data.")
    result = Column(JSONB, nullable=True, comment="Task-specific output data.")
    error_message = Column(Text, nullable=True)
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    parent_task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    sub_tasks = relationship("Task", backref="parent_task", remote_side=[id], lazy="selectin")
    
    action_history = Column(JSONB, nullable=True, default=list, comment="For ThinkTool: sequence of actions taken within the task.")
    current_objective = Column(Text, nullable=True)
    
    knowledge_ids_used = Column(JSONB, nullable=True, default=list)
    knowledge_ids_generated = Column(JSONB, nullable=True, default=list)

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, agent='{self.agent_name.value}', status='{self.status.value}')>"

class InteractionLog(Base): # type: ignore
    __tablename__ = "interaction_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False)
    
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=False, index=True)
    client = relationship("Client", back_populates="interactions", lazy="joined")
    
    agent_name = Column(SAEnum(AgentName), nullable=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True)
    
    type = Column(SAEnum(InteractionType), nullable=False, index=True)
    channel = Column(String(50), nullable=True)
    content_summary = Column(Text, nullable=True)
    external_id = Column(String(255), nullable=True, index=True)
    
    json_data = Column(JSONB, nullable=True, comment="Flexible JSON field for extra details like email open location, call duration, etc.")

    def __repr__(self) -> str:
        return f"<InteractionLog(id={self.id}, client_id={self.client_id}, type='{self.type.value}', ts='{self.timestamp}')>"

class AccountCredentials(Base): # type: ignore
    __tablename__ = "account_credentials"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    service = Column(String(100), nullable=False, index=True)
    account_identifier = Column(String(255), nullable=False, index=True)
    
    _encrypted_password = Column("password", String(1024), nullable=True)
    
    email_associated = Column(String(255), nullable=True, index=True)
    status = Column(SAEnum(AccountStatus), default=AccountStatus.ACTIVE, nullable=False)
    last_used = Column(DateTime(timezone=True), nullable=True)
    proxy_used = Column(String(255), nullable=True)
    creation_agent = Column(SAEnum(AgentName), nullable=True)
    
    notes = Column(Text, nullable=True)
    last_status_update_ts = Column(DateTime(timezone=True), nullable=True)

    @hybrid_property
    def password(self) -> Optional[str]:
        if not UTILS_AND_SETTINGS_AVAILABLE or not app_settings.DATABASE_ENCRYPTION_KEY:
            logger.error(
                f"Attempted to decrypt password for account '{self.account_identifier}' but encryption utils/key are unavailable. "
                "This is a CRITICAL configuration issue."
            )
            return "[PASSWORD_DECRYPTION_UNAVAILABLE]"
        return decrypt_data(self._encrypted_password) if self._encrypted_password else None

    @password.setter
    def password(self, value: Optional[str]) -> None:
        if not UTILS_AND_SETTINGS_AVAILABLE or not app_settings.DATABASE_ENCRYPTION_KEY:
            logger.critical(
                f"Attempted to encrypt password for account '{self.account_identifier}' but encryption utils/key are unavailable. "
                "Password NOT set. This is a CRITICAL configuration issue."
            )
            raise EnvironmentError(
                "DATABASE_ENCRYPTION_KEY not configured or utils unavailable. Cannot encrypt password."
            )
        if value is None:
            self._encrypted_password = None
        else:
            self._encrypted_password = encrypt_data(value)
            
    __table_args__ = (UniqueConstraint('service', 'account_identifier', name='uq_account_service_identifier'),)
    def __repr__(self) -> str:
        return f"<AccountCredentials(id={self.id}, service='{self.service}', identifier='{self.account_identifier}', status='{self.status.value}')>"

class StrategicDirective(Base): # type: ignore
    __tablename__ = "strategic_directives"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    source = Column(String(100), nullable=False)
    target_agent = Column(SAEnum(AgentName), nullable=False, index=True)
    directive_type = Column(String(100), nullable=False, index=True)
    
    content = Column(JSONB, nullable=False, comment="Detailed parameters and context for the directive.")
    priority = Column(Integer, default=5, nullable=False, index=True)
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    
    result_summary = Column(Text, nullable=True)
    related_task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<StrategicDirective(id={self.id}, type='{self.directive_type}', target='{self.target_agent.value}', status='{self.status.value}')>"

class KnowledgeFragment(Base): # type: ignore
    __tablename__ = "knowledge_fragments"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_accessed_ts = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    agent_source = Column(SAEnum(AgentName), nullable=False)
    data_type = Column(String(100), index=True, nullable=False)
    content = Column(Text, nullable=False)
    item_hash = Column(String(64), nullable=False, unique=True, index=True)
    
    relevance_score = Column(Float, default=0.5, nullable=False)
    confidence_score = Column(Float, default=0.5, nullable=False)
    
    tags = Column(JSONB, nullable=True, default=list)
    related_client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True)
    source_reference = Column(Text, nullable=True)
    related_directive_id = Column(UUID(as_uuid=True), ForeignKey("strategic_directives.id", ondelete="SET NULL"), nullable=True)

    def __repr__(self) -> str:
        return f"<KnowledgeFragment(id={self.id}, type='{self.data_type}', source='{self.agent_source.value}')>"

# >>> THIS IS THE CLASS THAT WAS MISSING <<<
class LearnedPattern(Base): # type: ignore
    __tablename__ = "learned_patterns"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Renamed timestamp to created_at for consistency with other new models
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    pattern_description = Column(Text, nullable=False)
    # Added pattern_type from ThinkTool v5.9
    pattern_type = Column(String(100), nullable=True, index=True, comment="e.g., observational, causal, exploit_hypothesis, successful_tactic")
    
    supporting_fragment_ids = Column(JSONB, nullable=True, comment="List of KnowledgeFragment IDs supporting this pattern")
    confidence_score = Column(Float, default=0.0, nullable=False, index=True)
    implications = Column(Text, nullable=True, comment="What this pattern implies or enables")
    
    tags = Column(JSONB, nullable=True, default=list) # List of strings for searchable tags
    status = Column(String(50), default="active", nullable=False, index=True) # e.g., active, deprecated, under_review
    
    # Added fields from ThinkTool v5.9 usage
    usage_count = Column(Integer, default=0, nullable=False)
    last_applied_at = Column(DateTime(timezone=True), nullable=True)
    potential_exploit_details = Column(Text, nullable=True, comment="Specifics if this pattern relates to an exploit")
    related_service_or_platform = Column(String(100), nullable=True, index=True)


    def __repr__(self) -> str:
        return f"<LearnedPattern(id={self.id}, type='{self.pattern_type}', confidence={self.confidence_score:.2f})>"
# >>> END OF LearnedPattern CLASS DEFINITION <<<

class EmailStyles(Base): # type: ignore
    __tablename__ = "email_styles"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    style_name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    # Example fields - can be expanded based on how EmailAgent uses it
    # For instance, if it stores actual template snippets:
    subject_template_snippet = Column(Text, nullable=True)
    body_template_snippet_html = Column(Text, nullable=True) 
    
    tags = Column(JSONB, nullable=True, default=list, comment="Tags for categorization, e.g., 'formal', 'follow-up', 'cold_outreach'")
    effectiveness_score = Column(Float, default=0.0, comment="Learned effectiveness of this style")
    usage_count = Column(Integer, default=0)
    
    author_agent = Column(SAEnum(AgentName), nullable=True, comment="Agent that might have suggested or refined this style")
    is_active = Column(Boolean, default=True, index=True)

    def __repr__(self) -> str:
        return f"<EmailStyles(id={self.id}, style_name='{self.style_name}')>"

class KVStore(Base): # type: ignore
    __tablename__ = "kv_store"
    # Using a composite primary key for simplicity if keys are unique per category
    # Or use a UUID id if preferred and make 'key_name' unique perhaps with a scope
    
    key_name = Column(String(255), primary_key=True, index=True)
    # Optional: Add a category or scope if keys are not globally unique
    # category = Column(String(100), primary_key=True, default="default", index=True) 
    
    value_text = Column(Text, nullable=True)
    value_json = Column(JSONB, nullable=True)
    value_blob = Column(LargeBinary, nullable=True) # For binary data if needed

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    notes = Column(Text, nullable=True)

    # If you add 'category':
    # __table_args__ = (UniqueConstraint('category', 'key_name', name='uq_kv_category_key'),)

    def __repr__(self) -> str:
        # If you add 'category':
        # return f"<KVStore(category='{self.category}', key='{self.key_name}')>"
        return f"<KVStore(key='{self.key_name}')>"

class Invoice(Base): # type: ignore
    __tablename__ = "invoices"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True)
    client = relationship("Client", back_populates="invoices", lazy="joined")
    
    invoice_number = Column(String(50), unique=True, nullable=False, index=True)
    due_date = Column(DateTime(timezone=True), nullable=False)
    
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    status = Column(SAEnum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False, index=True)
    
    line_items = Column(JSONB, nullable=False)
    notes = Column(Text, nullable=True)
    
    paid_at = Column(DateTime(timezone=True), nullable=True)
    payment_method = Column(String(50), nullable=True)
    
    financial_transactions = relationship("FinancialTransaction", back_populates="invoice", cascade="all, delete-orphan", lazy="selectin")

    __table_args__ = (
        CheckConstraint('amount >= 0', name='chk_invoice_amount_positive'),
        Index('idx_invoice_status_due_date', 'status', 'due_date'),
    )
    def __repr__(self) -> str:
        return f"<Invoice(id={self.id}, number='{self.invoice_number}', status='{self.status.value}', total={self.amount})>"

class FinancialTransaction(Base): # type: ignore
    __tablename__ = "financial_transactions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False)
    
    type = Column(String(50), nullable=False, index=True)
    status = Column(SAEnum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False, index=True)
    
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    
    description = Column(Text, nullable=True)
    
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id", ondelete="SET NULL"), nullable=True, index=True)
    invoice = relationship("Invoice", back_populates="financial_transactions", lazy="joined")
    
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True)
    
    external_transaction_id = Column(String(255), nullable=True, index=True)
    payment_gateway = Column(String(50), nullable=True)
    
    json_data = Column(JSONB, nullable=True, comment="Gateway response, fee details, etc.")
    def __repr__(self) -> str:
        return f"<FinancialTransaction(id={self.id}, type='{self.type}', status='{self.status.value}', amount={self.amount})>"

class Note(Base): # type: ignore
    __tablename__ = "notes"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=True, index=True)
    client = relationship("Client", back_populates="notes", lazy="joined")

    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True)

    author_agent_name = Column(SAEnum(AgentName), nullable=True)
    author_user_identifier = Column(String(255), nullable=True, comment="Identifier for a human user if UI allows manual notes")
    
    content = Column(Text, nullable=False)
    note_type = Column(String(50), default="general", nullable=False)
    is_pinned = Column(Boolean, default=False, nullable=False)
    def __repr__(self) -> str:
        return f"<Note(id={self.id}, type='{self.note_type}', client_id={self.client_id}, content='{self.content[:30]}...')>"

class EmailLog(Base): # type: ignore
    __tablename__ = "email_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey('clients.id', ondelete="SET NULL"), nullable=True, index=True)
    recipient = Column(String(255), nullable=False, index=True)
    subject = Column(String(512), nullable=False)
    content_preview = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    agent_version = Column(String(50), nullable=True)
    sender_account = Column(String(255), nullable=True)
    message_id = Column(String(255), nullable=True, unique=True, index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True)
    responded_at = Column(DateTime(timezone=True), nullable=True)
    json_data = Column(JSONB, nullable=True, comment="Extra data like composition IDs, campaign ID, etc.")
    compositions = relationship("EmailComposition", back_populates="email_log", cascade="all, delete-orphan", lazy="selectin")

class CallLog(Base): # type: ignore
    __tablename__ = "call_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey('clients.id', ondelete="SET NULL"), nullable=True, index=True)
    call_sid = Column(String(255), unique=True, nullable=False, index=True)
    phone_number = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    transcript = Column(Text, nullable=True)
    outcome = Column(String(100), nullable=True, index=True)
    duration_seconds = Column(Integer, nullable=True)
    recording_url = Column(String(1024), nullable=True)
    final_twilio_status = Column(String(50), nullable=True)
    json_data = Column(JSONB, nullable=True, comment="Additional call metadata")

class ConversationState(Base): # type: ignore
    __tablename__ = "conversation_states"
    call_sid = Column(String(255), primary_key=True)
    state = Column(String(100), nullable=False)
    conversation_log = Column(Text, nullable=False)
    discovered_needs_log = Column(Text, nullable=True)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

class PromptTemplate(Base): # type: ignore
    __tablename__ = "prompt_templates"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_name = Column(SAEnum(AgentName), nullable=False, index=True)
    prompt_key = Column(String(100), nullable=False, index=True)
    version = Column(Integer, default=1, nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    author_agent = Column(SAEnum(AgentName), nullable=True)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    notes = Column(Text, nullable=True)
    __table_args__ = (UniqueConstraint('agent_name', 'prompt_key', 'version', name='uq_prompt_agent_key_version'),)

class ExpenseLog(Base): # type: ignore
    __tablename__ = "expense_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    agent_name = Column(SAEnum(AgentName), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    category = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    task_id_reference = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    json_data = Column(JSONB, nullable=True, comment="Additional details like token counts, API call specifics")

class EmailComposition(Base): # type: ignore
    __tablename__ = "email_compositions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    email_log_id = Column(UUID(as_uuid=True), ForeignKey('email_logs.id', ondelete="SET NULL"), nullable=True, index=True)
    email_log = relationship("EmailLog", back_populates="compositions", lazy="selectin") # Assuming EmailLog will have a 'compositions' relationship

    subject_template_id = Column(UUID(as_uuid=True), ForeignKey('prompt_templates.id', ondelete="SET NULL"), nullable=True)
    body_template_id = Column(UUID(as_uuid=True), ForeignKey('prompt_templates.id', ondelete="SET NULL"), nullable=True)
    
    # Store IDs of knowledge fragments used
    knowledge_fragment_ids = Column(JSONB, nullable=True, default=list)
    
    # Store IDs or names of email styles used
    email_style_ids_or_names = Column(JSONB, nullable=True, default=list)
    
    # Store any specific enriched data points that were key to this composition
    key_enriched_data_points = Column(JSONB, nullable=True, default=dict)
    
    llm_model_used = Column(String(255), nullable=True)
    generation_parameters = Column(JSONB, nullable=True, comment="e.g., temperature, max_tokens used for generation")
    
    notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<EmailComposition(id={self.id}, email_log_id={self.email_log_id})>"

# --- Optional: Function to create tables (for dev/testing, Alembic for prod) ---
def create_all_tables_dev(db_url_sync: str) -> None:
    logger_models_dev = logging.getLogger(__name__ + "_dev_create")
    if not db_url_sync:
        logger_models_dev.error("Synchronous Database URL not provided. Cannot create tables.")
        return
    try:
        engine = create_engine(db_url_sync)
        Base.metadata.create_all(engine) # type: ignore
        logger_models_dev.info(f"All tables created successfully (if they didn't exist) on {db_url_sync}.")
    except Exception as e_create:
        logger_models_dev.error(f"Error creating tables on {db_url_sync}: {e_create}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_logger = logging.getLogger(__name__ + "_main")

    if not UTILS_AND_SETTINGS_AVAILABLE:
        main_logger.critical(
            "Cannot proceed with __main__ block in models.py because critical utilities "
            "(utils.database or config.settings) failed to import earlier. "
        )
    elif app_settings.DATABASE_URL: # Check if DATABASE_URL is not None
        sync_db_url = str(app_settings.DATABASE_URL).replace("postgresql+asyncpg", "postgresql")
        sync_db_url = sync_db_url.replace("postgresql+psycopg", "postgresql")
        
        main_logger.info(f"Attempting to create tables for database (using sync URL): {sync_db_url}")
        create_all_tables_dev(sync_db_url)
        main_logger.info(
            "Table creation process finished. Review logs for success or errors. "
            "IMPORTANT: For production, use Alembic for database migrations."
        )
    else:
        main_logger.error(
            "DATABASE_URL not set in environment or settings. Cannot create tables."
        )

# --- End of models.py ---