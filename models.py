# Filename: models.py
# Description: SQLAlchemy Models for the AI Agency Database.
# Version: 3.1 (IGNIS Final Transmutation - Added LearnedPattern, fixed Optional import)

import uuid
import enum
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Union # Ensure Optional and others are here

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
    import logging
    logging.basicConfig(level=logging.ERROR)
    logger_models = logging.getLogger(__name__)
    logger_models.critical(
        f"CRITICAL FAILURE in models.py: Could not import 'utils.database' or 'config.settings'. "
        f"Cause: {e_import_utils}. Database encryption and potentially other model functionalities will be broken. "
        "This indicates a severe problem in project structure or dependencies that needs immediate attention."
    )
    CustomBaseFromUtil = declarative_base() # type: ignore
    def encrypt_data(data: Optional[str]) -> Optional[str]:
        logger_models.error("Dummy encrypt_data called: Encryption unavailable due to import failure.")
        return data
    def decrypt_data(data: Optional[str]) -> Optional[str]:
        logger_models.error("Dummy decrypt_data called: Decryption unavailable due to import failure.")
        return data
    class DummyAppSettings:
        DATABASE_ENCRYPTION_KEY: Optional[str] = None
    app_settings = DummyAppSettings() # type: ignore
    UTILS_AND_SETTINGS_AVAILABLE = False

Base = CustomBaseFromUtil

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

class AgentName(enum.Enum):
    ORCHESTRATOR = "Orchestrator"
    THINK_TOOL = "ThinkTool"
    EMAIL_AGENT = "EmailAgent"
    VOICE_SALES_AGENT = "VoiceSalesAgent"
    SOCIAL_MEDIA_MANAGER = "SocialMediaManager"
    LEGAL_AGENT = "LegalAgent"
    GMAIL_CREATOR_AGENT = "GmailCreatorAgent"
    Browse_AGENT = "BrowseAgent" # Note: Your Orchestrator imports BrowseAgent as Browse_agent
    VIDEO_CREATION_AGENT = "VideoCreationAgent"
    CLIENT_RESEARCH_AGENT = "ClientResearchAgent"
    PAYMENT_PROCESSOR_AGENT = "PaymentProcessorAgent"
    UNKNOWN = "Unknown"

class ClientStatus(enum.Enum):
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

class InteractionType(enum.Enum):
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

class PaymentStatus(enum.Enum):
    PENDING = "pending"
    PAID = "paid"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_PAID = "partially_paid"
    PARTIALLY_REFUNDED = "partially_refunded"
    VOIDED = "voided"
    OVERDUE = "overdue"

class AccountStatus(enum.Enum):
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

    company_name = Column(String(255), nullable=False, index=True) # Changed from name to company_name for clarity
    website = Column(String(512), nullable=True)
    primary_contact_name = Column(String(255), nullable=True) # Added for primary contact
    primary_contact_email = Column(String(255), nullable=True, index=True) # Added for primary contact
    primary_contact_phone = Column(String(50), nullable=True) # Added for primary contact
    industry = Column(String(100), nullable=True)
    company_size = Column(String(50), nullable=True) # e.g., "1-10", "11-50"
    country = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    
    client_status = Column(SAEnum(ClientStatus), default=ClientStatus.LEAD, nullable=False) # Using new enum
    opt_in_status = Column(String(50), default="pending", index=True) # e.g., pending, opted_in_email, opted_in_call, opted_out_all
    client_score = Column(Float, default=0.0, nullable=False) # Renamed from engagement_score for clarity
    source = Column(String(100), nullable=True) # e.g., clay_import, manual_entry, web_form
    
    service_tier = Column(String(100), nullable=True) # e.g., "Standard UGC", "Premium Voice"
    contract_start_date = Column(DateTime(timezone=True), nullable=True)
    contract_end_date = Column(DateTime(timezone=True), nullable=True)

    json_data = Column(JSONB, nullable=True, comment="Flexible JSON field for additional structured client data like enriched_data from Clay, custom fields, etc.")

    # Relationships
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

    priority = Column(Integer, default=5, nullable=False) # Default priority
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
    current_objective = Column(Text, nullable=True) # For ThinkTool's multi-step reasoning
    
    # For linking KFs to tasks
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
    
    agent_name = Column(SAEnum(AgentName), nullable=True) # Which agent performed/logged
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True)
    
    type = Column(SAEnum(InteractionType), nullable=False, index=True)
    channel = Column(String(50), nullable=True) # e.g., email, call, linkedin, x.com
    content_summary = Column(Text, nullable=True) # e.g., email subject, call outcome summary
    external_id = Column(String(255), nullable=True, index=True) # e.g., email Message-ID, Call SID
    
    json_data = Column(JSONB, nullable=True, comment="Flexible JSON field for extra details like email open location, call duration, etc.")

    def __repr__(self) -> str:
        return f"<InteractionLog(id={self.id}, client_id={self.client_id}, type='{self.type.value}', ts='{self.timestamp}')>"


class AccountCredentials(Base): # type: ignore
    __tablename__ = "account_credentials"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    service = Column(String(100), nullable=False, index=True) # Renamed from service_name for consistency
    account_identifier = Column(String(255), nullable=False, index=True) # Renamed from username
    
    _encrypted_password = Column("password", String(1024), nullable=True) # Keep column name 'password' for simplicity if preferred
    
    email_associated = Column(String(255), nullable=True, index=True)
    status = Column(SAEnum(AccountStatus), default=AccountStatus.ACTIVE, nullable=False)
    last_used = Column(DateTime(timezone=True), nullable=True) # Renamed from last_used_at
    proxy_used = Column(String(255), nullable=True) # Simplified proxy storage
    creation_agent = Column(SAEnum(AgentName), nullable=True) # Renamed
    
    notes = Column(Text, nullable=True) # Changed from JSONB to Text for general notes
    last_status_update_ts = Column(DateTime(timezone=True), nullable=True) # Added for tracking status changes

    @hybrid_property
    def password(self) -> Optional[str]:
        if not UTILS_AND_SETTINGS_AVAILABLE or not app_settings.DATABASE_ENCRYPTION_KEY:
            logger = logging.getLogger(__name__)
            logger.error(
                f"Attempted to decrypt password for account '{self.account_identifier}' but encryption utils/key are unavailable. "
                "This is a CRITICAL configuration issue."
            )
            return "[PASSWORD_DECRYPTION_UNAVAILABLE]"
        return decrypt_data(self._encrypted_password) if self._encrypted_password else None

    @password.setter
    def password(self, value: Optional[str]) -> None:
        if not UTILS_AND_SETTINGS_AVAILABLE or not app_settings.DATABASE_ENCRYPTION_KEY:
            logger = logging.getLogger(__name__)
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
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)) # Renamed from created_at
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    source = Column(String(100), nullable=False) # e.g., ThinkTool, OperatorUI, SystemEvent
    target_agent = Column(SAEnum(AgentName), nullable=False, index=True)
    directive_type = Column(String(100), nullable=False, index=True) # e.g., initiate_outreach_campaign, analyze_market_trend, test_new_exploit
    
    content = Column(JSONB, nullable=False, comment="Detailed parameters and context for the directive.")
    priority = Column(Integer, default=5, nullable=False, index=True) # 1 (highest) to 10 (lowest)
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True) # Use TaskStatus enum
    
    result_summary = Column(Text, nullable=True)
    related_task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True) # Link to a primary task if applicable
    notes = Column(Text, nullable=True) # For human-readable notes about the directive

    def __repr__(self) -> str:
        return f"<StrategicDirective(id={self.id}, type='{self.directive_type}', target='{self.target_agent.value}', status='{self.status.value}')>"


class KnowledgeFragment(Base): # type: ignore
    __tablename__ = "knowledge_fragments"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)) # Renamed from created_at
    last_accessed_ts = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True) # Added for LRU cache/purge
    
    agent_source = Column(SAEnum(AgentName), nullable=False) # Which agent logged this
    data_type = Column(String(100), index=True, nullable=False) # e.g., client_osint_summary, email_template_performance, competitor_pricing_info
    content = Column(Text, nullable=False) # Can be JSON string, plain text, etc.
    item_hash = Column(String(64), nullable=False, unique=True, index=True) # SHA256 hash of content for deduplication
    
    relevance_score = Column(Float, default=0.5, nullable=False)
    confidence_score = Column(Float, default=0.5, nullable=False) # Added for ThinkTool's assessment
    
    tags = Column(JSONB, nullable=True, default=list) # List of strings
    related_client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True)
    source_reference = Column(Text, nullable=True) # URL, document name, task ID that generated this
    related_directive_id = Column(UUID(as_uuid=True), ForeignKey("strategic_directives.id", ondelete="SET NULL"), nullable=True) # Link to directive

    def __repr__(self) -> str:
        return f"<KnowledgeFragment(id={self.id}, type='{self.data_type}', source='{self.agent_source.value}')>"

# ADDED LearnedPattern CLASS DEFINITION
class LearnedPattern(Base): # type: ignore
    __tablename__ = "learned_patterns"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    pattern_description = Column(Text, nullable=False)
    pattern_type = Column(String(100), nullable=True, index=True, comment="e.g., observational, causal, exploit_hypothesis, successful_tactic")
    
    supporting_fragment_ids = Column(JSONB, nullable=True, comment="List of KnowledgeFragment IDs supporting this pattern")
    confidence_score = Column(Float, default=0.0, nullable=False, index=True)
    implications = Column(Text, nullable=True, comment="What this pattern implies or enables")
    
    tags = Column(JSONB, nullable=True, default=list) # List of strings for searchable tags
    status = Column(String(50), default="active", nullable=False, index=True) # e.g., active, deprecated, under_review
    
    usage_count = Column(Integer, default=0, nullable=False)
    last_applied_at = Column(DateTime(timezone=True), nullable=True)
    
    potential_exploit_details = Column(Text, nullable=True, comment="Specifics if this pattern relates to an exploit")
    related_service_or_platform = Column(String(100), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<LearnedPattern(id={self.id}, type='{self.pattern_type}', confidence={self.confidence_score:.2f})>"


class Invoice(Base): # type: ignore
    __tablename__ = "invoices"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)) # Renamed from issue_date
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True)
    client = relationship("Client", back_populates="invoices", lazy="joined")
    
    invoice_number = Column(String(50), unique=True, nullable=False, index=True)
    due_date = Column(DateTime(timezone=True), nullable=False)
    
    amount = Column(Float, nullable=False) # Renamed from total_amount
    currency = Column(String(10), default="USD", nullable=False)
    status = Column(SAEnum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False, index=True)
    
    line_items = Column(JSONB, nullable=False)
    notes = Column(Text, nullable=True) # Renamed from notes_to_client
    
    paid_at = Column(DateTime(timezone=True), nullable=True)
    payment_method = Column(String(50), nullable=True) # Renamed from payment_method_details (simpler)
    
    # Relationship to FinancialTransaction
    financial_transactions = relationship("FinancialTransaction", back_populates="invoice", cascade="all, delete-orphan", lazy="selectin")

    __table_args__ = (
        CheckConstraint('amount >= 0', name='chk_invoice_amount_positive'), # Updated constraint name
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
    # client = relationship("Client") # Can be added if direct link needed often
    
    external_transaction_id = Column(String(255), nullable=True, index=True)
    payment_gateway = Column(String(50), nullable=True) # Renamed from payment_gateway_name
    
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
    # task = relationship("Task") # Can add if needed

    author_agent_name = Column(SAEnum(AgentName), nullable=True)
    author_user_identifier = Column(String(255), nullable=True, comment="Identifier for a human user if UI allows manual notes")
    
    content = Column(Text, nullable=False)
    note_type = Column(String(50), default="general", nullable=False) # e.g., general, meeting_summary, important_info
    is_pinned = Column(Boolean, default=False, nullable=False)
    def __repr__(self) -> str:
        return f"<Note(id={self.id}, type='{self.note_type}', client_id={self.client_id}, content='{self.content[:30]}...')>"

# --- Additional Models from other agent files (EmailLog, CallLog, ConversationState, PromptTemplate, ExpenseLog) ---

class EmailLog(Base): # type: ignore
    __tablename__ = "email_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey('clients.id', ondelete="SET NULL"), nullable=True, index=True)
    recipient = Column(String(255), nullable=False, index=True)
    subject = Column(String(512), nullable=False)
    content_preview = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, index=True) # e.g., sent, opened, clicked, responded, bounced, failed_verification
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    agent_version = Column(String(50), nullable=True)
    sender_account = Column(String(255), nullable=True) # e.g., MailerSend, SMTP1
    message_id = Column(String(255), nullable=True, unique=True, index=True) # Provider's message ID
    opened_at = Column(DateTime(timezone=True), nullable=True)
    responded_at = Column(DateTime(timezone=True), nullable=True)
    # Removed composition_ids, as this can be complex. Store in json_data if needed.
    json_data = Column(JSONB, nullable=True, comment="Extra data like composition IDs, campaign ID, etc.")

class CallLog(Base): # type: ignore
    __tablename__ = "call_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey('clients.id', ondelete="SET NULL"), nullable=True, index=True)
    call_sid = Column(String(255), unique=True, nullable=False, index=True) # Twilio Call SID
    phone_number = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    transcript = Column(Text, nullable=True) # Full transcript as JSON string or text
    outcome = Column(String(100), nullable=True, index=True) # e.g., success_sale, success_meeting_booked, no_answer, voicemail, failed_compliance
    duration_seconds = Column(Integer, nullable=True)
    recording_url = Column(String(1024), nullable=True)
    final_twilio_status = Column(String(50), nullable=True) # e.g., completed, failed, no-answer
    json_data = Column(JSONB, nullable=True, comment="Additional call metadata")

class ConversationState(Base): # type: ignore
    __tablename__ = "conversation_states"
    call_sid = Column(String(255), primary_key=True) # Using Twilio Call SID as PK
    state = Column(String(100), nullable=False)
    conversation_log = Column(Text, nullable=False) # JSON string of conversation turns
    discovered_needs_log = Column(Text, nullable=True) # JSON string of discovered needs
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

class PromptTemplate(Base): # type: ignore
    __tablename__ = "prompt_templates"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_name = Column(SAEnum(AgentName), nullable=False, index=True)
    prompt_key = Column(String(100), nullable=False, index=True) # e.g., 'email_subject_generation', 'voice_call_greeting'
    version = Column(Integer, default=1, nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    author_agent = Column(SAEnum(AgentName), nullable=True) # Which agent authored/updated this
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    notes = Column(Text, nullable=True) # For critique summaries or version notes
    __table_args__ = (UniqueConstraint('agent_name', 'prompt_key', 'version', name='uq_prompt_agent_key_version'),)

class ExpenseLog(Base): # type: ignore
    __tablename__ = "expense_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    agent_name = Column(SAEnum(AgentName), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    category = Column(String(100), nullable=False, index=True) # e.g., LLM, API_Call_Twilio, API_Call_Deepgram, Proxy
    description = Column(Text, nullable=True)
    task_id_reference = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True) # Link to task if applicable
    json_data = Column(JSONB, nullable=True, comment="Additional details like token counts, API call specifics")


# --- Optional: Function to create tables (for dev/testing, Alembic for prod) ---
def create_all_tables_dev(db_url_sync: str) -> None:
    """
    Development utility to create all tables.
    WARNING: For production, use a migration tool like Alembic.
    Expects a synchronous database URL.
    """
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
            "Please check the logs for import errors related to these modules and their dependencies "
            "(e.g., missing environment variables like DATABASE_ENCRYPTION_KEY)."
        )
    elif app_settings.DATABASE_URL:
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
            "DATABASE_URL not set in environment (via .env or system variables). "
            "Cannot create tables. Please configure DATABASE_URL."
            "Example: DATABASE_URL='postgresql+asyncpg://user:pass@host:port/dbname'"
        )

# --- End of models.py ---