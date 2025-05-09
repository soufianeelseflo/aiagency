# Filename: models.py
# Description: SQLAlchemy Models for the AI Agency Database.
# Version: 3.0 (IGNIS Final Transmutation - SQLAlchemy 'metadata' fix, relationships, enums)

from typing import Optional,Any, List, Dict
import uuid
import enum
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    create_engine, Column, String, DateTime, ForeignKey, Text, Boolean,
    Integer, Float, Enum as SAEnum, UniqueConstraint, Index, CheckConstraint, LargeBinary
) # Renamed Enum to SAEnum to avoid conflict with standard Python enum
from sqlalchemy.orm import relationship, validates, sessionmaker, declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import UUID, JSONB

# --- Settings and Utilities Import ---
# This structure attempts to import essential utilities. If they fail,
# it logs critically and uses fallbacks that will limit functionality,
# signaling a severe configuration issue that must be manually resolved.
try:
    from utils.database import encrypt_data, decrypt_data, Base as CustomBaseFromUtil
    from config.settings import settings as app_settings # Renamed to avoid conflict
    UTILS_AND_SETTINGS_AVAILABLE = True
except ImportError as e_import_utils:
    import logging # Ensure logging is available for this critical fallback
    logging.basicConfig(level=logging.ERROR) # Basic config if not already set
    logger_models = logging.getLogger(__name__)
    logger_models.critical(
        f"CRITICAL FAILURE in models.py: Could not import 'utils.database' or 'config.settings'. "
        f"Cause: {e_import_utils}. Database encryption and potentially other model functionalities will be broken. "
        "This indicates a severe problem in project structure or dependencies that needs immediate attention."
    )
    CustomBaseFromUtil = declarative_base() # type: ignore
    # Define dummy encryption functions that will fail if DATABASE_ENCRYPTION_KEY is expected but not found by a real function
    def encrypt_data(data: Optional[str]) -> Optional[str]:
        logger_models.error("Dummy encrypt_data called: Encryption unavailable due to import failure.")
        return data # Or raise an error
    def decrypt_data(data: Optional[str]) -> Optional[str]:
        logger_models.error("Dummy decrypt_data called: Decryption unavailable due to import failure.")
        return data # Or raise an error
    class DummyAppSettings: # Minimal fallback for settings
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
    Browse_AGENT = "BrowseAgent"
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
    PAID = "paid" # Main successful status for invoice payments
    COMPLETED = "completed" # Generic success for other financial transactions
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
    DISABLED = "disabled" # Manually disabled by operator

# --- Model Definitions ---

class Client(Base): # type: ignore
    __tablename__ = "clients"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    company_name = Column(String(255), nullable=False, index=True)
    website = Column(String(512), nullable=True)
    primary_contact_name = Column(String(255), nullable=True)
    primary_contact_email = Column(String(255), nullable=True, index=True) # Consider unique=True if it's a hard constraint
    primary_contact_phone = Column(String(50), nullable=True)
    industry = Column(String(100), nullable=True)
    company_size = Column(String(50), nullable=True)
    country = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    
    client_status = Column(SAEnum(ClientStatus), default=ClientStatus.LEAD, nullable=False)
    opt_in_status = Column(String(50), default="pending", index=True) # e.g., pending, opted_in, opted_out_email, opted_out_calls
    client_score = Column(Float, default=0.0, nullable=False)
    source = Column(String(100), nullable=True) # e.g., clay_import, manual_entry, web_form
    
    service_tier = Column(String(100), nullable=True)
    contract_start_date = Column(DateTime(timezone=True), nullable=True)
    contract_end_date = Column(DateTime(timezone=True), nullable=True)

    # Renamed 'metadata' to 'json_data'
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
    client = relationship("Client", back_populates="tasks", lazy="joined") # Eager load client for common task views

    priority = Column(Integer, default=0, nullable=False)
    payload = Column(JSONB, comment="Task-specific input data.")
    result = Column(JSONB, nullable=True, comment="Task-specific output data.")
    error_message = Column(Text, nullable=True)
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    parent_task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True) # Allow parent deletion without cascading to subtasks directly, manage logic in app
    sub_tasks = relationship("Task", backref="parent_task", remote_side=[id], lazy="selectin") # Avoids full cascade delete if parent task removed
    
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
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True) # Keep log even if task is deleted
    
    type = Column(SAEnum(InteractionType), nullable=False, index=True)
    channel = Column(String(50), nullable=True)
    content_summary = Column(Text, nullable=True)
    external_id = Column(String(255), nullable=True, index=True)
    
    # Renamed 'metadata' to 'json_data'
    json_data = Column(JSONB, nullable=True, comment="Flexible JSON field for extra details like email open location, call duration, etc.")

    def __repr__(self) -> str:
        return f"<InteractionLog(id={self.id}, client_id={self.client_id}, type='{self.type.value}', ts='{self.timestamp}')>"

class AccountCredentials(Base): # type: ignore
    __tablename__ = "account_credentials"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    service_name = Column(String(100), nullable=False, index=True)
    username = Column(String(255), nullable=False, index=True)
    
    _encrypted_password = Column("encrypted_password", String(1024), nullable=True)
    
    email_associated = Column(String(255), nullable=True, index=True)
    status = Column(SAEnum(AccountStatus), default=AccountStatus.ACTIVE, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    proxy_details = Column(JSONB, nullable=True, comment="Details of proxy used during creation or last successful use.")
    creation_agent_name = Column(SAEnum(AgentName), nullable=True)
    
    # Renamed 'metadata' to 'json_data'
    json_data = Column(JSONB, nullable=True, comment="Flexible JSON field for security questions, recovery email, 2FA codes (encrypted within JSON if highly sensitive), etc.")

    @hybrid_property
    def password(self) -> Optional[str]:
        if not UTILS_AND_SETTINGS_AVAILABLE or not app_settings.DATABASE_ENCRYPTION_KEY:
            # Log this failure clearly, as it implies a critical misconfiguration.
            logger = logging.getLogger(__name__) # Ensure logger is accessible
            logger.error(
                f"Attempted to decrypt password for account '{self.username}' but encryption utils/key are unavailable. "
                "This is a CRITICAL configuration issue."
            )
            return "[PASSWORD_DECRYPTION_UNAVAILABLE]" # Or raise an exception
        return decrypt_data(self._encrypted_password) if self._encrypted_password else None

    @password.setter
    def password(self, value: Optional[str]) -> None:
        if not UTILS_AND_SETTINGS_AVAILABLE or not app_settings.DATABASE_ENCRYPTION_KEY:
            logger = logging.getLogger(__name__)
            logger.critical(
                f"Attempted to encrypt password for account '{self.username}' but encryption utils/key are unavailable. "
                "Password NOT set. This is a CRITICAL configuration issue."
            )
            # It's crucial not to store the password in plain text or a dummy encrypted form.
            # Raising an error is safer to prevent data mishandling.
            raise EnvironmentError(
                "DATABASE_ENCRYPTION_KEY not configured or utils unavailable. Cannot encrypt password."
            )
        if value is None:
            self._encrypted_password = None
        else:
            self._encrypted_password = encrypt_data(value)
            
    __table_args__ = (UniqueConstraint('service_name', 'username', name='uq_account_service_username'),)
    def __repr__(self) -> str:
        return f"<AccountCredentials(id={self.id}, service='{self.service_name}', username='{self.username}', status='{self.status.value}')>"


class StrategicDirective(Base): # type: ignore
    __tablename__ = "strategic_directives"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    source = Column(String(100), nullable=False) # e.g., operator_input, system_reflection
    priority = Column(Integer, default=0, nullable=False)
    status = Column(String(50), default="active", nullable=False) # e.g., active, completed, paused, archived
    
    # Renamed 'parameters' to 'directive_parameters' to be more specific
    directive_parameters = Column(JSONB, nullable=True, comment="Specific constraints or targets for the directive.")
    
    kpis_to_track = Column(JSONB, nullable=True)
    current_performance_metrics = Column(JSONB, nullable=True)
    
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    def __repr__(self) -> str:
        return f"<StrategicDirective(id={self.id}, title='{self.title}', status='{self.status}')>"


class KnowledgeFragment(Base): # type: ignore
    __tablename__ = "knowledge_fragments"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    type = Column(String(100), index=True, nullable=False)
    content = Column(Text, nullable=False)
    
    # Renamed 'metadata' to 'json_data'
    json_data = Column(JSONB, nullable=True, comment="Source, context, tags, confidence score, performance metrics, etc.")
    
    source_agent_name = Column(SAEnum(AgentName), nullable=True)
    source_task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    related_client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True)
    
    embedding_vector = Column(LargeBinary, nullable=True)
    embedding_model_name = Column(String(100), nullable=True)
    
    effectiveness_score = Column(Float, default=0.0, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    tags = Column(JSONB, nullable=True, default=list) # List of strings for searchable tags
    def __repr__(self) -> str:
        return f"<KnowledgeFragment(id={self.id}, type='{self.type}', usage_count={self.usage_count})>"


class Invoice(Base): # type: ignore
    __tablename__ = "invoices"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True) # Keep invoice even if client deleted
    client = relationship("Client", back_populates="invoices", lazy="joined")
    
    invoice_number = Column(String(50), unique=True, nullable=False, index=True)
    issue_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    due_date = Column(DateTime(timezone=True), nullable=False)
    
    total_amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    status = Column(SAEnum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False, index=True)
    
    line_items = Column(JSONB, nullable=False) # [{"description": "...", "quantity": 1, "unit_price": 100, "total": 100}, ...]
    notes_to_client = Column(Text, nullable=True)
    
    paid_at = Column(DateTime(timezone=True), nullable=True)
    payment_method_details = Column(JSONB, nullable=True, comment="e.g., card_last4, transaction_type") # More specific than just string
    
    financial_transactions = relationship("FinancialTransaction", back_populates="invoice", cascade="all, delete-orphan", lazy="selectin")

    __table_args__ = (
        CheckConstraint('total_amount >= 0', name='chk_invoice_total_amount_positive'),
        Index('idx_invoice_status_due_date', 'status', 'due_date'),
    )
    def __repr__(self) -> str:
        return f"<Invoice(id={self.id}, number='{self.invoice_number}', status='{self.status.value}', total={self.total_amount})>"


class FinancialTransaction(Base): # type: ignore
    __tablename__ = "financial_transactions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False)
    
    type = Column(String(50), nullable=False, index=True) # e.g., payment, refund, fee, commission_payout, adjustment
    status = Column(SAEnum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False, index=True)
    
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    
    description = Column(Text, nullable=True)
    
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id", ondelete="SET NULL"), nullable=True, index=True)
    invoice = relationship("Invoice", back_populates="financial_transactions", lazy="joined")
    
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True)
    # client = relationship("Client") # Could add if needed, but invoice.client is often sufficient
    
    external_transaction_id = Column(String(255), nullable=True, index=True)
    payment_gateway_name = Column(String(50), nullable=True)
    
    # Renamed 'metadata' to 'json_data'
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
    note_type = Column(String(50), default="general", nullable=False) # e.g., general, meeting_summary, important_info
    is_pinned = Column(Boolean, default=False, nullable=False)
    def __repr__(self) -> str:
        return f"<Note(id={self.id}, type='{self.note_type}', client_id={self.client_id}, content='{self.content[:30]}...')>"


# --- Optional: Function to create tables (for dev/testing, Alembic for prod) ---
def create_all_tables_dev(db_url_sync: str) -> None:
    """
    Development utility to create all tables.
    WARNING: For production, use a migration tool like Alembic.
    Expects a synchronous database URL.
    """
    logger_models_dev = logging.getLogger(__name__ + "_dev_create") # Separate logger
    if not db_url_sync:
        logger_models_dev.error("Synchronous Database URL not provided. Cannot create tables.")
        return
    try:
        # Ensure using a synchronous engine for create_all
        engine = create_engine(db_url_sync)
        Base.metadata.create_all(engine) # type: ignore
        logger_models_dev.info(f"All tables created successfully (if they didn't exist) on {db_url_sync}.")
    except Exception as e_create:
        logger_models_dev.error(f"Error creating tables on {db_url_sync}: {e_create}", exc_info=True)

if __name__ == "__main__":
    # This block is for direct script execution (e.g., python models.py create_tables)
    # Primarily for initial setup in a development environment.
    # Production environments should use Alembic for schema migrations.
    logging.basicConfig(level=logging.INFO) # Ensure logging is configured for this script execution
    main_logger = logging.getLogger(__name__ + "_main")

    if not UTILS_AND_SETTINGS_AVAILABLE:
        main_logger.critical(
            "Cannot proceed with __main__ block in models.py because critical utilities "
            "(utils.database or config.settings) failed to import earlier. "
            "Please check the logs for import errors related to these modules and their dependencies "
            "(e.g., missing environment variables like DATABASE_ENCRYPTION_KEY)."
        )
    elif app_settings.DATABASE_URL:
        # Convert async DSN to sync DSN for create_all if necessary
        sync_db_url = str(app_settings.DATABASE_URL).replace("postgresql+asyncpg", "postgresql")
        sync_db_url = sync_db_url.replace("postgresql+psycopg", "postgresql") # Another common async dialect
        
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