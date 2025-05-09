# Filename: models.py
# Description: SQLAlchemy Models for the AI Agency Database.
# Version: 2.1 (IGNIS Transmutation - Standardized Naming, Financials, Relationships)

import uuid
import enum
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    create_engine, Column, String, DateTime, ForeignKey, Text, Boolean,
    Integer, Float, Enum, UniqueConstraint, Index, CheckConstraint, LargeBinary
)
from sqlalchemy.orm import relationship, validates, sessionmaker, declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Import encryption utilities and custom Base AFTER settings are loaded
# to ensure DATABASE_ENCRYPTION_KEY is available if utils.database uses it at import.
# However, utils.database is designed to fetch the key on first use of encrypt/decrypt.
try:
    from utils.database import encrypt_data, decrypt_data, Base as CustomBase
    from config.settings import settings # For potential direct use or context
    UTILS_AVAILABLE = True
except ImportError as e:
    # This fallback is problematic for model definitions if encryption is integral.
    # The real fix is to ensure utils.database and config.settings are importable and configured.
    import logging
    logging.getLogger(__name__).critical(f"CRITICAL: Failed to import utils/database or config.settings for models.py: {e}. Encryption/Decryption will fail.")
    CustomBase = declarative_base() # type: ignore
    def encrypt_data(data): return data # Dummy
    def decrypt_data(data): return data # Dummy
    class settings: DATABASE_ENCRYPTION_KEY = None # Dummy
    UTILS_AVAILABLE = False


# Use the custom Base from utils.database if available, otherwise SQLAlchemy's default
Base = CustomBase if UTILS_AVAILABLE else declarative_base() # type: ignore

# --- Enum Definitions ---
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
    HALTED_BY_REFLECTION = "halted_by_reflection" # ThinkTool specific
    SKIPPED = "skipped" # e.g. client already contacted


class AgentName(enum.Enum):
    ORCHESTRATOR = "Orchestrator"
    THINK_TOOL = "ThinkTool"
    EMAIL_AGENT = "EmailAgent"
    VOICE_SALES_AGENT = "VoiceSalesAgent"
    SOCIAL_MEDIA_MANAGER = "SocialMediaManager"
    LEGAL_AGENT = "LegalAgent"
    GMAIL_CREATOR_AGENT = "GmailCreatorAgent"
    Browse_AGENT = "BrowseAgent"
    VIDEO_CREATION_AGENT = "VideoCreationAgent" # Added for video generation
    CLIENT_RESEARCH_AGENT = "ClientResearchAgent" # For Clay.com or other research
    PAYMENT_PROCESSOR_AGENT = "PaymentProcessorAgent" # Hypothetical for payment interactions
    UNKNOWN = "Unknown"


class ClientStatus(enum.Enum):
    LEAD = "lead"  # Initial state from discovery
    CONTACTED = "contacted"  # First outreach made
    ENGAGED = "engaged"  # Two-way communication established
    QUALIFIED = "qualified"  # Meets ideal customer profile
    PROPOSAL_SENT = "proposal_sent"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"  # Deal won
    CLOSED_LOST = "closed_lost"  # Deal lost
    ONBOARDING = "onboarding"
    ACTIVE_SERVICE = "active_service" # Receiving ongoing services
    DORMANT = "dormant"  # Previously active, now inactive
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
    CALL_COMPLETED = "call_completed" # Successful conversation
    CALL_VOICEMAIL_LEFT = "call_voicemail_left"
    CALL_FAILED = "call_failed" # Technical failure, busy, no answer after retries
    CALL_SCHEDULED = "call_scheduled"
    SOCIAL_MEDIA_POST = "social_media_post"
    SOCIAL_MEDIA_DM = "social_media_dm"
    WEBSITE_VISIT_TRACKED = "website_visit_tracked" # If tracking is implemented
    FORM_SUBMISSION = "form_submission"
    MEETING_SCHEDULED = "meeting_scheduled"
    MEETING_COMPLETED = "meeting_completed"
    INVOICE_SENT = "invoice_sent"
    PAYMENT_RECEIVED = "payment_received"
    NOTE_ADDED = "note_added" # Manual note by system or operator
    TASK_COMPLETED_FOR_CLIENT = "task_completed_for_client" # Generic task completion


class PaymentStatus(enum.Enum):
    PENDING = "pending"
    PAID = "paid"
    COMPLETED = "completed" # Alias for PAID, more generic for financial transactions
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_PAID = "partially_paid"
    PARTIALLY_REFUNDED = "partially_refunded"
    VOIDED = "voided"
    OVERDUE = "overdue"


# --- Model Definitions ---

class Client(Base): # type: ignore
    __tablename__ = "clients"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Company & Contact Info
    company_name = Column(String(255), nullable=False, index=True)
    website = Column(String(512), nullable=True)
    primary_contact_name = Column(String(255), nullable=True)
    primary_contact_email = Column(String(255), nullable=True, index=True, unique=True) # Could be unique if strictly one client per email
    primary_contact_phone = Column(String(50), nullable=True) # Store in E.164 if possible
    industry = Column(String(100), nullable=True)
    company_size = Column(String(50), nullable=True) # e.g., "1-10 employees", "50-200", or integer range
    country = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    
    # Status & Engagement
    client_status = Column(Enum(ClientStatus), default=ClientStatus.LEAD, nullable=False)
    opt_in_status = Column(String(50), default="pending", index=True) # e.g., pending, opted_in, opted_out
    client_score = Column(Float, default=0.0) # Lead score or engagement score
    source = Column(String(100), nullable=True) # e.g., clay_import, manual, web_form
    
    # Financial & Service Details
    service_tier = Column(String(100), nullable=True) # e.g., basic, premium
    contract_start_date = Column(DateTime(timezone=True), nullable=True)
    contract_end_date = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    tasks = relationship("Task", back_populates="client", cascade="all, delete-orphan")
    interactions = relationship("InteractionLog", back_populates="client", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="client", cascade="all, delete-orphan")
    notes = relationship("Note", back_populates="client", cascade="all, delete-orphan")
    # Custom fields (JSONB for flexibility)
    custom_fields = Column(JSONB, nullable=True, default=dict) # Store additional structured data

    __table_args__ = (
        Index('idx_client_company_name_email', 'company_name', 'primary_contact_email'),
    )

    def __repr__(self) -> str:
        return f"<Client(id={self.id}, company_name='{self.company_name}', email='{self.primary_contact_email}')>"


class Task(Base): # type: ignore
    __tablename__ = "tasks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    agent_name = Column(Enum(AgentName), nullable=False, index=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=True, index=True) # Can be null for system tasks
    client = relationship("Client", back_populates="tasks")

    priority = Column(Integer, default=0) # Higher number = higher priority
    payload = Column(JSONB) # Task specific data, e.g., email content, call script context
    result = Column(JSONB, nullable=True) # Result of the task execution
    error_message = Column(Text, nullable=True)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    
    scheduled_at = Column(DateTime(timezone=True), nullable=True) # For tasks to be run at a specific time
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # For ThinkTool or multi-step tasks
    parent_task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True) # Self-referential for sub-tasks
    sub_tasks = relationship("Task", backref="parent_task", remote_side=[id], cascade="all, delete-orphan")
    
    # For ThinkTool: sequence of actions taken
    action_history = Column(JSONB, nullable=True, default=list) # List of dicts: {"action": "...", "params": {...}, "timestamp": "...", "outcome": "..."}
    current_objective = Column(Text, nullable=True) # Current specific goal for this task
    
    # For learning/reflection
    knowledge_ids_used = Column(JSONB, nullable=True, default=list) # List of KnowledgeFragment IDs used
    knowledge_ids_generated = Column(JSONB, nullable=True, default=list) # List of KnowledgeFragment IDs created


    def __repr__(self) -> str:
        return f"<Task(id={self.id}, agent='{self.agent_name.value}', status='{self.status.value}')>"


class InteractionLog(Base): # type: ignore
    __tablename__ = "interaction_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=False, index=True)
    client = relationship("Client", back_populates="interactions")
    
    agent_name = Column(Enum(AgentName), nullable=True) # Agent responsible, if any
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True) # Associated task, if any
    
    type = Column(Enum(InteractionType), nullable=False, index=True)
    channel = Column(String(50), nullable=True) # e.g., email, phone, linkedin, website
    content_summary = Column(Text, nullable=True) # e.g., email subject, call summary snippet
    external_id = Column(String(255), nullable=True, index=True) # e.g., email message ID, call SID
    
    metadata = Column(JSONB, nullable=True) # Extra details, e.g., email open location, call duration, link clicked

    def __repr__(self) -> str:
        return f"<InteractionLog(id={self.id}, client_id={self.client_id}, type='{self.type.value}', ts='{self.timestamp}')>"

# Simplified EmailLog and CallLog, can be absorbed into InteractionLog or kept separate for detail
class EmailLog(Base): # type: ignore
    __tablename__ = "email_logs" # Consider merging into InteractionLog
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), index=True)
    email_address = Column(String(255), nullable=False)
    subject = Column(Text, nullable=True)
    body_hash = Column(String(64), nullable=True) # To detect duplicate content
    status = Column(String(50), nullable=True) # e.g., sent, opened, clicked, bounced, replied, failed
    external_message_id = Column(String(255), nullable=True, index=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)

class CallLog(Base): # type: ignore
    __tablename__ = "call_logs" # Consider merging into InteractionLog
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), index=True)
    phone_number = Column(String(50), nullable=False)
    duration_seconds = Column(Integer, nullable=True)
    status = Column(String(50), nullable=True) # e.g., initiated, answered, completed, voicemail, failed, busy
    recording_url = Column(String(512), nullable=True)
    call_sid = Column(String(255), nullable=True, index=True) # Twilio SID or similar
    transcription_summary = Column(Text, nullable=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)


class AccountCredentials(Base): # type: ignore
    __tablename__ = "account_credentials"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    service_name = Column(String(100), nullable=False, index=True) # e.g., Gmail, LinkedIn, Twitter
    username = Column(String(255), nullable=False, index=True)
    
    _encrypted_password = Column("encrypted_password", String(1024), nullable=True) # Encrypted
    
    email_associated = Column(String(255), nullable=True, index=True) # Often same as username for email services
    status = Column(String(50), default="active") # e.g., active, needs_review, banned, needs_password_reset, locked
    last_used = Column(DateTime(timezone=True), nullable=True)
    proxy_info = Column(JSONB, nullable=True) # Details of proxy used during creation or last use
    creation_agent = Column(Enum(AgentName), nullable=True)
    metadata = Column(JSONB, nullable=True) # e.g., security questions, recovery email used (if any)

    # Ensure DATABASE_ENCRYPTION_KEY is set in .env for these to work
    @hybrid_property
    def password(self) -> Optional[str]:
        if not UTILS_AVAILABLE or not settings.DATABASE_ENCRYPTION_KEY: return "[Encryption Unavailable]"
        return decrypt_data(self._encrypted_password) if self._encrypted_password else None

    @password.setter
    def password(self, value: Optional[str]) -> None:
        if not UTILS_AVAILABLE or not settings.DATABASE_ENCRYPTION_KEY:
            raise EnvironmentError("DATABASE_ENCRYPTION_KEY not configured. Cannot encrypt password.")
        if value is None:
            self._encrypted_password = None
        else:
            self._encrypted_password = encrypt_data(value)
            
    __table_args__ = (UniqueConstraint('service_name', 'username', name='uq_service_username'),)


class StrategicDirective(Base): # type: ignore
    __tablename__ = "strategic_directives" # For ThinkTool
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False) # High-level goal or strategy
    source = Column(String(100)) # e.g., operator_input, system_generated_reflection
    priority = Column(Integer, default=0) # Higher is more important
    status = Column(String(50), default="active") # e.g., active, completed, paused, archived
    parameters = Column(JSONB, nullable=True) # Specific constraints or targets for the directive
    
    # Performance tracking against directive
    kpis_to_track = Column(JSONB, nullable=True) # e.g., {"conversion_rate": 0.05, "leads_generated_per_week": 10}
    current_performance = Column(JSONB, nullable=True) # Actual values for kpis_to_track
    
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True) # Directives can be time-bound


class KnowledgeFragment(Base): # type: ignore
    __tablename__ = "knowledge_fragments" # For ThinkTool learning
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    type = Column(String(100), index=True) # e.g., successful_email_template, effective_call_script_opener, client_objection_handling
    content = Column(Text, nullable=False) # The actual knowledge content
    metadata = Column(JSONB, nullable=True) # Source, context, tags, confidence score, performance metrics
    source_agent = Column(Enum(AgentName), nullable=True)
    source_task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)
    related_client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=True)
    
    # For vectorization / embeddings if used
    embedding_vector = Column(LargeBinary, nullable=True) # Store embeddings (e.g. from OpenAI)
    embedding_model = Column(String(100), nullable=True) # e.g., text-embedding-ada-002
    
    # Learning metrics
    effectiveness_score = Column(Float, default=0.0) # How effective this fragment has been
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships or tags for easier retrieval
    tags = Column(JSONB, nullable=True, default=list) # List of strings


class Invoice(Base): # type: ignore
    __tablename__ = "invoices"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True) # Keep invoice even if client is deleted
    client = relationship("Client", back_populates="invoices")
    
    invoice_number = Column(String(50), unique=True, nullable=False)
    issue_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    due_date = Column(DateTime(timezone=True), nullable=False)
    
    total_amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False, index=True)
    
    line_items = Column(JSONB) # [{"description": "...", "quantity": 1, "unit_price": 100, "total": 100}, ...]
    notes_to_client = Column(Text, nullable=True)
    
    paid_at = Column(DateTime(timezone=True), nullable=True)
    payment_method = Column(String(50), nullable=True) # e.g., credit_card, bank_transfer
    
    # Relationships
    financial_transactions = relationship("FinancialTransaction", back_populates="invoice", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint('total_amount >= 0', name='chk_invoice_total_amount_positive'),
    )

class FinancialTransaction(Base): # type: ignore
    __tablename__ = "financial_transactions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    type = Column(String(50), nullable=False, index=True) # e.g., payment, refund, fee, commission_payout
    status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False, index=True)
    
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    
    description = Column(Text, nullable=True)
    
    # Related entities
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id", ondelete="SET NULL"), nullable=True, index=True)
    invoice = relationship("Invoice", back_populates="financial_transactions")
    
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True) # For payments not tied to a specific invoice directly
    
    external_transaction_id = Column(String(255), nullable=True, index=True) # e.g., Stripe charge ID
    payment_gateway = Column(String(50), nullable=True) # e.g., Stripe, PayPal
    
    metadata = Column(JSONB, nullable=True) # e.g., gateway response, fee details


class Note(Base): # type: ignore
    __tablename__ = "notes"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=True)
    client = relationship("Client", back_populates="notes")

    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True) # Note related to a specific task

    author_agent_name = Column(Enum(AgentName), nullable=True) # If an agent created this note
    author_user_id = Column(String(255), nullable=True) # If a human user created this note (placeholder for future user model)
    
    content = Column(Text, nullable=False)
    type = Column(String(50), default="general") # e.g., general, meeting_summary, important_info, client_feedback
    is_pinned = Column(Boolean, default=False)

    def __repr__(self):
        return f"<Note(id={self.id}, client_id={self.client_id}, type='{self.type}', content='{self.content[:50]}...')>"


# --- Function to create tables (optional, can be managed by Alembic) ---
def create_all_tables(db_url: str) -> None:
    """Creates all tables in the database."""
    if not db_url:
        logger.error("Database URL not provided. Cannot create tables.")
        return
    try:
        engine = create_engine(db_url)
        Base.metadata.create_all(engine) # type: ignore
        logger.info("All tables created successfully (if they didn't exist).")
    except Exception as e:
        logger.error(f"Error creating tables: {e}", exc_info=True)

if __name__ == "__main__":
    # Example usage:
    # This is for direct execution for table creation, ensure DATABASE_URL is set in .env
    # In a real app, you'd likely use Alembic for migrations.
    if not settings.DATABASE_URL:
        print("DATABASE_URL not set in environment. Cannot create tables.")
        print("Please create a .env file with DATABASE_URL='postgresql+asyncpg://user:pass@host:port/dbname'")
    else:
        print(f"Attempting to create tables for database: {settings.DATABASE_URL}")
        try:
            # For direct execution, use a synchronous engine if create_engine is synchronous by default.
            # If using asyncpg, the create_engine from sqlalchemy.ext.asyncio should be used,
            # but create_all is a synchronous operation.
            # For simplicity in this direct script, using a standard create_engine.
            from sqlalchemy import create_engine as create_sync_engine
            sync_db_url = settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
            engine = create_sync_engine(sync_db_url)
            Base.metadata.create_all(engine) # type: ignore
            print("Tables created successfully (if they didn't exist).")
            print("IMPORTANT: For production, use Alembic for database migrations.")
        except Exception as e:
            print(f"Error creating tables: {e}")
            print("Ensure your database server is running and accessible, and credentials are correct.")

# --- End of models.py ---