from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import pytz

Base = declarative_base()

class Client(Base):
    __tablename__ = 'clients'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String)
    country = Column(String)
    timezone = Column(String, default="America/New_York")
    interests = Column(String)  # Comma-separated or JSON
    last_interaction = Column(DateTime, default=lambda: datetime.now(pytz.UTC))
    response_rate = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    opt_in = Column(Boolean, default=True)

class Metric(Base):
    __tablename__ = 'metrics'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(pytz.UTC))
    metric_name = Column(String)
    value = Column(Text)

class CallLog(Base):
    __tablename__ = 'call_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'))
    timestamp = Column(DateTime, default=lambda: datetime.now(pytz.UTC))
    conversation = Column(Text)  # JSON
    outcome = Column(String)

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    call_sid = Column(String, primary_key=True)
    state = Column(String)
    conversation_log = Column(Text)  # JSON

class Invoice(Base):
    __tablename__ = 'invoices'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'))
    amount = Column(Float)
    status = Column(String, default='pending')  # e.g., 'pending', 'paid'
    timestamp = Column(DateTime, default=lambda: datetime.now(pytz.UTC))

class Campaign(Base):
    __tablename__ = 'campaigns'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    status = Column(String)

class EmailLog(Base):
    __tablename__ = 'email_logs'
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'))
    timestamp = Column(DateTime, default=lambda: datetime.now(pytz.UTC))
    status = Column(String)  # e.g., 'sent', 'responded'

class Account(Base):
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True)
    service = Column(String)
    email = Column(String)
    password = Column(String)
    api_key = Column(String)
    phone = Column(String)
    cookies = Column(Text)  # JSON-encoded
    created_at = Column(DateTime, default=datetime.utcnow)