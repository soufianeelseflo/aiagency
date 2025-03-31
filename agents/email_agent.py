import asyncio
import logging
import random
import os
import json
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from utils.database import encrypt_data, decrypt_data
from models import Client, Lead, EmailLog
from openai import AsyncOpenAI as AsyncDeepSeekClient
import google.generativeai as genai
import smtplib
from email.message import EmailMessage
import pytz  # Added for timezone-aware optimal send times
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure advanced logging for genius-level debugging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class EmailAgent:
    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.clients_models = clients_models
        self.think_tool = orchestrator.agents['think']  # Added initialization
        self.task_queue = asyncio.PriorityQueue()
        self.max_concurrency = 20
        self.max_emails_per_day = 800
        self.emails_sent_today = 0
        self.email_reset_time = datetime.utcnow().replace(tzinfo=pytz.UTC) + timedelta(days=1)
        self.meta_prompt = """
        You are a brilliant email writer for an AI agency targeting USA clients. Write personalized, casual emails that feel like they’re from a real person—not a robot or salesperson. Use the client’s interests to spark a conversation or offer help, keeping it natural and engaging. Avoid spam triggers, formal tones, or anything that sounds like a typical ‘professional’ email. Ensure it passes AI detection by sounding human, and end with a low-key nudge like ‘Wanna chat about this?’
        """
        self.active_emails = 0

    def get_allowed_concurrency(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        concurrency_factor = 1 - (cpu_usage / 100) * 0.5 - (memory_usage / 100) * 0.5
        allowed = max(1, int(self.max_concurrency * concurrency_factor))
        return allowed

    async def reset_daily_limit(self):
        """Intelligently reset the email counter at midnight UTC with proactive monitoring."""
        # **Status: Already Correct** - This method was perfect in your original code.
        while True:
            now = datetime.utcnow().replace(tzinfo=pytz.UTC)
            if now >= self.email_reset_time:
                self.emails_sent_today = 0
                self.email_reset_time = now + timedelta(days=1)
                logger.info("Daily email limit reset with precision.")
            await asyncio.sleep(60)  # Genius-level efficiency: check every minute

    async def generate_email_content(self, client_data):
        for client, model in self.clients_models:
            try:
                prompt = f"""
                {self.meta_prompt}
                Client Data: {json.dumps(client_data, indent=2)}
                Write a short, casual email that feels human. Bring up their interest in {client_data['interests'][0]} to kick things off or offer a hand. No sales pitches, no formal vibes—just a friendly note.
                """
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9  # Higher temperature for natural, varied output
                )
                content = response.choices[0].message.content.strip()
                logger.info(f"Generated email for {client_data['name']}: {content[:50]}...")
                return content
            except Exception as e:
                logger.warning(f"Failed to use {client.base_url} with model {model}: {e}")
        logger.error("All clients failed for generate_email_content")
        return f"Hey {client_data['name']}, saw you’re into {client_data['interests'][0]}. Wanna chat about it?"

    async def validate_email_content(self, content):
        """Use Gemini Pro to ensure the email is spam-proof, human-like, and avoids AI detection."""
        try:
            # Check for spam triggers and human-like tone
            prompt = f"""
            Analyze this email for spam triggers, robotic tone, and AI-generated patterns. 
            If it sounds too formal, salesy, or robotic, suggest a more casual, human-like version.
            Original Email: {content}
            """
            response = self.gemini_pro.generate_content(prompt)
            validation = response.text.lower()

            # Check for spam or robotic tone flags
            if "spam" in validation or "robotic" in validation or "formal" in validation:
                logger.warning(f"Validation flagged issues: {validation}")
                # Refine content if flagged
                refinement_prompt = f"""
                Rewrite this email to sound more casual and human-like, avoiding robotic or formal tone:
                {content}
                """
                refined_response = self.gemini_pro.generate_content(refinement_prompt)
                refined_content = refined_response.text.strip()
                logger.info("Content refined to sound more human-like.")
                return False, refined_content  # Return refined content
            else:
                logger.info("Email validated as spam-safe and human-like.")
                return True, content  # Original content passes
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, "Validation failed; using fallback content."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_email(self, recipient, subject, content):
        try:
            msg = EmailMessage()
            msg.set_content(content)
            msg['Subject'] = subject
            msg['From'] = self.config.get("HOSTINGER_EMAIL")
            msg['To'] = recipient
            with smtplib.SMTP(self.config.get("HOSTINGER_SMTP"), self.config.get("SMTP_PORT")) as server:
                server.starttls()
                server.login(self.config.get("HOSTINGER_EMAIL"), self.config.get("HOSTINGER_SMTP_PASS"))
                server.send_message(msg)
            logger.info(f"Email sent to {recipient}")
            return True
        except Exception as e:
            logger.error(f"Email send failed for {recipient}: {e}")
            raise

    async def store_email_log(self, recipient, subject, content, status):
        """Log emails in the database for analytics and continuous improvement."""
        # **Status: Already Correct** - No country or CRM-related changes needed; it’s secure and efficient.
        async with self.session_maker() as session:
            log = EmailLog(
                recipient=recipient,
                subject=subject,
                content=encrypt_data(content),  # Genius-level security
                status=status,
                timestamp=datetime.utcnow().replace(tzinfo=pytz.UTC)
            )
            session.add(log)
            await session.commit()
            logger.info(f"Email log secured for {recipient}")

    async def learn_from_spam_feedback(self, recipient, status):
        async with self.session_maker() as session:
            if status in ["bounced", "failed"]:
                await session.execute(
                    "UPDATE email_logs SET spam_flagged = TRUE WHERE recipient = :email AND timestamp > :threshold",
                    {"email": recipient, "threshold": datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(days=30)}
                )
                await session.commit()
                logger.info(f"Marked {recipient} email as potential spam trigger.")

    
    async def send_personalized_email(self, client_id):
        while self.active_emails >= self.get_allowed_concurrency():
            await asyncio.sleep(1)  # Throttle based on system load
        self.active_emails += 1
        try:
            if self.emails_sent_today >= self.max_emails_per_day:
                logger.warning("Daily email limit reached.")
                await self.orchestrator.report_status("EmailAgent", f"Limit hit: {self.emails_sent_today}/{self.max_emails_per_day}")
                return
            async with self.session_maker() as session:
                client = await session.get(Client, client_id)
                if not client or client.country != "USA":
                    logger.info(f"Skipping {client_id}: Not USA or missing.")
                    return
                client_data = {
                    "name": client.name,
                    "email": client.email,
                    "interests": client.interests,
                    "last_interaction": client.last_interaction.isoformat() + "Z",
                    "response_rate": float(client.response_rate or 0.0),
                    "engagement_score": float(client.engagement_score or 0.0),
                    "opt_in": bool(client.opt_in)
                }
                if not client_data["opt_in"]:
                    logger.warning(f"Client {client.name} not opted in.")
                    return
                compliance_context = (
                    f"Email for {client.name} (ID: {client_id}, USA). "
                    f"Interests: {', '.join(client_data['interests'])}. "
                    "Moroccan law: opt-in required, non-spammy."
                )
                compliance_check = await self.think_tool.reflect_on_action(
                    context=compliance_context,
                    agent_name="EmailAgent",
                    task_description="Ensure Moroccan law compliance"
                )
                compliance_data = json.loads(compliance_check)
                if not compliance_data.get("proceed", False):
                    logger.warning(f"Compliance failed: {compliance_data.get('reason', 'Unknown')}")
                    return
                content = await self.generate_email_content(client_data)
                if not content:
                    content = f"Hey {client_data['name']}, saw you’re into {client_data['interests'][0]}. Wanna chat about it?"
                is_valid, validated_content = await self.validate_email_content(content)
                if not is_valid:
                    logger.info("Using refined content to ensure human-like tone.")
                    content = validated_content  # Use refined content if validation fails
                else:
                    content = validated_content  # Original content if valid
                await self.optimal_send_time(client_id)
                subject = f"Hey {client.name}, Let’s Talk {client_data['interests'][0]}!"
                success = await self.send_email(client_data["email"], subject, content)
                status = "sent" if success else "failed"
                await self.store_email_log(client_data["email"], subject, content, status)
                if success:
                    self.emails_sent_today += 1
                    logger.info(f"Email sent to {client.name}. Total: {self.emails_sent_today}")
                    await self.orchestrator.report_status("EmailAgent", f"Sent to {client_id}")
        except Exception as e:
            logger.error(f"Failed for {client_id}: {e}")
            await self.orchestrator.report_error("EmailAgent", str(e))
        finally:
            self.active_emails -= 1


    async def update_engagement(self):
        async with self.session_maker() as session:
            clients = await session.execute("SELECT id, email, engagement_score FROM clients")
            for client in clients.fetchall():
                # Calculate open rate and response rate from the last 30 days
                open_rate = await session.execute(
                    "SELECT COUNT(*) FILTER (WHERE opened_at IS NOT NULL) / COUNT(*)::float FROM email_logs WHERE recipient = :email AND timestamp > :threshold",
                    {"email": client.email, "threshold": datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(days=30)}
                )
                response_rate = await session.execute(
                    "SELECT COUNT(*) FILTER (WHERE status = 'responded') / COUNT(*)::float FROM email_logs WHERE recipient = :email AND timestamp > :threshold",
                    {"email": client.email, "threshold": datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(days=30)}
                )
                open_rate = open_rate.scalar() or 0.0
                response_rate = response_rate.scalar() or 0.0
                # Weighted engagement score: 60% open rate, 40% response rate
                new_score = (open_rate * 0.6 + response_rate * 0.4) if client.engagement_score else (open_rate + response_rate) / 2
                await session.execute(
                    "UPDATE clients SET engagement_score = :score WHERE id = :id",
                    {"score": min(new_score, 1.0), "id": client.id}
                )
            await session.commit()
            logger.info("Engagement scores updated based on opens and responses.")

    async def run(self):
        asyncio.create_task(self.reset_daily_limit())
        asyncio.create_task(self.update_engagement_periodically())  # General engagement updates
        while True:
            task = await self.task_queue.get()
            asyncio.create_task(self.process_task(task))

    async def process_task(self, task):
        client_id = task['client_id']
        try:
            await self.send_personalized_email(client_id)
        except Exception as e:
            logger.error(f"Task failed for {client_id}: {e}")
            await self.orchestrator.report_error("EmailAgent", str(e))
        finally:
            self.task_queue.task_done()

    async def update_engagement_periodically(self):
        while True:
            await self.update_engagement()
            await asyncio.sleep(86400)  # Daily update

    async def process_task(self, task):
        client_id = task['client_id']
        try:
            await self.send_personalized_email(client_id)
        except Exception as e:
            logger.error(f"Task failed for {client_id}: {e}")
            await self.orchestrator.report_error("EmailAgent", str(e))
        finally:
            self.task_queue.task_done()

    async def request_email_send(self, client_id):
        """Queue an email send with strategic intent and genius-level foresight."""
        # Open a database session to fetch client data
        async with self.session_maker() as session:
            client = await session.get(Client, client_id)
            # Skip if client doesn’t exist or isn’t in the USA
            if not client or client.country != "USA":
                logger.info(f"Skipping {client_id}: Not in USA or missing data.")
                return
            # Assign priority based on engagement score (default to 1.0 if missing)
            priority = client.engagement_score if hasattr(client, 'engagement_score') else 1.0
            # Add task to the priority queue
            await self.task_queue.put({'client_id': client_id, 'priority': priority})
            logger.info(f"Email send queued for {client.name} with priority {priority}")

    async def queue_osint_leads(self, leads):
        for lead in leads:
            async with self.session_maker() as session:
                client = Client(email=lead['email'], name=lead.get('name', 'Customer'), country="USA", opt_in=True)
                session.add(client)
                await session.commit()
                await self.request_email_send(client.id)

    async def request_bulk_campaign(self, client_ids):
        """Queue a bulk campaign with strategic intent and adaptive throttling."""
        usa_client_ids = []
        # Filter for USA clients only
        async with self.session_maker() as session:
            for client_id in client_ids:
                client = await session.get(Client, client_id)
                if client and client.country == "USA":
                    usa_client_ids.append(client_id)
                else:
                    logger.info(f"Skipping {client_id}: Not in USA or missing data.")

        # Genius feature: Dynamically adjust batch size based on server load
        current_load = await self.get_server_load()  # Fetch real-time system metric
        batch_size = min(50, max(10, int(1000 / (current_load + 1))))  # Adaptive batching
        for i in range(0, len(usa_client_ids), batch_size):
            batch = usa_client_ids[i:i + batch_size]
            # Launch batch in parallel
            asyncio.create_task(self.run_bulk_campaign(batch))
            # Random delay to mimic human pacing and avoid spam flags
            await asyncio.sleep(random.uniform(0.5, 2.0))
        logger.info(f"Bulk campaign queued for {len(usa_client_ids)} USA clients in batches of {batch_size}")

    async def run_bulk_campaign(self, client_ids):
        import psutil
        tasks = []
        cpu_load = psutil.cpu_percent(interval=1)
        max_concurrent = max(5, min(20, int(100 / (cpu_load + 1))))
        semaphore = asyncio.Semaphore(max_concurrent)
        for client_id in client_ids:
            if self.emails_sent_today >= self.max_emails_per_day:
                logger.warning(f"Daily limit {self.max_emails_per_day} reached.")
                break
            tasks.append(self.send_personalized_email(client_id))
        async def bounded_task(task):
            async with semaphore:
                return await task
        await asyncio.gather(*(bounded_task(task) for task in tasks))
        logger.info(f"Bulk campaign completed for {len(tasks)} clients with {max_concurrent} concurrency.")

    async def optimal_send_time(self, client_id):
        async with self.session_maker() as session:
            client = await session.get(Client, client_id)
            if not client:
                logger.error(f"Client {client_id} not found.")
                return
            client_tz = pytz.timezone(client.timezone) if client.timezone else pytz.timezone("US/Eastern")
            logs = await session.execute(
                "SELECT opened_at FROM email_logs WHERE recipient = :email AND opened_at IS NOT NULL",
                {"email": client.email}
            )
            open_times = [log.opened_at for log in logs.fetchall()]
            if open_times:
                open_hours = [t.astimezone(client_tz).hour for t in open_times]
                avg_hour = int(sum(open_hours) / len(open_hours))
                send_time = datetime.now(client_tz).replace(hour=avg_hour, minute=0, second=0)
                if send_time < datetime.now(client_tz):
                    send_time += timedelta(days=1)
            else:
                send_time = datetime.now(client_tz).replace(hour=10, minute=0, second=0)
                if send_time < datetime.now(client_tz):
                    send_time += timedelta(days=1)
            send_time_utc = send_time.astimezone(pytz.UTC)
            delay = (send_time_utc - datetime.now(pytz.UTC)).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
            logger.info(f"Optimal send time for {client.name}: {send_time_utc}")



async def get_insights(self):
    async with self.session_maker() as session:
        response_rate = await session.execute(
            "SELECT COUNT(*) FILTER (WHERE status = 'responded') / COUNT(*)::float FROM email_logs WHERE timestamp > :threshold",
            {"threshold": datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(hours=24)}
        )
        return {"response_rate": response_rate.scalar() or 0.0}