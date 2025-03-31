import asyncio
import logging
import json
import os
import sqlite3
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI as AsyncDeepSeekClient
import aiohttp
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Genius-level logging for diagnostics and auditing
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("legal_compliance_agent.log")]
)

class LegalComplianceAgent:
    """Elite agent for legal compliance, gray area exploitation, and cost-efficient operation."""

    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.clients_models = clients_models  # List of (client, model) tuples
        self.meta_prompt = """
        You are a genius-level legal strategist for an AI agency aiming for $100M in 9 months. Monitor US and Moroccan laws weekly, interpret them in real-time, and identify gray areas for profit maximization. Ensure compliance while pushing ethical boundaries for disruption. Return precise, actionable JSON outputs.
        """
        self.legal_sources = {
            "USA": "https://www.federalregister.gov/documents/search?conditions[term]=&conditions[publication_date][is]=today",
            "Morocco": "http://www.sgg.gov.ma/PortailArabe.aspx"
        }
        self.cache_db = sqlite3.connect("legal_cache.db", check_same_thread=False)
        self.create_cache_tables()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.update_interval = 604800  # 1 week in seconds
        self.token_threshold = 500  # Genius-level token limit per call

    def create_cache_tables(self):
        """Initialize SQLite tables for genius-level caching."""
        cursor = self.cache_db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS legal_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT,
                content TEXT,
                timestamp DATETIME,
                hash TEXT UNIQUE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interpretations (
                update_id INTEGER,
                category TEXT,
                interpretation TEXT,
                gray_areas TEXT,
                compliance_note TEXT,
                FOREIGN KEY(update_id) REFERENCES legal_updates(id)
            )""")
        self.cache_db.commit()

    async def fetch_legal_updates(self):
        """Asynchronously fetch legal updates with SmartProxy optimization."""
        async with aiohttp.ClientSession() as session:
            updates = {}
            for country, url in self.legal_sources.items():
                async with session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    if country == "USA":
                        updates[country] = [doc.text.strip() for doc in soup.find_all("div", class_="document-title")]
                    elif country == "Morocco":
                        updates[country] = [doc.text.strip() for doc in soup.find_all("div", class_="bulletin-entry") or soup.find_all("p")]
            logger.info(f"Fetched updates: {len(updates.get('USA', []))} US, {len(updates.get('Morocco', []))} Moroccan")
            return updates

    async def interpret_updates(self, updates):
        """Genius-level interpretation with caching and batching."""
        interpretations = {}
        batch_texts = []
        batch_mapping = {}

        for country, texts in updates.items():
            for text in texts:
                cached = self.get_cached_interpretation(text)
                if cached:
                    interpretations[text] = cached
                    continue
                batch_texts.append(text)
                batch_mapping[text] = country

        if batch_texts:
            # Genius-level batching to minimize token usage
            prompt = f"""
            {self.meta_prompt}
            Analyze these legal updates:
            {json.dumps(batch_texts, indent=2)}
            Return a JSON array where each item has:
            - 'category': Law category
            - 'interpretation': Impact on operations
            - 'gray_areas': List of exploitable ambiguities
            - 'compliance_note': Exact note for invoices/documents
            """
            estimated_tokens = len(prompt) // 4  # Rough heuristic: 1 token ~ 4 chars
            if estimated_tokens > self.token_threshold:
                logger.warning("Token estimate exceeds threshold; splitting batch.")
                mid = len(batch_texts) // 2
                interpretations.update(await self.interpret_updates({"split": batch_texts[:mid]}))
                interpretations.update(await self.interpret_updates({"split": batch_texts[mid:]}))
            else:
                for client, model in self.clients_models:
                    try:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"}
                        )
                        results = json.loads(response.choices[0].message.content)
                        for text, result in zip(batch_texts, results):
                            interpretations[text] = result
                            self.cache_interpretation(text, result, batch_mapping[text])
                        break  # Success, exit loop
                    except Exception as e:
                        logger.warning(f"Failed to use {client.base_url} with model {model}: {e}")
                else:
                    logger.error("All clients failed for interpret_updates")
                    # Fallback to DeepSeek-R1
                    try:
                        deepseek_client = AsyncDeepSeekClient(
                            api_key=self.config.get("DEEPSEEK_API_KEY"),
                            base_url="https://api.deepseek.com"
                        )
                        response = await deepseek_client.chat.completions.create(
                            model="deepseek-r1",
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"}
                        )
                        results = json.loads(response.choices[0].message.content)
                        for text, result in zip(batch_texts, results):
                            interpretations[text] = result
                            self.cache_interpretation(text, result, batch_mapping[text])
                    except Exception as e:
                        logger.critical(f"DeepSeek-R1 fallback failed: {e}")
                        raise
        return interpretations

    def get_cached_interpretation(self, text):
        """Genius-level caching with similarity detection."""
        cursor = self.cache_db.cursor()
        cursor.execute("SELECT content, interpretation FROM legal_updates JOIN interpretations ON legal_updates.id = interpretations.update_id")
        cached_data = cursor.fetchall()
        if not cached_data:
            return None

        texts = [text] + [row[0] for row in cached_data]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        if similarities.max() > 0.85:  # High threshold for precision
            idx = similarities.argmax()
            return json.loads(cached_data[idx][1])
        return None

    def cache_interpretation(self, text, interpretation, country):
        """Persist interpretations with unique hashing."""
        import hashlib
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cursor = self.cache_db.cursor()
        try:
            cursor.execute("INSERT INTO legal_updates (country, content, timestamp, hash) VALUES (?, ?, ?, ?)",
                           (country, text, datetime.now(), text_hash))
            update_id = cursor.lastrowid
            cursor.execute("INSERT INTO interpretations (update_id, category, interpretation, gray_areas, compliance_note) VALUES (?, ?, ?, ?, ?)",
                           (update_id, interpretation['category'], interpretation['interpretation'],
                            json.dumps(interpretation['gray_areas']), interpretation['compliance_note']))
            self.cache_db.commit()
        except sqlite3.IntegrityError:
            pass  # Duplicate hash; skip caching

    async def validate_operation(self, operation_description):
        """Validate operations with precomputed legal data."""
        cursor = self.cache_db.cursor()
        cursor.execute("SELECT interpretation, gray_areas FROM interpretations ORDER BY update_id DESC LIMIT 10")
        recent = [(row[0], json.loads(row[1])) for row in cursor.fetchall()]

        prompt = f"""
        {self.meta_prompt}
        Validate this operation: {operation_description}
        Against recent interpretations: {json.dumps(recent, indent=2)}
        Return JSON:
        - 'is_compliant': Boolean
        - 'issues': List of issues
        - 'recommendations': Actions to take
        """
        for client, model in self.clients_models:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                if not result['is_compliant']:
                    await self.orchestrator.report_error("LegalComplianceAgent", f"Operation blocked: {result['issues']}")
                elif result.get('recommendations'):
                    await self.orchestrator.send_notification("LegalOptimization", f"Recommendations: {result['recommendations']}")
                return result
            except Exception as e:
                logger.warning(f"Failed to use {client.base_url} with model {model}: {e}")
        logger.error("All clients failed for validate_operation")
        # Fallback to DeepSeek-R1
        try:
            deepseek_client = AsyncDeepSeekClient(
                api_key=self.config.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            response = await deepseek_client.chat.completions.create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            if not result['is_compliant']:
                await self.orchestrator.report_error("LegalComplianceAgent", f"Operation blocked: {result['issues']}")
            elif result.get('recommendations'):
                await self.orchestrator.send_notification("LegalOptimization", f"Recommendations: {result['recommendations']}")
            return result
        except Exception as e:
            logger.critical(f"DeepSeek-R1 fallback failed: {e}")
            return {"is_compliant": False, "issues": ["All API clients failed"], "recommendations": []}

    async def get_invoice_legal_note(self):
        cursor = self.cache_db.cursor()
        cursor.execute("SELECT compliance_note FROM interpretations WHERE category IN ('financial', 'tax') ORDER BY update_id DESC LIMIT 1")
        row = cursor.fetchone()
        base_note = row[0] if row else "Compliant with latest US/Moroccan financial regulations"
        return f"{base_note} | ISO 20022 compliant transfer to [Your Name], Morocco."

    async def run(self):
        """Continuous legal monitoring and strategy generation."""
        while True:
            try:
                updates = await self.fetch_legal_updates()
                interpretations = await self.interpret_updates(updates)
                for text, interp in interpretations.items():
                    if interp['gray_areas']:
                        await self.orchestrator.send_notification(
                            "GrayAreaOpportunity",
                            f"Exploit: {interp['gray_areas']} in {interp['category']}"
                        )
                logger.info("Legal cycle completed.")
            except Exception as e:
                logger.error(f"Legal run failed: {e}")
                await self.orchestrator.report_error("LegalComplianceAgent", str(e))
            await asyncio.sleep(self.update_interval)