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
            "USA": [
                "https://www.federalregister.gov",
                "https://uscode.house.gov",
                "https://www.consumerfinance.gov"
            ],
            "Morocco": [
                "https://www.sgg.gov.ma",
                "https://www.bkam.ma",
                "https://www.justice.gov.ma"
            ]
        }
        self.cache_db = sqlite3.connect("legal_cache.db", check_same_thread=False)
        self.create_cache_tables()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.update_interval = 604800  # 1 week in seconds
        self.token_threshold = 500  # Genius-level token limit per call
        self.gray_area_strategies = {}

    def update_contract_date(self, contract_template):
        today = datetime.now().strftime("%Y-%m-%d")  # Makes date like "2025-04-07"
        updated_contract = contract_template.replace("[CONTRACT_DATE]", today)
        return updated_contract

    async def get_w8_data(self):
        w8_data_str = await self.secure_storage.get_secret("w8ben_data")
        w8_data = json.loads(w8_data_str)
        return w8_data

    async def validate_w8_data(self):
        w8_data_str = await self.secure_storage.get_secret("w8ben_data")
        w8_data = json.loads(w8_data_str)
        required_fields = ["name", "country", "address", "tin"]
        if not all(field in w8_data for field in required_fields):
            logger.error("W-8BEN data incomplete")
            return False
        return True

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
        """Interpret legal updates and generate dynamic gray area strategies."""
        interpretations = {}
        batch_texts = []
        batch_mapping = {}

        for country, texts in updates.items():
            for text in texts:
                cached = self.get_cached_interpretation(text)
                if cached:
                    interpretations[text] = cached
                    self.gray_area_strategies[country] = cached.get('gray_areas', [])
                    continue
                batch_texts.append(text)
                batch_mapping[text] = country

        if batch_texts:
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
            clients = await self.orchestrator.get_available_openrouter_clients()
            for client in clients:
                try:
                    response = await client.chat.completions.create(
                        model="google/gemini-2.5-pro-exp-03-25:free",
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}
                    )
                    results = json.loads(response.choices[0].message.content)
                    for text, result in zip(batch_texts, results):
                        interpretations[text] = result
                        self.gray_area_strategies[batch_mapping[text]] = result.get('gray_areas', [])
                        self.cache_interpretation(text, result, batch_mapping[text])
                    break
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        async with self.session_maker() as session:
                            await session.execute(
                                "UPDATE accounts SET is_available = FALSE WHERE api_key = :api_key",
                                {"api_key": client.api_key}
                            )
                            await session.commit()
                        continue
                    logger.warning(f"Failed to use client {client.api_key}: {e}")
            else:
                logger.error("All clients failed for interpret_updates")
                return {"error": "All API clients exhausted"}
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

    # FILE: agents/legal_compliance_agent.py
# Modify the validate_operation() method

    async def validate_operation(self, operation_description: str) -> dict:
        """Validate operations, providing quantified risk assessment."""
        cursor = self.cache_db.cursor()
        # Fetch more interpretations for better context
        cursor.execute("SELECT interpretation, gray_areas, compliance_note FROM interpretations ORDER BY update_id DESC LIMIT 20")
        recent_interpretations = [
            {"interp": row[0], "gray_areas": json.loads(row[1] or '[]'), "note": row[2]}
            for row in cursor.fetchall()
        ]

        prompt = f"""
        {self.meta_prompt}
        **Operation to Validate:** {operation_description}

        **Recent Legal Interpretations & Grey Areas:**
        {json.dumps(recent_interpretations, indent=2)}

        **Task:** Analyze the operation against the interpretations. Focus on USA/Moroccan law implications and agency risk tolerance (assume high tolerance for calculated risks aiming for profit, but zero tolerance for actions with high probability of severe legal/financial penalties).

        **Output (JSON):**
        - 'is_compliant' (bool): Strict compliance assessment (True/False).
        - 'risk_score' (float): Estimated risk score (0.0 = No Risk, 1.0 = Critical Risk). Factor in probability and severity of potential negative outcomes (legal action, fines, account bans, reputational damage).
        - 'compliance_issues' (list): Specific laws/regulations potentially violated.
        - 'gray_area_exploitation' (bool): Does this operation leverage an identified grey area?
        - 'potential_consequences' (list): Brief description of potential negative outcomes if non-compliant or grey area is challenged.
        - 'recommendations' (list): Concrete actions to mitigate risk or ensure compliance *while still attempting to achieve the operation's goal if possible*.
        - 'proceed_recommendation' (str): Explicit recommendation ('Proceed', 'Proceed with Caution', 'Halt - High Risk', 'Halt - Non-Compliant').
        """
        # Use the ThinkTool's robust _call_llm method
        validation_json = await self.orchestrator.agents['think']._call_llm(
            prompt, model_key='legal_validate', temperature=0.2, max_tokens=800, is_json_output=True
        )

        if validation_json:
            try:
                result = json.loads(validation_json)
                # Add default values for robustness
                result.setdefault('is_compliant', False)
                result.setdefault('risk_score', 1.0) # Default high risk on failure
                result.setdefault('compliance_issues', ['Analysis Failed'])
                result.setdefault('gray_area_exploitation', False)
                result.setdefault('potential_consequences', [])
                result.setdefault('recommendations', [])
                result.setdefault('proceed_recommendation', 'Halt - Analysis Failed')

                logger.info(f"Legal Validation for '{operation_description[:50]}...': Compliant={result['is_compliant']}, Risk Score={result['risk_score']:.2f}, Recommendation={result['proceed_recommendation']}")

                # Trigger notifications/errors based on severity
                if not result['is_compliant'] or result['risk_score'] > 0.8:
                    await self.orchestrator.report_error(
                        "LegalComplianceAgent",
                        f"Operation blocked/high-risk: {operation_description[:100]}... Issues: {result['compliance_issues']}, Risk: {result['risk_score']:.2f}"
                    )
                elif result['risk_score'] > 0.5 or result.get('recommendations'):
                     await self.orchestrator.send_notification(
                         "LegalCompliance Advisory",
                         f"Operation: {operation_description[:100]}... Risk: {result['risk_score']:.2f}. Recommendations: {result['recommendations']}"
                     )
                return result
            except json.JSONDecodeError:
                 logger.error(f"LegalAgent: Failed to decode JSON validation: {validation_json}")
        # Fallback
        return {
            'is_compliant': False, 'risk_score': 1.0, 'compliance_issues': ['LLM Error/Invalid JSON'],
            'gray_area_exploitation': False, 'potential_consequences': [], 'recommendations': [],
            'proceed_recommendation': 'Halt - Analysis Failed'
        }

    async def get_invoice_legal_note(self, client_country):
        cursor = self.cache_db.cursor()
        cursor.execute("""
            SELECT compliance_note FROM interpretations
            WHERE category IN ('financial', 'tax')
            AND country = ?
            ORDER BY update_id DESC LIMIT 1
        """, (client_country,))
        row = cursor.fetchone()
        base_note = row[0] if row else "Compliant with latest financial regulations"

        bulletproof_terms = (
            "This agreement is irrevocable and binding upon acceptance. "
            "UGC Genius shall not be liable for any indirect, incidental, or consequential damages, including but not limited to loss of profits or data. "
            "All disputes shall be resolved exclusively under Moroccan law in the courts of [Your City], Morocco. "
            f"Payment to {self.orchestrator.user_name}'s account in Morocco is non-refundable and non-cancellable once services commence. "
            "Client acknowledges that UGC Genius adheres to ISO 20022 standards for secure international transactions."
        )
        return f"{base_note} | {bulletproof_terms}"


        

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