import asyncio
import logging
import json
from openai import AsyncOpenAI as AsyncDeepSeekClient
import google.generativeai as genai

# Configure logging
logger = logging.getLogger(__name__)

class ThinkTool:
    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.clients_models = clients_models  # List of (client, model) tuples
        # Initialize Gemini Pro for multimodal tasks
        try:
            self.gemini_pro = genai.GenerativeModel("gemini-2.0-pro", api_key=os.getenv("GEMINI_API_KEY"))
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Pro: {e}")
            raise
        self.meta_prompt = """
        You are a genius-level AI designed to enhance the decision-making and reliability of other agents in an AI agency targeting rapid growth and extreme profitability ($6,000 in 24 hours, $100M in 9 months). Your role is to reflect on tasks, validate outputs, and solve challenges efficiently. Prioritize accuracy, cost-saving, and high-ROI actions. Adapt to dynamic situations, minimize risks, and ensure compliance with agency rules. Be resourceful, decisive, and transparent in your reasoning.
        """

    async def reflect_on_action(self, context, agent_name, task_description):
        for client, model in self.clients_models:
            try:
                prompt = f"""
                {self.meta_prompt}
                Agent: {agent_name}
                Task: {task_description}
                Context: {context}
                Reflect on the current state. Assess data completeness, compliance, and risks.
                Respond in JSON with:
                - 'proceed' (true/false): Whether to continue the task.
                - 'reason' (text): Why this decision was made.
                - 'next_step' (text): Recommended action.
                """
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                reflection = json.loads(response.choices[0].message.content)
                logger.info(f"Reflection for {agent_name}: {reflection}")
                return reflection
            except Exception as e:
                logger.warning(f"Failed to use {client.base_url} with model {model}: {e}")
        logger.error("All clients failed for reflect_on_action")
        return {
            "proceed": True,
            "reason": "All API clients failed; proceeding with default action",
            "next_step": "Continue with caution"
        }

    async def validate_output(self, output, expected_format, agent_name):
        """Validate if an agent's output meets the expected criteria."""
        for client, model in self.clients_models:
            try:
                prompt = f"""
                {self.meta_prompt}
                Agent: {agent_name}
                Output: {output}
                Expected Format: {expected_format}

                Check if the output is valid and usable.
                Respond in JSON with:
                - 'valid' (true/false): Whether the output meets criteria.
                - 'feedback' (text): Explanation of the validation result.
                """
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                validation = json.loads(response.choices[0].message.content)
                logger.info(f"Validation for {agent_name}: {validation}")
                return validation
            except Exception as e:
                logger.warning(f"Failed to use {client.base_url} with model {model}: {e}")
        logger.error("All clients failed for validate_output")
        return {
            "valid": False,
            "feedback": "Validation failed due to all API clients failing"
        }

    async def analyze_visual(self, image_data, task_description):
        """Analyze visual data (e.g., screenshots) using Gemini Pro.
        Note: This method uses self.gemini_pro directly due to multimodal requirements,
        as self.clients_models may not support image inputs natively.
        """
        try:
            prompt = f"{self.meta_prompt}\nAnalyze this image for: {task_description}"
            response = self.gemini_pro.generate_content([prompt, image_data])
            analysis = response.text
            logger.info(f"Visual analysis result: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return "Failed to analyze image due to an error"

    async def handle_quick_challenge(self, challenge_type, context):
        """Solve quick challenges (e.g., CAPTCHAs) using self.clients_models."""
        prompt = f"{self.meta_prompt}\nQuick challenge: {challenge_type}. Context: {context}"
        for client, model in self.clients_models:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                solution = response.choices[0].message.content.strip()
                logger.info(f"Quick challenge solution: {solution}")
                return solution
            except Exception as e:
                logger.warning(f"Failed to use {client.base_url} with model {model}: {e}")
        logger.error("All clients failed for handle_quick_challenge")
        return "Failed to solve challenge due to all API clients failing"
    
    async def extract_api_key(self, page_content):
        """Extract API key from page content using available clients."""
        prompt = f"""
        {self.meta_prompt}
        Given this HTML content: {page_content[:2000]}...
        Extract a 32+ character alphanumeric API key. Return only the key as a string or the word 'None' if not found.
        """
        for client, mdl in self.clients_models:
            try:
                # Assuming client is OpenAI/DeepSeek compatible for chat.completions
                if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                    response = await client.chat.completions.create(
                        model=mdl,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=60 # Increased slightly for potential key length + 'None'
                    )
                    key = response.choices[0].message.content.strip()
                    # Check if the key seems valid based on length and alphanumeric characters
                    # Allow for minor variations, focus on length primarily
                    if key and len(key) >= 32 and all(c.isalnum() or c in '-_' for c in key): # Allow alphanumeric plus common key chars
                        logger.info(f"Extracted potential API key using {client.base_url} model {mdl}: {key[:10]}...")
                        return key
                    else:
                        # Log if the response was received but wasn't a valid key format
                        logger.warning(f"Client {client.base_url} model {mdl} returned invalid key format or 'None': {key}")
                        continue # Try the next client if format is invalid
                else:
                    logger.warning(f"Client {type(client)} does not support chat.completions.create, skipping for extract_api_key.")
                    continue # Try next client
            except Exception as e:
                logger.warning(f"Failed to use client {type(client)} ({getattr(client, 'base_url', 'N/A')}) with model {mdl} for extract_api_key: {e}")
                # Continue to the next client if this one fails
        
        # If the loop completes without returning a valid key
        logger.error("All clients failed to extract a valid API key.")
        return None

    async def run(self):
        logger.info("ThinkTool monitoring system.")
        while True:
            try:
                async with self.session_maker() as session:
                    metrics = await session.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 100")
                    metrics_data = [dict(row) for row in metrics.fetchall()]
                    prompt = f"{self.meta_prompt}\nAnalyze metrics: {json.dumps(metrics_data, indent=2)}\nSuggest optimizations."
                    for client, model in self.clients_models:
                        try:
                            response = await client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=200
                            )
                            suggestions = response.choices[0].message.content.strip()
                            await self.orchestrator.send_notification("Optimization Suggestions", suggestions)
                            break
                        except Exception as e:
                            logger.warning(f"Failed to use {client.base_url} with model {model}: {e}")
                    else:
                        logger.error("All clients failed for run method")
                        # Optionally, handle the failure (e.g., retry later or notify)
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Monitoring failed: {e}")
                await asyncio.sleep(60)