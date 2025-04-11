import asyncio
import numpy as np
import psutil
from stable_baselines3 import PPO
from gym.spaces import Box
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from models import Metric
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("agency.log"), logging.StreamHandler()]
)

class OptimizationAgent:
    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.clients_models = clients_models  # List of (client, model) tuples
        
        # Agents with concurrency control
        self.agents_with_concurrency = [
            agent for agent in orchestrator.agents.values() 
            if hasattr(agent, 'max_concurrency')
        ]
        self.num_agents = len(self.agents_with_concurrency)
        
        # Observation: [profit, avg_client_score, cpu, memory, tasks] + concurrencies
        observation_size = 5 + self.num_agents
        self.observation_space = Box(low=0, high=np.inf, shape=(observation_size,), dtype=np.float32)
        self.action_space = Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32)
        
        # PPO initialization
        self.rl_model = PPO(
            "MlpPolicy", 
            env=None,  # Custom training loop, no Gym env
            verbose=0, 
            n_steps=2048, 
            batch_size=64, 
            learning_rate=0.0003
        )
        
        # Experience buffer
        self.experience_buffer = []
        self.buffer_size = 1000
        self.batch_size = 64
        self.train_interval = 3600  # Hourly training
        logger.info(f"OptimizationAgent initialized for {self.num_agents} agents.")

    async def get_system_state(self):
        """Retrieve current system state."""
        async with self.session_maker() as session:
            profit = await self.orchestrator.agents['scoring'].calculate_total_profit(session)
            client_scores = await self.orchestrator.agents['scoring'].score_all_clients()
            avg_client_score = np.mean(list(client_scores.values())) if client_scores else 0.0
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            active_tasks = sum(
                agent.task_queue.qsize() for agent in self.orchestrator.agents.values()
                if hasattr(agent, 'task_queue')
            )
            concurrency_limits = [agent.max_concurrency for agent in self.agents_with_concurrency]
        state = [profit, avg_client_score, cpu, memory, active_tasks] + concurrency_limits
        return np.array(state, dtype=np.float32)

    async def apply_action(self, action):
        """Apply concurrency adjustments and track costs."""
        for i, agent in enumerate(self.agents_with_concurrency):
            new_concurrency = max(1, min(20, int(action[i] * 20)))
            old_concurrency = agent.max_concurrency
            agent.max_concurrency = new_concurrency
            # Calculate cost based on concurrency change
            cost_per_unit = 0.1  # $0.10 per concurrency unit
            cost_change = (new_concurrency - old_concurrency) * cost_per_unit
            if cost_change != 0:
                await self.orchestrator.agents['budget'].track_expense(
                    amount=abs(cost_change),
                    category="Concurrency",
                    description=f"Concurrency adjustment for {agent.__class__.__name__}: {old_concurrency} -> {new_concurrency}"
                )
        limits = {agent.__class__.__name__: agent.max_concurrency for agent in self.agents_with_concurrency}
        logger.info(f"Applied concurrency limits: {limits}")

    async def calculate_reward(self):
        """Compute reward based on profit and resource usage."""
        async with self.session_maker() as session:
            profit = await self.orchestrator.agents['scoring'].calculate_total_profit(session)
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        penalty = 0
        if cpu > 80:
            penalty += (cpu - 80) * 10
        if memory > 80:
            penalty += (memory - 80) * 10
        reward = profit - penalty
        logger.debug(f"Reward: {reward} (Profit: {profit}, Penalty: {penalty})")
        return reward

    async def train_model(self):
        """Train PPO model with collected experiences."""
        if len(self.experience_buffer) < self.batch_size:
            logger.info("Not enough experiences to train.")
            return
        
        observations = np.array([exp[0] for exp in self.experience_buffer[-self.batch_size:]], dtype=np.float32)
        actions = np.array([exp[1] for exp in self.experience_buffer[-self.batch_size:]], dtype=np.float32)
        rewards = np.array([exp[2] for exp in self.experience_buffer[-self.batch_size:]], dtype=np.float32)
        next_observations = np.array([exp[3] for exp in self.experience_buffer[-self.batch_size:]], dtype=np.float32)
        
        # Simulate one step of training
        for obs, act, rew, next_obs in zip(observations, actions, rewards, next_observations):
            self.rl_model.policy.observe(obs, act, rew, next_obs)
            self.rl_model.policy.update()
        logger.info(f"Trained model with {self.batch_size} experiences.")

    # FILE: agents/optimization_agent.py
# Modify the run() method

    async def run(self):
        """Run optimization loop with integrated training."""
        last_train_time = asyncio.get_event_loop().time()
        logger.info("OptimizationAgent run loop started.") # Added start log

        while True:
            if self.orchestrator.approved:
                try:
                    # --- State Observation ---
                    state = await self.get_system_state()
                    if state is None: # Add check if state fetching fails
                        logger.warning("OptimizationAgent: Failed to get system state, skipping cycle.")
                        await asyncio.sleep(60) # Wait before retrying
                        continue

                    # --- Action Prediction ---
                    # Use context manager for potential numpy issues? (Optional)
                    action, _ = self.rl_model.predict(state, deterministic=False) # Use non-deterministic for exploration

                    # --- Action Application ---
                    await self.apply_action(action)

                    # --- Environment Step & Reward ---
                    await asyncio.sleep(60)  # Allow time for action effects
                    next_state = await self.get_system_state()
                    if next_state is None:
                         logger.warning("OptimizationAgent: Failed to get next system state after action. Using previous state for buffer.")
                         next_state = state # Use previous state if next fails, less ideal but prevents crash

                    reward = await self.calculate_reward()

                    # --- Store Experience ---
                    self.experience_buffer.append((state, action, reward, next_state))
                    if len(self.experience_buffer) > self.buffer_size:
                        self.experience_buffer.pop(0) # Maintain buffer size

                    logger.info(f"Optimization Step: Reward={reward:.2f}, Action={np.round(action, 2)}")

                    # --- Training ---
                    current_time = asyncio.get_event_loop().time()
                    # Check if enough time passed AND buffer has sufficient samples
                    if (current_time - last_train_time >= self.train_interval) and (len(self.experience_buffer) >= self.batch_size):
                        logger.info(f"OptimizationAgent: Triggering model training. Buffer size: {len(self.experience_buffer)}")
                        await self.train_model() # *** CALL THE TRAINING METHOD ***
                        last_train_time = current_time
                        # Optional: Clear buffer after training or use more sophisticated replay
                        # self.experience_buffer = []

                except Exception as e:
                    logger.error(f"OptimizationAgent: Error in optimization cycle: {e}", exc_info=True)
                    await self.orchestrator.report_error("OptimizationAgent", str(e))
                    await asyncio.sleep(60) # Wait after error
            else:
                logger.info("OptimizationAgent: Awaiting orchestrator approval.")
                await asyncio.sleep(300) # Check less frequently if not approved

            # Main loop sleep - adjust interval as needed
            await asyncio.sleep(300) # Run optimization cycle every 5 minutes (adjust as needed)

    async def get_insights(self):
        """Provide concurrency insights for BudgetAgent and UI."""
        limits = {agent.__class__.__name__: agent.max_concurrency for agent in self.agents_with_concurrency}
        async with self.session_maker() as session:
            profit = await self.orchestrator.agents['scoring'].calculate_total_profit(session)
        return {"concurrency_limits": limits, "current_profit": profit}