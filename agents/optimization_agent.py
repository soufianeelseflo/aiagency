# Filename: agents/optimization_agent.py
# Description: Agent using Soft Actor-Critic (SAC) Reinforcement Learning
#              to dynamically optimize agent concurrency based on profit and resources.
# Version: 2.0 (Full Implementation)

import asyncio
import numpy as np
import psutil
import logging
import time
import os
from datetime import datetime, timezone
import sqlalchemy # For error handling
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import text as sql_text
from typing import Optional, Dict, Any, List, Tuple, cast # Added cast

# --- RL Imports ---
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:
    import gym # Fallback
    from gym.spaces import Box
    logging.warning("Gymnasium not found, falling back to Gym. Install Gymnasium for latest features.")

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    # from stable_baselines3.common.noise import NormalActionNoise # Not typically needed for SAC
    from stable_baselines3.common.callbacks import BaseCallback # For potential custom callbacks
    from stable_baselines3.common.logger import configure as sb3_configure_logger
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
    from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file # For saving/loading
except ImportError:
    logging.critical("Stable-Baselines3 library not found. Please install: pip install stable-baselines3[extra]")
    raise ImportError("Stable-Baselines3 is required for OptimizationAgent")

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase, KBInterface # Use relative import
except ImportError:
    from base_agent import GeniusAgentBase, KBInterface # Fallback

from models import Metric # Assuming Metric model is used for logging RL metrics
# Assuming Orchestrator provides access to other agents and config
# from agents.scoring_agent import ScoringAgent
# from agents.budget_agent import BudgetAgent

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Configure dedicated operational logger (assuming setup elsewhere)
op_logger = logging.getLogger('OperationalLog')
if not op_logger.hasHandlers(): # Add basic handler if none configured
    op_handler = logging.StreamHandler()
    op_formatter = logging.Formatter('%(asctime)s - OP_LOG - %(levelname)s - %(message)s')
    op_handler.setFormatter(op_formatter)
    op_logger.addHandler(op_handler)
    op_logger.setLevel(logging.INFO)
    op_logger.propagate = False

class OptimizationAgent(GeniusAgentBase):
    """
    OptimizationAgent v2.0 (SAC): Uses Soft Actor-Critic (SAC) RL for dynamic
    concurrency optimization based on profit and resource utilization.
    """
    AGENT_NAME = "OptimizationAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any):
        """Initializes the OptimizationAgent."""
        config = getattr(orchestrator, 'config', None)
        kb_interface = getattr(orchestrator, 'kb_interface', None)
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface, orchestrator=orchestrator, config=config)

        self.session_maker = session_maker # Needed for ScoringAgent calls

        # Identify agents to control (dynamically at runtime is safer)
        self._update_controllable_agents() # Initial identification
        if self.num_agents == 0:
            self.logger.critical(f"{self.AGENT_NAME}: Found no other agents with 'max_concurrency' attribute to optimize. Agent will be inactive.")
            # Agent cannot function without agents to control
            self.status = "error"
            # Raise error or handle gracefully depending on system design
            # For now, allow init but agent will fail in run loop if num_agents is 0
            # raise ValueError("OptimizationAgent requires controllable agents.")

        # --- Define Observation and Action Spaces ---
        self._setup_spaces()

        # --- SAC Model Initialization ---
        buffer_size = int(self.config.get('OPTIMIZATION_BUFFER_SIZE', 100000))
        self.learning_starts = int(self.config.get('OPTIMIZATION_LEARNING_STARTS', 1000))
        self.batch_size = int(self.config.get('OPTIMIZATION_BATCH_SIZE', 256))
        tau = float(self.config.get('OPTIMIZATION_TAU', 0.005))
        gamma = float(self.config.get('OPTIMIZATION_GAMMA', 0.99))
        learning_rate = float(self.config.get('OPTIMIZATION_LR', 3e-4))
        raw_train_freq = self.config.get('OPTIMIZATION_TRAIN_FREQ', (1, "step"))
        if isinstance(raw_train_freq, int): self.train_freq = TrainFreq(raw_train_freq, TrainFrequencyUnit.STEP)
        elif isinstance(raw_train_freq, (list, tuple)) and len(raw_train_freq) == 2: self.train_freq = TrainFreq(int(raw_train_freq[0]), TrainFrequencyUnit(raw_train_freq[1]))
        else: self.train_freq = TrainFreq(1, TrainFrequencyUnit.STEP); self.logger.warning("Invalid OPTIMIZATION_TRAIN_FREQ format, defaulting to (1, 'step')")

        self.gradient_steps = int(self.config.get('OPTIMIZATION_GRADIENT_STEPS', self.train_freq.frequency)) # Default to train_freq frequency

        policy_kwargs = dict(net_arch=[256, 256]) # Standard SAC architecture

        # Initialize SAC model - env=None as we use a custom loop
        self.rl_model = SAC(
            "MlpPolicy", env=None, observation_space=self.observation_space, action_space=self.action_space,
            policy_kwargs=policy_kwargs, verbose=0, buffer_size=buffer_size, learning_starts=self.learning_starts,
            batch_size=self.batch_size, tau=tau, gamma=gamma, train_freq=self.train_freq,
            gradient_steps=self.gradient_steps, action_noise=None, learning_rate=learning_rate,
            replay_buffer_class=ReplayBuffer, seed=self.config.get('RANDOM_SEED', None), device='auto'
        )
        self.replay_buffer = self.rl_model.replay_buffer # Reference to internal buffer

        # --- State Tracking ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['total_steps'] = 0
        self.internal_state['last_log_time'] = time.time()
        self.internal_state['last_save_time'] = time.time()
        self.internal_state['model_save_path'] = self.config.get('OPTIMIZATION_MODEL_SAVE_PATH', "rl_models/optimization_agent") # Specific path
        self.internal_state['model_load_path'] = self.config.get('OPTIMIZATION_MODEL_LOAD_PATH', None) # Path to load from

        # --- Load Pre-trained Model ---
        if self.internal_state['model_load_path'] and os.path.exists(f"{self.internal_state['model_load_path']}.zip"):
            try:
                self.logger.info(f"Loading pre-trained SAC model from: {self.internal_state['model_load_path']}.zip")
                # Ensure custom_objects are handled if needed
                self.rl_model = SAC.load(
                    self.internal_state['model_load_path'],
                    env=None, # Still no env needed for custom loop
                    device='auto',
                    # Pass other necessary parameters if they changed from defaults used during saving
                    buffer_size=buffer_size, learning_starts=self.learning_starts, batch_size=self.batch_size,
                    tau=tau, gamma=gamma, train_freq=self.train_freq, gradient_steps=self.gradient_steps,
                    learning_rate=learning_rate, seed=self.config.get('RANDOM_SEED', None)
                )
                self.replay_buffer = self.rl_model.replay_buffer # Update buffer reference
                # Optionally load replay buffer if saved separately
                if os.path.exists(f"{self.internal_state['model_load_path']}_replay_buffer.pkl"):
                    self.logger.info("Loading replay buffer...")
                    self.rl_model.load_replay_buffer(f"{self.internal_state['model_load_path']}_replay_buffer.pkl")
                self.logger.info("Pre-trained model and buffer loaded successfully.")
            except Exception as load_err:
                self.logger.error(f"Failed to load pre-trained model from {self.internal_state['model_load_path']}: {load_err}", exc_info=True)
                # Continue with the newly initialized model

        self.logger.info(f"{self.AGENT_NAME} v2.0 (SAC) initialized for {self.num_agents} agents. Buffer: {buffer_size}, Starts: {self.learning_starts}, Batch: {self.batch_size}, Train Freq: {self.train_freq}")

    def _update_controllable_agents(self):
        """Updates the list of agents this optimizer can control."""
        self.agents_with_concurrency = [
            agent for agent_name, agent in self.orchestrator.agents.items()
            if hasattr(agent, 'max_concurrency') and agent != self and agent_name != self.AGENT_NAME # Ensure agent exists and has attribute
        ]
        self.num_agents = len(self.agents_with_concurrency)
        self.logger.debug(f"Updated controllable agents list. Count: {self.num_agents}. Agents: {[a.AGENT_NAME for a in self.agents_with_concurrency]}")

    def _setup_spaces(self):
        """Sets up the observation and action spaces based on controllable agents."""
        if self.num_agents == 0:
            # Define dummy spaces if no agents are controllable, prevents errors but agent is inactive
            self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.logger.warning("Setting up dummy spaces as no controllable agents found.")
            return

        # Observation: [profit, avg_client_score, cpu, memory, total_tasks_in_queues] + [agent1_concurrency, agent2_concurrency, ...]
        low_obs = np.array([0, 0, 0, 0, 0] + [1] * self.num_agents, dtype=np.float32) # Min concurrency 1
        max_profit_obs = float(self.config.get('OPTIMIZATION_MAX_PROFIT_OBS', 1e6))
        max_score_obs = float(self.config.get('OPTIMIZATION_MAX_SCORE_OBS', 1000))
        max_tasks_obs = float(self.config.get('OPTIMIZATION_MAX_TASKS_OBS', 10000))
        self.max_concurrency_limit = int(self.config.get('OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT', 50))
        high_obs = np.array([max_profit_obs, max_score_obs, 100, 100, max_tasks_obs] + [self.max_concurrency_limit] * self.num_agents, dtype=np.float32)
        observation_size = 5 + self.num_agents
        self.observation_space = Box(low=low_obs, high=high_obs, shape=(observation_size,), dtype=np.float32)

        # Action: Scaled concurrency factor (0 to 1) for each agent
        self.action_space = Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32)
        self.logger.info(f"Observation space size: {observation_size}, Action space size: {self.num_agents}")

    async def get_system_state(self) -> Optional[np.ndarray]:
        """Retrieve current system state for the RL observation."""
        # --- Structured Thinking Step ---
        await self._internal_think("Getting system state: Fetching profit, scores, resources, tasks, and current concurrencies.")
        # --- End Structured Thinking Step ---
        try:
            # Refresh controllable agents list in case agents were added/removed
            self._update_controllable_agents()
            if self.num_agents == 0:
                 self.logger.warning("Cannot get system state: No controllable agents.")
                 return None # Return None if no agents to control

            # Re-setup spaces if number of agents changed
            # This is complex as it requires re-initializing the RL model.
            # For now, assume agent list is stable or handle model re-init if needed.
            # A simpler approach is to return None and let the run loop handle it.
            expected_obs_size = 5 + self.num_agents
            if self.observation_space.shape[0] != expected_obs_size:
                 self.logger.error(f"Number of controllable agents changed ({self.num_agents}) since initialization. Observation space mismatch. Cannot get state.")
                 # TODO: Implement model re-initialization or handle this state change robustly.
                 return None


            scoring_agent = self.orchestrator.agents.get('scoring')
            if not scoring_agent or not hasattr(scoring_agent, 'calculate_total_profit') or not hasattr(scoring_agent, 'score_all_clients'):
                self.logger.error("ScoringAgent not found or missing required methods.")
                return None # Cannot get state without scoring

            profit = 0.0
            avg_client_score = 0.0
            try:
                # Use a single session for scoring calls within this state retrieval
                async with self.session_maker() as session:
                    async with session.begin(): # Ensure commit/rollback for profit metric logging
                        profit = await scoring_agent.calculate_total_profit(session)

                # Score all clients outside the transaction
                client_scores = await scoring_agent.score_all_clients()
                if client_scores:
                    avg_client_score = np.mean(list(client_scores.values())) if client_scores else 0.0

            except SQLAlchemyError as db_err:
                 self.logger.error(f"Database error getting profit/scores: {db_err}", exc_info=True)
                 return None # Return None on DB error
            except Exception as score_err:
                 self.logger.error(f"Error interacting with ScoringAgent: {score_err}", exc_info=True)
                 return None

            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            # Sum tasks from controllable agents' queues + potentially orchestrator queue
            active_tasks = sum(
                getattr(agent.internal_state.get('task_queue', None), 'qsize', lambda: 0)()
                for agent in self.agents_with_concurrency if hasattr(agent, 'internal_state')
            )
            # Add orchestrator queue size if relevant
            # if hasattr(self.orchestrator, 'task_queue'): active_tasks += self.orchestrator.task_queue.qsize()

            # Get current concurrency limits accurately
            concurrency_limits = []
            for agent in self.agents_with_concurrency:
                 # Ensure agent still exists and has the attribute
                 if agent in self.orchestrator.agents.values() and hasattr(agent, 'internal_state') and 'max_concurrency' in agent.internal_state:
                      concurrency_limits.append(agent.internal_state['max_concurrency'])
                 else:
                      self.logger.warning(f"Could not get max_concurrency for agent {getattr(agent, 'AGENT_NAME', 'Unknown')}. Using default 1.")
                      concurrency_limits.append(1) # Default value if agent missing/state incorrect

            # Assemble state
            state_list = [profit, avg_client_score, cpu, memory, active_tasks] + concurrency_limits
            state = np.array(state_list, dtype=np.float32)

            # Validate and clamp state
            if state.shape[0] != expected_obs_size:
                 self.logger.error(f"State vector size mismatch! Expected {expected_obs_size}, got {state.shape[0]}. State: {state}. Concurrencies: {concurrency_limits}")
                 # This indicates a serious issue, likely with agent tracking.
                 return None

            if not self.observation_space.contains(state):
                 self.logger.warning(f"Raw state {state} is outside defined observation space {self.observation_space}. Clamping state.")
                 state = np.clip(state, self.observation_space.low, self.observation_space.high)

            return state

        except Exception as e:
            self.logger.error(f"Failed to get system state: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'):
                await self.orchestrator.report_error(self.AGENT_NAME, f"Failed to get system state: {e}")
            return None


    async def apply_action(self, action: np.ndarray):
        """Apply concurrency adjustments based on scaled action from the RL model."""
        # --- Structured Thinking Step ---
        await self._internal_think(f"Applying RL action (concurrency factors): {np.round(action, 3)}")
        # --- End Structured Thinking Step ---

        self._update_controllable_agents() # Refresh agent list
        if self.num_agents == 0:
            self.logger.warning("Apply action called but no controllable agents found.")
            return
        if action.shape != self.action_space.shape:
             self.logger.error(f"Action shape mismatch: Expected {self.action_space.shape}, got {action.shape}")
             return

        limits_applied = {}
        budget_agent = self.orchestrator.agents.get('budget')
        cost_per_unit = float(self.config.get('CONCURRENCY_COST_PER_UNIT', 0.01)) # Example cost

        agents_to_update = list(self.agents_with_concurrency) # Iterate over a copy

        for i, agent in enumerate(agents_to_update):
             agent_name = getattr(agent, 'AGENT_NAME', f'Agent_{i}')
             # Check agent still exists and has the necessary state
             if agent not in self.orchestrator.agents.values() or not hasattr(agent, 'internal_state') or 'max_concurrency' not in agent.internal_state:
                  self.logger.warning(f"Agent {agent_name} no longer valid or missing state. Skipping action application.")
                  continue

             action_value = float(action[i]) # Ensure scalar
             # Scale action [0, 1] to integer range [1, max_limit]
             new_concurrency_float = action_value * (self.max_concurrency_limit - 1) + 1
             new_concurrency = max(1, min(self.max_concurrency_limit, int(round(new_concurrency_float))))

             old_concurrency = agent.internal_state['max_concurrency']

             if old_concurrency != new_concurrency:
                 self.logger.debug(f"Setting max_concurrency for {agent_name} from {old_concurrency} to {new_concurrency}")
                 agent.internal_state['max_concurrency'] = new_concurrency # Update agent's state
                 # Also update the semaphore if the agent uses one directly
                 if 'send_semaphore' in agent.internal_state and isinstance(agent.internal_state['send_semaphore'], asyncio.Semaphore):
                      # Recreate semaphore - Note: This might disrupt agents currently waiting on it.
                      # A more complex approach involves adjusting semaphore value carefully.
                      agent.internal_state['send_semaphore'] = asyncio.Semaphore(new_concurrency)
                      self.logger.debug(f"Recreated semaphore for {agent_name} with value {new_concurrency}")

                 limits_applied[agent_name] = new_concurrency

                 # Report cost change to BudgetAgent
                 cost_change = (new_concurrency - old_concurrency) * cost_per_unit
                 if abs(cost_change) > 1e-6 and budget_agent and hasattr(budget_agent, 'record_expense'):
                     try:
                         # Use await as BudgetAgent methods are async
                         await budget_agent.record_expense(
                             agent_name=self.AGENT_NAME, amount=abs(cost_change),
                             category="OptimizationAdjustment", # Specific category
                             description=f"Concurrency adjusted for {agent_name}: {old_concurrency} -> {new_concurrency}"
                         )
                     except Exception as track_err:
                          self.logger.error(f"Error tracking concurrency cost via BudgetAgent: {track_err}", exc_info=True)
                 elif not budget_agent:
                      self.logger.warning("BudgetAgent not found, cannot track concurrency cost.")

        if limits_applied:
            await self.log_operation('info', f"Applied concurrency limits: {limits_applied}")


    async def calculate_reward(self) -> float:
        """Compute reward based on profit change and resource usage penalties."""
        # --- Structured Thinking Step ---
        await self._internal_think("Calculating reward: Fetching profit, checking resource usage, applying penalties.")
        # --- End Structured Thinking Step ---
        reward = 0.0
        profit = 0.0
        cpu_penalty = 0.0
        mem_penalty = 0.0
        try:
            scoring_agent = self.orchestrator.agents.get('scoring')
            if not scoring_agent: raise ValueError("ScoringAgent not found.")

            try:
                 async with self.session_maker() as session:
                     # Use current profit as primary reward signal
                     # TODO: Consider using profit *rate* or *change* if absolute profit varies too much
                     profit = await scoring_agent.calculate_total_profit(session)
                     # No commit needed here as calculate_total_profit logs its own metric
            except Exception as score_err:
                 self.logger.error(f"Error getting profit for reward: {score_err}", exc_info=True)
                 # Return neutral reward if profit cannot be obtained
                 return 0.0

            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent

            # Penalties (configurable weights)
            cpu_penalty_threshold = float(self.config.get('OPTIMIZATION_CPU_PENALTY_THRESHOLD', 85.0)) # Higher threshold
            mem_penalty_threshold = float(self.config.get('OPTIMIZATION_MEM_PENALTY_THRESHOLD', 85.0)) # Higher threshold
            cpu_penalty_weight = float(self.config.get('OPTIMIZATION_CPU_PENALTY_WEIGHT', 1.0)) # Increase weight
            mem_penalty_weight = float(self.config.get('OPTIMIZATION_MEM_PENALTY_WEIGHT', 0.8)) # Increase weight

            # Quadratic penalty increases sharply near threshold
            if cpu > cpu_penalty_threshold:
                cpu_penalty = ((cpu - cpu_penalty_threshold) / (100 - cpu_penalty_threshold))**2 * cpu_penalty_weight
            if memory > mem_penalty_threshold:
                mem_penalty = ((memory - mem_penalty_threshold) / (100 - mem_penalty_threshold))**2 * mem_penalty_weight
            total_penalty = cpu_penalty + mem_penalty

            # Reward = Profit - Penalties
            # Scale profit? If profit is typically large, penalties might be insignificant.
            # Example scaling: reward = (profit / 100) - total_penalty
            # For now, use direct subtraction. Tune weights based on observed values.
            reward = profit - total_penalty

            # Log reward components periodically
            current_time = time.time()
            log_interval = 60 # seconds
            if current_time - self.internal_state['last_log_time'] > log_interval:
                self.logger.info(f"Reward Calc: Profit={profit:.2f}, CPU={cpu:.1f} (Pen={cpu_penalty:.2f}), Mem={memory:.1f} (Pen={mem_penalty:.2f}), Reward={reward:.2f}")
                self.internal_state['last_log_time'] = current_time

            return float(reward)

        except Exception as e:
            self.logger.error(f"Failed to calculate reward: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'):
                await self.orchestrator.report_error(self.AGENT_NAME, f"Failed to calculate reward: {e}")
            return 0.0 # Return neutral reward on unexpected error


    async def run(self):
        """Run optimization loop: observe, act, store experience, train."""
        self.logger.info(f"{self.AGENT_NAME} v2.0 (SAC) run loop starting.")
        self.status = "running"

        # Wait for orchestrator approval (if applicable)
        if hasattr(self.orchestrator, 'wait_for_approval'):
             await self.orchestrator.wait_for_approval(self.AGENT_NAME)
        elif hasattr(self.orchestrator, 'approved'):
             while not getattr(self.orchestrator, 'approved', False):
                  self.logger.info(f"{self.AGENT_NAME}: Awaiting orchestrator approval...")
                  await asyncio.sleep(60)

        self.logger.info(f"{self.AGENT_NAME}: Orchestrator approved. Starting optimization cycles.")

        # Initial state observation
        last_state = await self.get_system_state()
        if last_state is None:
            self.logger.critical(f"{self.AGENT_NAME}: Failed to get initial state. Aborting run loop.")
            self.status = "error"
            if hasattr(self.orchestrator, 'report_critical_failure'):
                 await self.orchestrator.report_critical_failure(self.AGENT_NAME, "Failed to get initial state.")
            return

        cycle_interval = float(self.config.get('OPTIMIZATION_CYCLE_INTERVAL_S', 60.0))
        action_effect_delay = float(self.config.get('OPTIMIZATION_ACTION_DELAY_S', cycle_interval / 2)) # Observe effect sooner

        while True:
            loop_start_time = time.time()
            try:
                if self.num_agents == 0: # Check if agents disappeared
                    self.logger.error("No controllable agents found. Stopping optimization loop.")
                    self.status = "error"
                    break

                # --- Action Prediction ---
                await self._internal_think("Predicting next action based on current state.")
                action, _states = self.rl_model.predict(last_state, deterministic=False) # Use exploration

                # --- Action Application ---
                await self.apply_action(action)

                # --- Environment Step & Reward ---
                await asyncio.sleep(action_effect_delay) # Wait for action effects

                current_state = await self.get_system_state()
                if current_state is None:
                     self.logger.warning("Failed to get current system state after action. Skipping experience storage and training for this step.")
                     await asyncio.sleep(cycle_interval) # Wait before next cycle
                     continue # Skip rest of loop iteration

                reward = await self.calculate_reward()
                done = False # Continuous task

                # --- Store Experience ---
                await self._internal_think(f"Storing experience: Reward={reward:.2f}, Done={done}")
                # Ensure shapes are correct for buffer (SB3 usually handles this for Box spaces)
                self.replay_buffer.add(last_state, current_state, action, np.array([reward], dtype=np.float32), np.array([done]), [{}])
                self.internal_state['total_steps'] += 1
                self.rl_model.num_timesteps = self.internal_state['total_steps']

                # --- Training ---
                should_train = False
                if self.internal_state['total_steps'] >= self.learning_starts and self.replay_buffer.pos >= self.batch_size:
                     if self.train_freq.unit == TrainFrequencyUnit.STEP and self.internal_state['total_steps'] % self.train_freq.frequency == 0:
                          should_train = True

                if should_train:
                    await self._internal_think(f"Starting training: {self.gradient_steps} gradient steps. Total steps: {self.internal_state['total_steps']}")
                    await self.log_operation('info', f"Triggering model training ({self.gradient_steps} gradient steps). Total steps: {self.internal_state['total_steps']}, Buffer pos: {self.replay_buffer.pos}")
                    # Run synchronous training in executor thread to avoid blocking event loop
                    await asyncio.to_thread(self.rl_model.train, gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                    self.logger.debug("Training step completed.")

                # Log periodically
                log_step_interval = int(self.config.get("OPTIMIZATION_LOG_INTERVAL_STEPS", 10))
                if self.internal_state['total_steps'] % log_step_interval == 0:
                    self.logger.info(f"Step {self.internal_state['total_steps']}: Reward={reward:.2f}, Action={np.round(action, 2)}, Buffer={self.replay_buffer.pos}/{self.replay_buffer.buffer_size}")
                    # Log metrics to DB
                    async with self.session_maker() as session:
                         async with session.begin():
                              metrics = [
                                   Metric(agent_name=self.AGENT_NAME, timestamp=datetime.now(timezone.utc), metric_name="rl_reward", value=str(reward)),
                                   Metric(agent_name=self.AGENT_NAME, timestamp=datetime.now(timezone.utc), metric_name="rl_buffer_pos", value=str(self.replay_buffer.pos)),
                                   Metric(agent_name=self.AGENT_NAME, timestamp=datetime.now(timezone.utc), metric_name="rl_total_steps", value=str(self.internal_state['total_steps']))
                              ]
                              session.add_all(metrics)

                # Update last state
                last_state = current_state

                # Save model periodically
                save_interval_seconds = int(self.config.get('OPTIMIZATION_SAVE_INTERVAL_S', 3600)) # Save hourly
                current_time = time.time()
                if save_interval_seconds > 0 and (current_time - self.internal_state['last_save_time'] >= save_interval_seconds):
                    save_path = self.internal_state['model_save_path']
                    await self._internal_think(f"Saving RL model periodically to {save_path}")
                    try:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        # Use save_to_zip_file for potentially better compatibility
                        save_to_zip_file(save_path, data=self.rl_model.get_parameters(), info=None)
                        # Optionally save replay buffer
                        self.rl_model.save_replay_buffer(f"{save_path}_replay_buffer")
                        self.logger.info(f"Saved RL model and buffer to {save_path}")
                        self.internal_state['last_save_time'] = current_time
                    except Exception as save_err:
                         self.logger.error(f"Failed to save RL model to {save_path}: {save_err}", exc_info=True)

            except asyncio.CancelledError:
                self.logger.info(f"{self.AGENT_NAME} run loop cancelled.")
                break
            except Exception as e:
                self.logger.critical(f"{self.AGENT_NAME}: CRITICAL error in run loop: {e}", exc_info=True)
                self.status = "error"
                if hasattr(self.orchestrator, 'report_error'):
                    await self.orchestrator.report_error(self.AGENT_NAME, f"Critical run loop error: {e}")
                await asyncio.sleep(60 * 5) # Wait 5 mins before potentially retrying

            # Main loop sleep interval
            loop_end_time = time.time()
            elapsed_time = loop_end_time - loop_start_time
            sleep_duration = max(0.1, cycle_interval - elapsed_time) # Ensure minimum sleep
            await asyncio.sleep(sleep_duration)

        self.status = "stopped"
        self.logger.info(f"{self.AGENT_NAME} run loop finished.")
        # Final model save on exit?
        # self.rl_model.save(self.internal_state['model_save_path'])


    # --- Abstract Method Implementations ---

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """OptimizationAgent is primarily driven by its run loop, not external tasks."""
        self.logger.warning(f"{self.AGENT_NAME} received execute_task, but primarily operates via run loop. Task: {task_details}")
        return {"status": "ignored", "message": "OptimizationAgent operates autonomously via its run loop."}

    async def learning_loop(self):
        """The main RL training happens within the run loop."""
        self.logger.info(f"{self.AGENT_NAME} learning_loop called, but learning is integrated into the main run() method.")
        # This method could potentially be used for meta-learning or analyzing RL performance trends.
        await asyncio.sleep(3600 * 24) # Sleep long

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates the performance of the RL optimization."""
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        # Analyze reward trends, policy entropy, exploration rate, etc.
        # Requires accessing internal SB3 logger data or tracking metrics manually.
        buffer_fill = (self.replay_buffer.pos / self.replay_buffer.buffer_size) * 100 if self.replay_buffer.buffer_size > 0 else 0
        feedback = f"Critique: Total Steps={self.internal_state['total_steps']}, Buffer Fill={buffer_fill:.1f}%. Needs analysis of reward trend and policy convergence."
        # TODO: Query Metric table for recent 'rl_reward' and analyze trend.
        return {"status": "ok", "feedback": feedback}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Generates prompts if OptimizationAgent needed direct LLM calls (unlikely)."""
        self.logger.warning(f"{self.AGENT_NAME} generate_dynamic_prompt called, but not typically used.")
        return f"Analyze optimization context: {json.dumps(task_context)}."

    async def collect_insights(self) -> Dict[str, Any]:
        """Provide concurrency insights and RL status."""
        self._update_controllable_agents() # Ensure agent list is fresh
        limits = {getattr(agent, 'AGENT_NAME', f'Agent_{i}'): agent.internal_state.get('max_concurrency', 'N/A')
                  for i, agent in enumerate(self.agents_with_concurrency) if hasattr(agent, 'internal_state')}

        # Fetch latest profit for context
        profit = 0.0
        try:
            scoring_agent = self.orchestrator.agents.get('scoring')
            if scoring_agent:
                async with self.session_maker() as session:
                    profit = await scoring_agent.calculate_total_profit(session)
            else: self.logger.warning("ScoringAgent not found for insights.")
        except Exception as e: self.logger.error(f"Failed to get profit for insights: {e}")

        buffer_fill = (self.replay_buffer.pos / self.replay_buffer.buffer_size) * 100 if self.replay_buffer.buffer_size > 0 else 0

        return {
            "agent_name": self.AGENT_NAME, "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.status, "num_controllable_agents": self.num_agents,
            "current_concurrency_limits": limits, "current_profit_for_context": round(profit, 2),
            "rl_total_steps": self.internal_state.get('total_steps', 0),
            "rl_buffer_fill_percentage": round(buffer_fill, 1)
            }

# --- End of agents/optimization_agent.py ---