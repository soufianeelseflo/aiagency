# Filename: agents/optimization_agent.py
# Description: Agent using Soft Actor-Critic (SAC) for dynamic concurrency optimization.
# Version: 1.1 (Implemented Core Logic, RL Loop, Integrations)

import asyncio
import numpy as np
import psutil
import logging
import time
import os
import json # For logging insights
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

# --- RL Imports ---
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:
    import gym # Fallback
    from gym.spaces import Box
    logging.warning("Gymnasium not found, falling back to Gym. Consider installing Gymnasium.")

# Ensure stable_baselines3 is installed (listed in requirements.txt)
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.noise import NormalActionNoise # Although not used by SAC default
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure as sb3_configure_logger
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
    from stable_baselines3.common.env_util import make_vec_env # For potential future env wrapping
except ImportError as e:
     logging.critical(f"stable-baselines3 import failed: {e}. OptimizationAgent cannot function. Please install.")
     # Raise error to prevent agent initialization if SB3 is missing
     raise ImportError("stable-baselines3 is required for OptimizationAgent.") from e

# --- Project Imports ---
from models import Metric # Assuming Metric model is used for logging RL metrics
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
# Base Agent Import
try:
    from .base_agent import GeniusAgentBase, KBInterface # Use relative import if applicable
except ImportError:
    from base_agent import GeniusAgentBase, KBInterface # Fallback

# Configure standard logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Configure SB3 Logger (optional, redirects SB3 logs)
# sb3_log_path = "logs/sb3_optimization_log"
# os.makedirs(sb3_log_path, exist_ok=True)
# new_sb3_logger = sb3_configure_logger(sb3_log_path, ["stdout", "csv", "tensorboard"])

class OptimizationAgent(GeniusAgentBase):
    """
    OptimizationAgent v1.1: Uses Soft Actor-Critic (SAC) for dynamic concurrency
    optimization. Learns from experience to maximize profit while managing resource usage.
    """
    AGENT_NAME = "OptimizationAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any):
        """Initializes the OptimizationAgent."""
        config = getattr(orchestrator, 'config', None)
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, session_maker=session_maker)

        # Identify agents to control (must have 'max_concurrency' attribute)
        self.agents_with_concurrency: List[GeniusAgentBase] = [] # Store agent instances
        self._update_controllable_agents() # Initial population

        if self.num_agents == 0:
            self.logger.critical(f"{self.AGENT_NAME}: Found no other agents with 'max_concurrency' attribute to optimize. Agent will be idle.")
            self.status = "error" # Mark as error state if no agents to control
            # Agent cannot function without controllable agents
            return

        # --- Define Observation and Action Spaces ---
        # Observation: [profit, avg_client_score, cpu, memory, active_tasks] + N concurrencies
        low_obs = np.array([0, 0, 0, 0, 0] + [1] * self.num_agents, dtype=np.float32) # Min concurrency 1
        max_profit_obs = float(self.config.get('OPTIMIZATION_MAX_PROFIT_OBS', 1e6))
        max_score_obs = float(self.config.get('OPTIMIZATION_MAX_SCORE_OBS', 1000))
        max_tasks_obs = float(self.config.get('OPTIMIZATION_MAX_TASKS_OBS', 10000))
        self.max_concurrency_limit = int(self.config.get('OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT', 50))
        high_obs = np.array([max_profit_obs, max_score_obs, 100, 100, max_tasks_obs] + [self.max_concurrency_limit] * self.num_agents, dtype=np.float32)
        observation_size = 5 + self.num_agents
        self.observation_space = Box(low=low_obs, high=high_obs, shape=(observation_size,), dtype=np.float32)
        self.action_space = Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32) # Scaled action [0, 1]

        # --- SAC Model Initialization ---
        buffer_size = int(self.config.get('OPTIMIZATION_BUFFER_SIZE', 100000))
        self.learning_starts = int(self.config.get('OPTIMIZATION_LEARNING_STARTS', 1000))
        batch_size = int(self.config.get('OPTIMIZATION_BATCH_SIZE', 256))
        tau = float(self.config.get('OPTIMIZATION_TAU', 0.005))
        gamma = float(self.config.get('OPTIMIZATION_GAMMA', 0.99))
        learning_rate = float(self.config.get('OPTIMIZATION_LR', 3e-4))
        raw_train_freq = self.config.get('OPTIMIZATION_TRAIN_FREQ', (1, "step")) # Default: train every step
        if isinstance(raw_train_freq, int): self.train_freq = TrainFreq(raw_train_freq, TrainFrequencyUnit.STEP)
        elif isinstance(raw_train_freq, (list, tuple)) and len(raw_train_freq) == 2: self.train_freq = TrainFreq(int(raw_train_freq[0]), TrainFrequencyUnit(raw_train_freq[1]))
        else: self.logger.warning(f"Invalid OPTIMIZATION_TRAIN_FREQ format: {raw_train_freq}. Defaulting to (1, 'step')."); self.train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)

        self.gradient_steps = int(self.config.get('OPTIMIZATION_GRADIENT_STEPS', self.train_freq.frequency if self.train_freq.unit == TrainFrequencyUnit.STEP else 1))

        policy_kwargs = dict(net_arch=[256, 256]) # Standard MLP size

        # --- Model Loading/Initialization ---
        self.model_save_path = self.config.get('OPTIMIZATION_MODEL_SAVE_PATH', "/app/rl_models") # Use path inside container
        self.model_load_path = self.config.get('OPTIMIZATION_MODEL_LOAD_PATH') # Path to load from (optional)
        load_successful = False
        if self.model_load_path and os.path.exists(f"{self.model_load_path}.zip"):
            try:
                self.rl_model = SAC.load(
                    self.model_load_path,
                    env=None, # Custom loop, no standard env needed here
                    device='auto',
                    # Pass necessary parameters if they changed since saving
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=self.learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=self.train_freq,
                    gradient_steps=self.gradient_steps,
                )
                # Ensure buffer is loaded correctly
                self.replay_buffer = self.rl_model.replay_buffer
                self.total_steps = self.rl_model.num_timesteps # Sync steps
                load_successful = True
                self.logger.info(f"Successfully loaded pre-trained SAC model from: {self.model_load_path}.zip")
            except Exception as load_err:
                self.logger.error(f"Failed to load SAC model from {self.model_load_path}: {load_err}. Initializing new model.", exc_info=True)
                load_successful = False

        if not load_successful:
            self.rl_model = SAC(
                "MlpPolicy", env=None, observation_space=self.observation_space, action_space=self.action_space,
                policy_kwargs=policy_kwargs, verbose=0, buffer_size=buffer_size, learning_starts=self.learning_starts,
                batch_size=batch_size, tau=tau, gamma=gamma, train_freq=self.train_freq, gradient_steps=self.gradient_steps,
                action_noise=None, learning_rate=learning_rate, replay_buffer_class=ReplayBuffer,
                seed=self.config.get('RANDOM_SEED'), device='auto'
            )
            self.replay_buffer = self.rl_model.replay_buffer
            self.total_steps = 0
            self.logger.info("Initialized new SAC model.")

        # self.rl_model.set_logger(new_sb3_logger) # Optional: Set custom SB3 logger

        self.last_log_time = time.time()
        self.save_interval_steps = int(self.config.get('OPTIMIZATION_SAVE_INTERVAL_STEPS', 1000)) # Save every 1000 steps

        self.logger.info(f"{self.AGENT_NAME} v1.1 (SAC) initialized for {self.num_agents} agents. "
                    f"Buffer: {buffer_size}, Starts: {self.learning_starts}, Batch: {batch_size}, "
                    f"Train Freq: {self.train_freq}, Grad Steps: {self.gradient_steps}")

    def _update_controllable_agents(self):
        """Updates the list of agents this agent can control."""
        if not self.orchestrator or not hasattr(self.orchestrator, 'agents'):
             self.logger.error("Orchestrator or orchestrator.agents not available for agent update.")
             self.agents_with_concurrency = []
             self.num_agents = 0
             return

        self.agents_with_concurrency = [
            agent for agent_name, agent in self.orchestrator.agents.items()
            if hasattr(agent, 'max_concurrency') and agent != self # Exclude self
        ]
        self.num_agents = len(self.agents_with_concurrency)
        self.logger.debug(f"Updated controllable agents list. Found {self.num_agents} agents.")
        # TODO: Handle changes in num_agents (e.g., resizing action/observation space if possible, or retraining)
        # This is complex and might require restarting the RL agent if the number changes significantly.

    async def get_system_state(self) -> Optional[np.ndarray]:
        """Retrieve current system state (profit, score, resources, concurrencies)."""
        # --- Structured Thinking Step ---
        thinking_process = f"""
        Structured Thinking: Get System State
        1. Goal: Collect current metrics for the RL observation space.
        2. Context: Need profit, avg score, CPU, memory, task queue size, current concurrencies of controllable agents.
        3. Constraints: Must handle potential errors from ScoringAgent or agent list changes. Output must match observation_space.
        4. Information Needed: Access to ScoringAgent, psutil, Orchestrator's agent list.
        5. Plan:
            a. Get profit and scores from ScoringAgent (handle errors).
            b. Get CPU/Memory using psutil.
            c. Get total active tasks from agent queues.
            d. Get current max_concurrency for controllable agents (handle list changes).
            e. Assemble state array.
            f. Validate state against observation_space bounds and clamp if necessary.
            g. Return clamped state array or None on critical failure.
        """
        await self._internal_think(thinking_process)
        # --- End Structured Thinking Step ---
        try:
            scoring_agent = self.orchestrator.agents.get('scoring')
            if not scoring_agent or not hasattr(scoring_agent, 'calculate_total_profit') or not hasattr(scoring_agent, 'score_all_clients'):
                raise ValueError("ScoringAgent not found or missing required methods.")

            profit = 0.0
            avg_client_score = 0.0
            try:
                # Use a single session if ScoringAgent methods accept it, otherwise rely on their internal session handling
                # Assuming methods handle their own sessions based on ScoringAgent code
                profit = await scoring_agent.calculate_total_profit(self.session_maker()) # Pass session_maker if needed
                client_scores = await scoring_agent.score_all_clients()
                if client_scores: avg_client_score = np.mean(list(client_scores.values()))

            except SQLAlchemyError as db_err: logger.error(f"DB error getting profit/scores: {db_err}", exc_info=True); return None
            except Exception as score_err: logger.error(f"Error interacting with ScoringAgent: {score_err}", exc_info=True); return None

            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            active_tasks = sum(
                getattr(agent.internal_state.get('task_queue', None), 'qsize', lambda: 0)() # Safer access to queue size
                for agent in self.orchestrator.agents.values() if hasattr(agent, 'internal_state')
            )

            # Refresh controllable agents list and check consistency
            self._update_controllable_agents()
            if self.num_agents == 0:
                 self.logger.error("No controllable agents found during state collection.")
                 return None # Cannot form state if no agents to control

            concurrency_limits = []
            for agent in self.agents_with_concurrency:
                 # Verify agent still exists in orchestrator (might have been removed)
                 if agent.agent_name in self.orchestrator.agents:
                      concurrency_limits.append(getattr(agent, 'max_concurrency', 1))
                 else:
                      self.logger.warning(f"Agent {agent.agent_name} no longer in orchestrator during state collection. State may be inconsistent.")
                      # Handle inconsistency - Option: return None to skip step
                      return None

            # Ensure the length matches the expected number of agents
            if len(concurrency_limits) != self.num_agents:
                 self.logger.error(f"Concurrency limit count ({len(concurrency_limits)}) mismatch with expected agent count ({self.num_agents}). Skipping state.")
                 return None

            state_list = [profit, avg_client_score, cpu, memory, active_tasks] + concurrency_limits
            state = np.array(state_list, dtype=np.float32)

            # Validate and clamp state
            if state.shape != self.observation_space.shape:
                 self.logger.error(f"Generated state shape {state.shape} does not match observation space {self.observation_space.shape}. Skipping state.")
                 return None
            if not self.observation_space.contains(state):
                 self.logger.warning(f"Raw state {state} outside observation space {self.observation_space}. Clamping.")
                 state = np.clip(state, self.observation_space.low, self.observation_space.high)

            return state

        except Exception as e:
            self.logger.error(f"Failed to get system state: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Failed get system state: {e}")
            return None

    async def apply_action(self, action: np.ndarray):
        """Apply concurrency adjustments based on scaled action [0, 1]."""
        # --- Structured Thinking Step ---
        thinking_process = f"""
        Structured Thinking: Apply Action
        1. Goal: Adjust max_concurrency for controllable agents based on RL action.
        2. Context: Action vector (shape: {action.shape}), list of controllable agents ({self.num_agents}), max concurrency limit ({self.max_concurrency_limit}).
        3. Constraints: Action values are [0, 1]. Output concurrency must be int [1, max_limit]. Report costs. Handle agent list changes.
        4. Information Needed: Current concurrency of each agent. BudgetAgent access via Orchestrator.
        5. Plan:
            a. Validate action shape.
            b. Iterate through action vector and corresponding agents.
            c. Scale action value to [1, max_limit]. Round and clamp.
            d. Get old concurrency.
            e. If changed, update agent's max_concurrency attribute.
            f. Calculate cost change.
            g. Report expense via Orchestrator -> BudgetAgent.
            h. Log applied limits.
        """
        await self._internal_think(thinking_process)
        # --- End Structured Thinking Step ---

        if self.num_agents == 0: self.logger.warning("Apply action called but no agents to control."); return
        if action.shape != self.action_space.shape:
             self.logger.error(f"Action shape mismatch: Expected {self.action_space.shape}, got {action.shape}")
             return

        limits_applied = {}
        budget_agent = self.orchestrator.agents.get('budget')
        cost_per_unit = float(self.config.get('CONCURRENCY_COST_PER_UNIT', 0.01)) # Example cost per unit increase

        # Use the current list of controllable agents
        agents_to_update = list(self.agents_with_concurrency)
        if len(agents_to_update) != self.num_agents:
             self.logger.warning(f"Number of controllable agents changed ({len(agents_to_update)} vs {self.num_agents}) before applying action. Skipping action.")
             return # Skip if list changed size

        for i, agent in enumerate(agents_to_update):
             # Verify agent still exists
             if agent.agent_name not in self.orchestrator.agents:
                  self.logger.warning(f"Agent {agent.agent_name} no longer exists. Skipping action application for this agent.")
                  continue

             action_value = float(action[i]) # Ensure scalar
             # Scale action [0, 1] to integer range [1, max_limit]
             new_concurrency_float = action_value * (self.max_concurrency_limit - 1) + 1
             new_concurrency = max(1, min(self.max_concurrency_limit, int(round(new_concurrency_float))))

             try:
                 old_concurrency = int(getattr(agent, 'max_concurrency', 1))
             except Exception as e:
                  self.logger.error(f"Could not get old concurrency for {agent.agent_name}: {e}. Skipping update.")
                  continue

             if old_concurrency != new_concurrency:
                 agent_name = agent.agent_name # Use consistent name
                 self.logger.debug(f"Setting max_concurrency for {agent_name} from {old_concurrency} to {new_concurrency}")
                 try:
                     setattr(agent, 'max_concurrency', new_concurrency) # Directly set attribute
                     limits_applied[agent_name] = new_concurrency

                     # Report cost change to BudgetAgent via Orchestrator
                     cost_change = (new_concurrency - old_concurrency) * cost_per_unit
                     if abs(cost_change) > 1e-6 and self.orchestrator and hasattr(self.orchestrator, 'report_expense'):
                          await self.orchestrator.report_expense(
                              agent_name=self.AGENT_NAME, amount=abs(cost_change), category="ConcurrencyAdjustment",
                              description=f"Adjusted {agent_name}: {old_concurrency} -> {new_concurrency}"
                          )
                 except AttributeError:
                      self.logger.error(f"Failed to set max_concurrency for {agent_name}. Attribute might not be writable.")
                 except Exception as e:
                      self.logger.error(f"Error setting concurrency or reporting expense for {agent_name}: {e}", exc_info=True)

        if limits_applied: logger.info(f"Applied concurrency limits: {limits_applied}")

    async def calculate_reward(self) -> float:
        """Compute reward based on profit and resource usage penalties."""
        # --- Structured Thinking Step ---
        thinking_process = f"""
        Structured Thinking: Calculate Reward
        1. Goal: Compute scalar reward for the last action based on current system state.
        2. Context: Need current profit, CPU usage, Memory usage. Config for penalty thresholds/weights.
        3. Constraints: Reward should reflect profit maximization while penalizing high resource usage.
        4. Information Needed: Access to ScoringAgent, psutil, config values.
        5. Plan:
            a. Get current profit from ScoringAgent.
            b. Get current CPU/Memory usage from psutil.
            c. Calculate CPU penalty (quadratic above threshold).
            d. Calculate Memory penalty (quadratic above threshold).
            e. Compute final reward = profit - cpu_penalty - mem_penalty.
            f. Log components periodically.
            g. Return reward as float.
        """
        await self._internal_think(thinking_process)
        # --- End Structured Thinking Step ---
        try:
            scoring_agent = self.orchestrator.agents.get('scoring')
            if not scoring_agent: raise ValueError("ScoringAgent not found.")

            profit = 0.0
            try:
                # Assuming calculate_total_profit handles its own session
                profit = await scoring_agent.calculate_total_profit(self.session_maker())
            except Exception as score_err: logger.error(f"Error getting profit for reward: {score_err}", exc_info=True); return 0.0 # Neutral reward

            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent

            # Penalties (Fetch from config)
            cpu_penalty_threshold = float(self.config.get('OPTIMIZATION_CPU_PENALTY_THRESHOLD', 85.0)) # Higher threshold
            mem_penalty_threshold = float(self.config.get('OPTIMIZATION_MEM_PENALTY_THRESHOLD', 85.0)) # Higher threshold
            cpu_penalty_weight = float(self.config.get('OPTIMIZATION_CPU_PENALTY_WEIGHT', 1.0)) # Increase weight
            mem_penalty_weight = float(self.config.get('OPTIMIZATION_MEM_PENALTY_WEIGHT', 1.0)) # Increase weight

            cpu_penalty = max(0, (cpu - cpu_penalty_threshold) / (100 - cpu_penalty_threshold))**2 * cpu_penalty_weight if cpu > cpu_penalty_threshold else 0
            mem_penalty = max(0, (memory - mem_penalty_threshold) / (100 - mem_penalty_threshold))**2 * mem_penalty_weight if memory > mem_penalty_threshold else 0
            total_penalty = cpu_penalty + mem_penalty

            # Reward = Profit - Penalties (Consider scaling profit if needed)
            # Example scaling: reward = (profit / expected_profit_scale) - total_penalty
            reward = profit - total_penalty

            # Log reward components periodically
            current_time = time.time()
            log_interval = 60
            if current_time - self.last_log_time > log_interval:
                self.logger.info(f"Reward Calc: Profit={profit:.2f}, CPU={cpu:.1f}(Pen={cpu_penalty:.2f}), Mem={memory:.1f}(Pen={mem_penalty:.2f}), Reward={reward:.2f}")
                self.last_log_time = current_time

            return float(reward)

        except Exception as e:
            self.logger.error(f"Failed to calculate reward: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Failed calculate reward: {e}")
            return 0.0 # Neutral reward on error

    async def run(self):
        """Run optimization loop: observe, act, store experience, train."""
        if self.status == "error": # Check if init failed
             self.logger.error(f"{self.AGENT_NAME} cannot start run loop due to initialization error (no controllable agents).")
             return

        self.logger.info(f"{self.AGENT_NAME} v1.1 (SAC) run loop starting.")
        self.status = "running"
        agent_status_gauge.labels(agent_name=self.AGENT_NAME).set(1)

        # Wait for orchestrator approval
        while not getattr(self.orchestrator, 'approved', False):
             self.logger.info(f"{self.AGENT_NAME}: Awaiting orchestrator approval...")
             await asyncio.sleep(30)
        self.logger.info(f"{self.AGENT_NAME}: Orchestrator approved. Starting optimization cycles.")

        last_state = await self.get_system_state()
        if last_state is None:
            self.logger.critical(f"{self.AGENT_NAME}: Failed to get initial state. Aborting run loop.")
            self.status = "error"; agent_status_gauge.labels(agent_name=self.AGENT_NAME).set(0)
            if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, "Critical: Failed to get initial state.")
            return

        cycle_interval = float(self.config.get('OPTIMIZATION_CYCLE_INTERVAL_S', 60.0))
        # Action effect delay should be less than or equal to cycle interval
        action_effect_delay = min(cycle_interval, float(self.config.get('OPTIMIZATION_ACTION_DELAY_S', 30.0)))

        while self.status == "running":
            loop_start_time = time.time()
            try:
                # --- Action Prediction ---
                action, _states = self.rl_model.predict(last_state, deterministic=False)

                # --- Action Application ---
                await self.apply_action(action)

                # --- Environment Step & Reward ---
                await asyncio.sleep(action_effect_delay) # Wait for action effects

                current_state = await self.get_system_state()
                if current_state is None:
                     self.logger.warning("Failed get current system state. Skipping experience storage/training.")
                     # Don't update last_state, retry getting state next cycle
                     await asyncio.sleep(max(0, cycle_interval - (time.time() - loop_start_time)))
                     continue

                reward = await self.calculate_reward()
                done = False # Continuous task

                # --- Store Experience ---
                # Ensure shapes are correct for buffer (SB3 usually handles this for Box spaces)
                self.replay_buffer.add(last_state, current_state, action, np.array([reward], dtype=np.float32), np.array([done]), [{}])
                self.total_steps += 1
                self.rl_model.num_timesteps = self.total_steps # Sync SB3 counter

                # --- Training ---
                if self.total_steps >= self.learning_starts and self.replay_buffer.pos >= self.batch_size:
                    if self.train_freq.unit == TrainFrequencyUnit.STEP and self.total_steps % self.train_freq.frequency == 0:
                        self.logger.info(f"Triggering training ({self.gradient_steps} steps). Total steps: {self.total_steps}, Buffer: {self.replay_buffer.pos}")
                        self.rl_model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                        self.logger.debug("Training step completed.")

                # Log periodically
                log_step_interval = 10
                if self.total_steps % log_step_interval == 0:
                    self.logger.info(f"Step {self.total_steps}: Reward={reward:.2f}, Action={np.round(action, 2)}, Buffer={self.replay_buffer.pos}/{self.replay_buffer.buffer_size}")

                # Update last state
                last_state = current_state

                # Save model periodically
                if self.save_interval_steps > 0 and self.total_steps % self.save_interval_steps == 0:
                    save_path = os.path.join(self.model_save_path, f"sac_optimization_agent_{self.total_steps}")
                    try:
                        os.makedirs(self.model_save_path, exist_ok=True)
                        self.rl_model.save(save_path)
                        self.logger.info(f"Saved RL model to {save_path}.zip")
                    except Exception as save_err: logger.error(f"Failed save RL model to {save_path}: {save_err}", exc_info=True)

            except asyncio.CancelledError:
                self.logger.info(f"{self.AGENT_NAME} run loop cancelled.")
                self.status = "stopped"
                break
            except Exception as e:
                self.logger.critical(f"CRITICAL error in OptimizationAgent run loop: {e}", exc_info=True)
                self.status = "error"
                if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Critical run loop error: {e}")
                await asyncio.sleep(60 * 5) # Wait after critical error

            # Ensure loop runs at desired interval
            elapsed_time = time.time() - loop_start_time
            sleep_duration = max(0, cycle_interval - elapsed_time)
            await asyncio.sleep(sleep_duration)

        self.logger.info(f"{self.AGENT_NAME} run loop finished.")
        agent_status_gauge.labels(agent_name=self.AGENT_NAME).set(0) # Mark as stopped

    # --- Abstract Method Implementations ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning(f"{self.AGENT_NAME} received execute_task (not primary function): {task_details}")
        # Can potentially handle tasks like "force_rescale", "get_optimization_status"
        action = task_details.get('action')
        if action == 'get_status':
             return await self.collect_insights() # Reuse insights method
        return {"status": "warning", "message": "OptimizationAgent primarily runs its background optimization loop."}

    async def learning_loop(self):
        # The core RL learning happens within the main run() loop
        self.logger.info(f"{self.AGENT_NAME} learning_loop: RL training occurs within the main run() cycle.")
        while self.status == "running": await asyncio.sleep(3600) # Sleep long, main logic is in run()

    async def self_critique(self) -> Dict[str, Any]:
        self.logger.debug(f"{self.AGENT_NAME} self_critique: Performing basic critique.")
        insights = await self.collect_insights() # Use existing insights method
        feedback = f"Critique: Current Profit={insights.get('current_profit', 0):.2f}. Concurrency: {insights.get('concurrency_limits', {})}. Buffer Fill: {insights.get('buffer_fill_percentage', 0):.1f}%."
        # Add checks: Is profit stagnant? Is buffer full but learning slow? Are limits maxed out?
        return {"status": "ok", "feedback": feedback}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        self.logger.warning(f"{self.AGENT_NAME} generate_dynamic_prompt called - not used by this agent.")
        return "OptimizationAgent does not use LLM prompts for its core function."

    async def collect_insights(self) -> Dict[str, Any]:
        """Provide concurrency insights for BudgetAgent and UI."""
        # Refresh agent list
        self._update_controllable_agents()
        limits = {agent.agent_name: getattr(agent, 'max_concurrency', 'N/A') for agent in self.agents_with_concurrency}

        profit = 0.0
        try:
            scoring_agent = self.orchestrator.agents.get('scoring')
            if scoring_agent: profit = await scoring_agent.calculate_total_profit(self.session_maker())
            else: logger.warning("ScoringAgent not found for insights.")
        except Exception as e: logger.error(f"Failed get profit for insights: {e}", exc_info=True)

        buffer_fill = 0
        if hasattr(self.replay_buffer, 'pos') and hasattr(self.replay_buffer, 'buffer_size') and self.replay_buffer.buffer_size > 0:
             buffer_fill = (self.replay_buffer.pos / self.replay_buffer.buffer_size) * 100

        return {
            "agent_name": self.AGENT_NAME, "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.status, "controllable_agents_count": self.num_agents,
            "concurrency_limits": limits, "current_profit": round(profit, 2),
            "total_rl_steps": self.total_steps, "buffer_fill_percentage": round(buffer_fill, 1)
            }

# --- End of agents/optimization_agent.py ---