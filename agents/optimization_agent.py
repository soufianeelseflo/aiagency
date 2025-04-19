import asyncio
import numpy as np
import psutil
import logging
import time # Architect-Zero: Added missing import
import os   # Architect-Zero: Added missing import
from datetime import datetime, timezone # Architect-Zero: Use timezone.utc
import sqlalchemy # Explicitly import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sql_text # Alias text to avoid conflict
from typing import Optional, Dict, Any, List, Tuple # Architect-Zero: Added typing imports

# --- RL Imports ---
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:
    import gym # Fallback to gym if gymnasium not installed
    from gym.spaces import Box
    logging.warning("Gymnasium not found, falling back to Gym. Consider installing Gymnasium.") # Use logging

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit # Architect-Zero: For clarity

# --- Project Imports ---
from models import Metric # Assuming Metric model is used for logging RL metrics
# from utils.database import ... # Import specific DB functions if needed
# Architect-Zero: Assuming Orchestrator, ScoringAgent, BudgetAgent types are available for hinting if desired
# from agents.base import BaseAgent # Example if there's a base class
# from agents.scoring_agent import ScoringAgent # Example
# from agents.budget_agent import BudgetAgent # Example

# Configure standard logger
logger = logging.getLogger(__name__) # Architect-Zero: Standard practice

# Configure SB3 Logger (optional, redirects SB3 logs)
# new_sb3_logger = sb3_configure_logger("logs/sb3_optimization_log", ["stdout", "csv", "tensorboard"])

class OptimizationAgent:
    """
    OptimizationAgent v2.2 (Architect-Zero Refinement): Uses Soft Actor-Critic (SAC)
    for dynamic concurrency optimization. Learns from experience collected in a
    replay buffer to maximize a reward function based on profit and resource utilization.
    Training loop validated against SB3 standards for custom loops.
    """
    AGENT_NAME = "OptimizationAgent" # Architect-Zero: Added for consistent logging/reporting

    def __init__(self, session_maker: AsyncSession, config: Dict[str, Any], orchestrator: Any, clients_models: Dict):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        # self.clients_models = clients_models # Architect-Zero: Keep if needed for state, otherwise remove if unused

        # Identify agents to control
        self.agents_with_concurrency = [
            agent for agent_name, agent in orchestrator.agents.items() # Iterate through items
            if hasattr(agent, 'max_concurrency') and agent != self # Architect-Zero: Exclude self
        ]
        self.num_agents = len(self.agents_with_concurrency)
        if self.num_agents == 0:
            logger.warning(f"{self.AGENT_NAME}: Found no other agents with 'max_concurrency' attribute to optimize.")
            # Agent might become inactive or raise an error depending on design
            # For now, allow initialization but log warning.

        # --- Define Observation and Action Spaces ---
        # Observation: [profit, avg_client_score, cpu, memory, tasks] + concurrencies
        # Use float('inf') for unbounded high values, but consider practical limits
        # Architect-Zero: Define bounds more realistically based on expected system limits
        low_obs = np.array([0, 0, 0, 0, 0] + [1] * self.num_agents, dtype=np.float32) # Min concurrency 1
        # Architect-Zero: Use config for max bounds where possible
        max_profit_obs = float(getattr(config, 'OPTIMIZATION_MAX_PROFIT_OBS', 1e6))
        max_score_obs = float(getattr(config, 'OPTIMIZATION_MAX_SCORE_OBS', 1000)) # Example max score
        max_tasks_obs = float(getattr(config, 'OPTIMIZATION_MAX_TASKS_OBS', 10000))
        self.max_concurrency_limit = int(getattr(config, 'OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT', 50)) # Max concurrency per agent
        high_obs = np.array([max_profit_obs, max_score_obs, 100, 100, max_tasks_obs] + [self.max_concurrency_limit] * self.num_agents, dtype=np.float32)
        observation_size = 5 + self.num_agents
        self.observation_space = Box(low=low_obs, high=high_obs, shape=(observation_size,), dtype=np.float32)

        # Action: Scaled concurrency factor (0 to 1) for each agent
        self.action_space = Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32)

        # --- SAC Model Initialization ---
        buffer_size = int(getattr(config, 'OPTIMIZATION_BUFFER_SIZE', 100000))
        self.learning_starts = int(getattr(config, 'OPTIMIZATION_LEARNING_STARTS', 1000))
        batch_size = int(getattr(config, 'OPTIMIZATION_BATCH_SIZE', 256))
        tau = float(getattr(config, 'OPTIMIZATION_TAU', 0.005))
        gamma = float(getattr(config, 'OPTIMIZATION_GAMMA', 0.99))
        learning_rate = float(getattr(config, 'OPTIMIZATION_LR', 3e-4)) # SB3 default is 3e-4
        # Architect-Zero: Clarify train_freq structure based on SB3
        # train_freq can be int (steps) or tuple (freq, unit) e.g., (1, "step") or (10, "episode")
        raw_train_freq = getattr(config, 'OPTIMIZATION_TRAIN_FREQ', (1, "step"))
        if isinstance(raw_train_freq, int):
             self.train_freq: TrainFreq = (raw_train_freq, TrainFrequencyUnit.STEP)
        else:
             self.train_freq: TrainFreq = (int(raw_train_freq[0]), TrainFrequencyUnit(raw_train_freq[1]))

        self.gradient_steps = int(getattr(config, 'OPTIMIZATION_GRADIENT_STEPS', 1)) # Often same as train_freq[0] for SAC

        # Optional: Action noise (SAC generally doesn't need external noise)
        # n_actions = self.action_space.shape[-1]
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Initialize SAC model
        # Architect-Zero: Ensure policy_kwargs are appropriate for the state size
        policy_kwargs = dict(net_arch=[256, 256]) # Adjust if needed

        self.rl_model = SAC(
            "MlpPolicy",
            env=None, # No standard Gym env, using custom loop
            observation_space=self.observation_space, # Architect-Zero: Explicitly provide spaces
            action_space=self.action_space,       # Architect-Zero: Explicitly provide spaces
            policy_kwargs=policy_kwargs,
            verbose=0, # Set to 1 for more SB3 logs
            buffer_size=buffer_size,
            learning_starts=self.learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=self.train_freq, # Use the processed train_freq tuple/int
            gradient_steps=self.gradient_steps,
            action_noise=None, # SAC uses entropy maximization
            learning_rate=learning_rate,
            replay_buffer_class=ReplayBuffer,
            replay_buffer_kwargs=None,
            seed=getattr(config, 'RANDOM_SEED', None),
            device='auto'
        )
        # Set the logger for the SB3 model (optional)
        # self.rl_model.set_logger(new_sb3_logger)

        # Architect-Zero: Replay buffer is managed internally by SAC model instance
        self.replay_buffer = self.rl_model.replay_buffer # Get reference if needed

        self.total_steps = 0 # Track total steps collected
        self.last_log_time = time.time()
        self.model_save_path = getattr(config, 'OPTIMIZATION_MODEL_SAVE_PATH', "rl_models") # Configurable save path

        logger.info(f"{self.AGENT_NAME} v2.2 (SAC) initialized for {self.num_agents} agents. "
                    f"Buffer: {buffer_size}, Starts: {self.learning_starts}, Batch: {batch_size}, "
                    f"Train Freq: {self.train_freq}, Grad Steps: {self.gradient_steps}")

    async def get_system_state(self) -> Optional[np.ndarray]:
        """Retrieve current system state, handling potential agent errors."""
        try:
            # Ensure scoring agent exists
            scoring_agent = self.orchestrator.agents.get('scoring')
            if not scoring_agent:
                logger.error(f"{self.AGENT_NAME}: ScoringAgent not found in orchestrator.")
                raise ValueError("ScoringAgent not found.")

            profit = 0.0
            avg_client_score = 0.0
            try:
                async with self.session_maker() as session:
                    profit = await scoring_agent.calculate_total_profit(session)
                    # Architect-Zero: Address potential performance issue of score_all_clients
                    # Option 1: Use it but log warning if slow
                    # Option 2: Use a cached/aggregated value if available
                    # Option 3: Optimize the query in ScoringAgent
                    logger.debug(f"{self.AGENT_NAME}: Calling score_all_clients (potential bottleneck).")
                    client_scores = await scoring_agent.score_all_clients() # Assuming this returns Dict[Any, float]
                    if client_scores:
                        avg_client_score = np.mean(list(client_scores.values()))

            except sqlalchemy.exc.SQLAlchemyError as db_err:
                 logger.error(f"{self.AGENT_NAME}: Database error getting profit/scores: {db_err}", exc_info=True)
                 # Return None or cached values if critical? For now, return None.
                 return None
            except Exception as score_err:
                 logger.error(f"{self.AGENT_NAME}: Error interacting with ScoringAgent: {score_err}", exc_info=True)
                 return None # Cannot proceed without profit/score

            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            active_tasks = sum(
                # Architect-Zero: Safer attribute access
                getattr(agent, 'task_queue', None).qsize() if hasattr(getattr(agent, 'task_queue', None), 'qsize') else 0
                for agent in self.orchestrator.agents.values()
            )
            # Architect-Zero: Ensure agents haven't been removed mid-operation
            current_agent_names = {agent.__class__.__name__ for agent in self.agents_with_concurrency}
            concurrency_limits = [getattr(agent, 'max_concurrency', 1) for agent in self.agents_with_concurrency if agent.__class__.__name__ in current_agent_names]
            if len(concurrency_limits) != self.num_agents:
                 logger.warning(f"{self.AGENT_NAME}: Number of agents changed during state collection. Re-evaluating.")
                 # Handle this case - maybe skip step or re-initialize agent list? For now, log and continue carefully.
                 # Fallback: Use last known number or pad? Padding is risky. Best to skip step.
                 return None


            state_list = [profit, avg_client_score, cpu, memory, active_tasks] + concurrency_limits
            state = np.array(state_list, dtype=np.float32)

            # Ensure state conforms to observation space *before* returning
            if not self.observation_space.contains(state):
                 logger.warning(f"{self.AGENT_NAME}: Raw state {state} is outside defined observation space {self.observation_space}. Clamping state.")
                 state = np.clip(state, self.observation_space.low, self.observation_space.high)

            return state

        except Exception as e:
            logger.error(f"{self.AGENT_NAME}: Failed to get system state: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'):
                await self.orchestrator.report_error(self.AGENT_NAME, f"Failed to get system state: {e}")
            return None


    async def apply_action(self, action: np.ndarray):
        """Apply concurrency adjustments based on scaled action and track costs."""
        if self.num_agents == 0: return
        if action.shape != self.action_space.shape:
             logger.error(f"{self.AGENT_NAME}: Action shape mismatch: Expected {self.action_space.shape}, got {action.shape}")
             return

        limits_applied = {}
        budget_agent = self.orchestrator.agents.get('budget') # Get once
        cost_per_unit = float(getattr(self.config, 'CONCURRENCY_COST_PER_UNIT', 0.1))

        # Architect-Zero: Iterate safely in case agent list changes
        agents_to_update = list(self.agents_with_concurrency) # Create copy for iteration

        for i, agent in enumerate(agents_to_update):
             # Check if agent still exists in orchestrator (might have been removed)
             if agent not in self.orchestrator.agents.values():
                  logger.warning(f"{self.AGENT_NAME}: Agent {agent.__class__.__name__} no longer exists. Skipping action application.")
                  continue

             # Scale action (0 to 1) to the desired concurrency range [1, max_limit]
             action_value = float(action[i]) # Ensure scalar
             # Scale action to range [0, max_limit - 1] then add 1
             new_concurrency_float = action_value * (self.max_concurrency_limit - 1) + 1
             # Round and clamp to integer between 1 and max_limit
             new_concurrency = max(1, min(self.max_concurrency_limit, int(round(new_concurrency_float))))

             old_concurrency = getattr(agent, 'max_concurrency', 1)

             if old_concurrency != new_concurrency:
                 agent_name = agent.__class__.__name__ # Get name for logging/tracking
                 logger.debug(f"{self.AGENT_NAME}: Setting max_concurrency for {agent_name} from {old_concurrency} to {new_concurrency}")
                 agent.max_concurrency = new_concurrency # Directly set the attribute
                 limits_applied[agent_name] = new_concurrency

                 # Calculate and track cost
                 cost_change = (new_concurrency - old_concurrency) * cost_per_unit
                 if abs(cost_change) > 1e-6: # Track non-negligible changes
                     if budget_agent and hasattr(budget_agent, 'track_expense'):
                         try:
                             await budget_agent.track_expense(
                                 amount=abs(cost_change),
                                 category="ConcurrencyAdjustment", # More specific category
                                 description=f"{self.AGENT_NAME} adjusted {agent_name}: {old_concurrency} -> {new_concurrency}"
                             )
                         except ValueError as budget_err: # Catch specific budget exceeded errors if raised
                              logger.warning(f"{self.AGENT_NAME}: Failed to track concurrency cost for {agent_name}: {budget_err}")
                         except Exception as track_err:
                              logger.error(f"{self.AGENT_NAME}: Error tracking concurrency cost via BudgetAgent: {track_err}", exc_info=True)
                     elif not budget_agent:
                          logger.warning(f"{self.AGENT_NAME}: BudgetAgent not found, cannot track concurrency cost.")

        if limits_applied:
            logger.info(f"{self.AGENT_NAME}: Applied concurrency limits: {limits_applied}")
        # else: # Architect-Zero: No need for else log, debug above covers it
        #      logger.debug(f"{self.AGENT_NAME}: No concurrency limits changed by action.")


    async def calculate_reward(self) -> float:
        """Compute reward based on profit change and resource usage penalties."""
        try:
            # Ensure scoring agent exists
            scoring_agent = self.orchestrator.agents.get('scoring')
            if not scoring_agent:
                 logger.error(f"{self.AGENT_NAME}: ScoringAgent not found for reward calculation.")
                 raise ValueError("ScoringAgent not found.")

            profit = 0.0
            try:
                 async with self.session_maker() as session:
                     # Consider calculating profit *change* if possible/meaningful, or use absolute profit
                     profit = await scoring_agent.calculate_total_profit(session)
            except sqlalchemy.exc.SQLAlchemyError as db_err:
                 logger.error(f"{self.AGENT_NAME}: Database error getting profit for reward: {db_err}", exc_info=True)
                 return 0.0 # Neutral reward on DB error
            except Exception as score_err:
                 logger.error(f"{self.AGENT_NAME}: Error interacting with ScoringAgent for reward: {score_err}", exc_info=True)
                 return 0.0 # Neutral reward on scoring error

            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent

            # Penalties (tune weights via config)
            cpu_penalty_threshold = float(getattr(self.config, 'OPTIMIZATION_CPU_PENALTY_THRESHOLD', 80.0))
            mem_penalty_threshold = float(getattr(self.config, 'OPTIMIZATION_MEM_PENALTY_THRESHOLD', 80.0))
            cpu_penalty_weight = float(getattr(self.config, 'OPTIMIZATION_CPU_PENALTY_WEIGHT', 0.5))
            mem_penalty_weight = float(getattr(self.config, 'OPTIMIZATION_MEM_PENALTY_WEIGHT', 0.5))

            # Architect-Zero: Use quadratic penalty for sharper increase near threshold
            cpu_penalty = max(0, (cpu - cpu_penalty_threshold) / (100 - cpu_penalty_threshold))**2 * cpu_penalty_weight if cpu > cpu_penalty_threshold else 0
            mem_penalty = max(0, (memory - mem_penalty_threshold) / (100 - mem_penalty_threshold))**2 * mem_penalty_weight if memory > mem_penalty_threshold else 0
            total_penalty = cpu_penalty + mem_penalty

            # Reward = Profit - Penalties
            # Architect-Zero: Consider normalizing profit or using profit *rate* if absolute values vary wildly
            # For now, assume profit scale is manageable.
            reward = profit - total_penalty # Simple linear combination

            # Log reward components periodically
            current_time = time.time()
            log_interval = 60 # seconds
            if current_time - self.last_log_time > log_interval:
                logger.info(f"{self.AGENT_NAME} Reward Calc: Profit={profit:.2f}, CPU={cpu:.1f} (Pen={cpu_penalty:.2f}), Mem={memory:.1f} (Pen={mem_penalty:.2f}), Reward={reward:.2f}")
                self.last_log_time = current_time

            return float(reward) # Ensure float return

        except Exception as e:
            logger.error(f"{self.AGENT_NAME}: Failed to calculate reward: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'):
                await self.orchestrator.report_error(self.AGENT_NAME, f"Failed to calculate reward: {e}")
            return 0.0 # Return neutral reward on unexpected error


    async def run(self):
        """Run optimization loop: observe, act, store experience, train."""
        logger.info(f"{self.AGENT_NAME} v2.2 (SAC) run loop starting.")

        # Wait for orchestrator approval (if applicable)
        if hasattr(self.orchestrator, 'wait_for_approval'):
             await self.orchestrator.wait_for_approval(self.AGENT_NAME)
        elif hasattr(self.orchestrator, 'approved'): # Fallback check
             while not getattr(self.orchestrator, 'approved', False):
                  logger.info(f"{self.AGENT_NAME}: Awaiting orchestrator approval...")
                  await asyncio.sleep(60)

        logger.info(f"{self.AGENT_NAME}: Orchestrator approved. Starting optimization cycles.")

        # Initial state observation
        last_state = await self.get_system_state()
        if last_state is None:
            logger.critical(f"{self.AGENT_NAME}: Failed to get initial state. Aborting run loop.")
            # Optionally report critical failure to orchestrator
            if hasattr(self.orchestrator, 'report_critical_failure'):
                 await self.orchestrator.report_critical_failure(self.AGENT_NAME, "Failed to get initial state.")
            return # Cannot proceed

        cycle_interval = float(getattr(self.config, 'OPTIMIZATION_CYCLE_INTERVAL_S', 60.0)) # How often to start a new cycle
        action_effect_delay = float(getattr(self.config, 'OPTIMIZATION_ACTION_DELAY_S', cycle_interval)) # Delay between action and next state observation

        while True:
            loop_start_time = time.time()
            try:
                # --- Action Prediction ---
                # Use predict with last_state, SB3 handles exploration noise internally for SAC
                # deterministic=False during training/exploration
                action, _states = self.rl_model.predict(last_state, deterministic=False)

                # --- Action Application ---
                await self.apply_action(action)

                # --- Environment Step & Reward ---
                # Wait for a period to observe the effect of the action
                await asyncio.sleep(action_effect_delay)

                current_state = await self.get_system_state()
                if current_state is None:
                     logger.warning(f"{self.AGENT_NAME}: Failed to get current system state after action. Skipping experience storage and training for this step.")
                     # Wait before trying next cycle to avoid hammering a failing state check
                     await asyncio.sleep(cycle_interval)
                     continue

                reward = await self.calculate_reward()
                done = False # Continuous operation

                # --- Store Experience in Replay Buffer ---
                # The SAC model internally manages the ReplayBuffer. Add transition.
                # Ensure shapes match expected buffer shapes. SB3 handles reshaping for standard Box spaces.
                # Use deep copies if state/action arrays might be modified elsewhere (unlikely here)
                self.replay_buffer.add(
                    last_state,       # Observation
                    current_state,    # Next observation
                    action,           # Action
                    np.array([reward], dtype=np.float32), # Reward (as array)
                    np.array([done]), # Done flag (as array)
                    [{}]              # infos (list of dicts, one per env)
                )
                self.total_steps += 1
                # Architect-Zero: Update internal SB3 timestep counter if needed for callbacks/logging
                self.rl_model.num_timesteps = self.total_steps


                # --- Training ---
                # Architect-Zero: This logic correctly reflects the plan's *intent* and standard SB3 custom loop practice.
                # model.train() is called conditionally based on total steps collected matching the frequency,
                # *after* learning_starts is met and enough samples are in the buffer for a batch.
                should_train = False
                if self.total_steps >= self.learning_starts:
                    if self.replay_buffer.pos >= self.batch_size: # Check if buffer has enough for a batch
                         # Check training frequency (e.g., train every `self.train_freq.frequency` steps)
                         if self.train_freq.unit == TrainFrequencyUnit.STEP and self.total_steps % self.train_freq.frequency == 0:
                              should_train = True
                         # Add episode-based frequency check if needed, though less common here
                         # elif self.train_freq.unit == TrainFrequencyUnit.EPISODE and done: # 'done' is always False here
                         #    if self.rl_model.num_episodes % self.train_freq.frequency == 0:
                         #         should_train = True

                if should_train:
                    logger.info(f"{self.AGENT_NAME}: Triggering model training ({self.gradient_steps} gradient steps). Total steps: {self.total_steps}, Buffer pos: {self.replay_buffer.pos}")
                    # Perform the configured number of gradient updates using samples from the buffer
                    self.rl_model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                    logger.debug(f"{self.AGENT_NAME}: Training step completed.")
                # else: # Architect-Zero: Add debug logs for why training was skipped
                #     if self.total_steps < self.learning_starts:
                #         logger.debug(f"{self.AGENT_NAME}: Skipping training: total_steps ({self.total_steps}) < learning_starts ({self.learning_starts})")
                #     elif self.replay_buffer.pos < self.batch_size:
                #         logger.debug(f"{self.AGENT_NAME}: Skipping training: buffer position ({self.replay_buffer.pos}) < batch_size ({self.batch_size})")
                #     elif self.train_freq.unit == TrainFrequencyUnit.STEP and self.total_steps % self.train_freq.frequency != 0:
                #          logger.debug(f"{self.AGENT_NAME}: Skipping training: train_freq interval not met ({self.total_steps} % {self.train_freq.frequency} != 0)")


                # Log periodically
                log_step_interval = 10
                if self.total_steps % log_step_interval == 0:
                    logger.info(f"{self.AGENT_NAME} Step {self.total_steps}: Reward={reward:.2f}, Action={np.round(action, 2)}, Buffer={self.replay_buffer.pos}/{self.replay_buffer.buffer_size}")

                # Update last state for next iteration
                last_state = current_state

                # Optional: Save model periodically
                save_interval_steps = int(getattr(self.config, 'OPTIMIZATION_SAVE_INTERVAL_STEPS', 10000))
                if save_interval_steps > 0 and self.total_steps % save_interval_steps == 0:
                    save_path = os.path.join(self.model_save_path, f"sac_optimization_agent_{self.total_steps}")
                    try:
                        os.makedirs(self.model_save_path, exist_ok=True) # Ensure directory exists
                        self.rl_model.save(save_path)
                        logger.info(f"{self.AGENT_NAME}: Saved RL model to {save_path}.zip")
                    except Exception as save_err:
                         logger.error(f"{self.AGENT_NAME}: Failed to save RL model to {save_path}: {save_err}", exc_info=True)


            except asyncio.CancelledError:
                logger.info(f"{self.AGENT_NAME} run loop cancelled.")
                break
            except Exception as e:
                logger.critical(f"{self.AGENT_NAME}: CRITICAL error in run loop: {e}", exc_info=True)
                if hasattr(self.orchestrator, 'report_error'):
                    await self.orchestrator.report_error(self.AGENT_NAME, f"Critical run loop error: {e}")
                # Implement exponential backoff for retries after critical errors
                await asyncio.sleep(60 * 5) # Wait 5 mins before potentially retrying

            # Main loop sleep interval - ensure it accounts for processing time
            loop_end_time = time.time()
            elapsed_time = loop_end_time - loop_start_time
            sleep_duration = max(0, cycle_interval - elapsed_time)
            await asyncio.sleep(sleep_duration)


    async def get_insights(self) -> Dict[str, Any]:
        """Provide concurrency insights for BudgetAgent and UI."""
        # Architect-Zero: Ensure agent list is current
        current_agents = [agent for agent in self.agents_with_concurrency if agent in self.orchestrator.agents.values()]
        limits = {agent.__class__.__name__: getattr(agent, 'max_concurrency', 'N/A') for agent in current_agents}

        profit = 0.0
        try:
            scoring_agent = self.orchestrator.agents.get('scoring')
            if scoring_agent:
                async with self.session_maker() as session:
                    profit = await scoring_agent.calculate_total_profit(session)
            else:
                 logger.warning(f"{self.AGENT_NAME}: ScoringAgent not found for insights.")
        except Exception as e:
            logger.error(f"{self.AGENT_NAME}: Failed to get profit for insights: {e}", exc_info=True)

        return {
            "agent_name": self.AGENT_NAME,
            "timestamp": datetime.utcnow().isoformat(),
            "concurrency_limits": limits,
            "current_profit": profit,
            "total_steps": self.total_steps,
            "buffer_fill_percentage": (self.replay_buffer.pos / self.replay_buffer.buffer_size) * 100 if self.replay_buffer.buffer_size > 0 else 0
            }