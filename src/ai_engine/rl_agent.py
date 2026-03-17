"""
Reinforcement Learning Agent for Trading Optimization
Uses DQN (Deep Q-Network) and PPO (Proximal Policy Optimization)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pathlib import Path
import pickle

try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    logger.warning("stable-baselines3 not installed. RL agent will not be available.")
    RL_AVAILABLE = False


class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for trading.

    State: Technical indicators + portfolio state
    Actions: 0=SELL, 1=HOLD, 2=BUY
    Reward: Profit/loss + risk-adjusted metrics
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0001,
        reward_scaling: float = 1.0
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with OHLC data and technical indicators
            initial_balance: Starting capital
            transaction_cost: Transaction cost per trade (0.01 = 1%)
            reward_scaling: Scale factor for rewards
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling

        # Extract features (exclude OHLC columns)
        feature_cols = [col for col in df.columns
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'time']]

        self.features = df[feature_cols].fillna(0).values
        self.prices = df['close'].values

        # State: features + portfolio state
        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 3,),  # +3 for position, balance, holdings
            dtype=np.float32
        )

        # Actions: 0=SELL, 1=HOLD, 2=BUY
        self.action_space = spaces.Discrete(3)

        # Episode state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.holdings = 0.0  # Number of units held
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trade_history = []

        logger.info(f"Trading environment initialized with {len(df)} steps, "
                   f"{n_features} features")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.holdings = 0.0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trade_history = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.

        Returns:
            State vector: [features, position, balance, holdings_value]
        """
        if self.current_step >= len(self.features):
            self.current_step = len(self.features) - 1

        # Current features
        features = self.features[self.current_step]

        # Portfolio state
        current_price = self.prices[self.current_step]
        holdings_value = self.holdings * current_price

        portfolio_state = np.array([
            self.position,  # Current position
            self.balance / self.initial_balance,  # Normalized balance
            holdings_value / self.initial_balance  # Normalized holdings value
        ])

        # Combine features and portfolio state
        obs = np.concatenate([features, portfolio_state]).astype(np.float32)

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=SELL, 1=HOLD, 2=BUY

        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.prices[self.current_step]
        reward = 0.0
        info = {}

        # Execute action
        if action == 2:  # BUY
            if self.position <= 0:  # Not long yet
                # Close short position if exists
                if self.position == -1:
                    profit = (self.entry_price - current_price) * self.holdings
                    reward += profit
                    self.balance += profit - (abs(profit) * self.transaction_cost)
                    info['closed_short'] = profit

                # Open long position
                max_units = self.balance / current_price
                self.holdings = max_units * 0.95  # Use 95% of balance
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.holdings * current_price * (1 + self.transaction_cost)
                info['opened_long'] = self.holdings

        elif action == 0:  # SELL
            if self.position >= 0:  # Not short yet
                # Close long position if exists
                if self.position == 1:
                    profit = (current_price - self.entry_price) * self.holdings
                    reward += profit
                    self.balance += self.holdings * current_price * (1 - self.transaction_cost)
                    self.holdings = 0
                    info['closed_long'] = profit

                # Open short position
                self.holdings = self.balance / current_price * 0.95
                self.position = -1
                self.entry_price = current_price
                info['opened_short'] = self.holdings

        # HOLD (action == 1): Do nothing

        # Calculate unrealized P&L for held positions
        if self.position == 1:  # Long
            unrealized_pnl = (current_price - self.entry_price) * self.holdings
            reward += unrealized_pnl * 0.01  # Small reward for unrealized gains
        elif self.position == -1:  # Short
            unrealized_pnl = (self.entry_price - current_price) * self.holdings
            reward += unrealized_pnl * 0.01

        # Risk penalty: penalize large drawdowns
        total_value = self.balance + (self.holdings * current_price if self.position == 1 else 0)
        drawdown = (self.initial_balance - total_value) / self.initial_balance
        if drawdown > 0.1:  # More than 10% drawdown
            reward -= drawdown * 100  # Penalty

        # Move to next step
        self.current_step += 1
        self.total_reward += reward

        # Check if episode is done
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        # Get next observation
        obs = self._get_observation()

        # Scale reward
        reward = reward * self.reward_scaling

        # Store trade info
        self.trade_history.append({
            'step': self.current_step,
            'action': action,
            'price': current_price,
            'reward': reward,
            'balance': self.balance,
            'position': self.position
        })

        return obs, reward, terminated, truncated, info

    def get_total_return(self) -> float:
        """Calculate total return percentage."""
        current_price = self.prices[self.current_step]
        holdings_value = self.holdings * current_price if self.position == 1 else 0
        total_value = self.balance + holdings_value
        return ((total_value - self.initial_balance) / self.initial_balance) * 100


class RLTradingAgent:
    """
    Reinforcement Learning agent for trading optimization.

    Supports multiple algorithms:
    - DQN (Deep Q-Network): Value-based, good for discrete actions
    - PPO (Proximal Policy Optimization): Policy-based, more stable
    - A2C (Advantage Actor-Critic): Fast, good for quick iterations
    """

    def __init__(
        self,
        algorithm: str = "DQN",
        learning_rate: float = 0.0001,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        gamma: float = 0.99,
        policy: str = "MlpPolicy"
    ):
        """
        Initialize RL agent.

        Args:
            algorithm: "DQN", "PPO", or "A2C"
            learning_rate: Learning rate
            buffer_size: Replay buffer size (for DQN)
            learning_starts: Steps before training starts
            batch_size: Training batch size
            gamma: Discount factor
            policy: Policy network architecture
        """
        if not RL_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RL agent")

        self.name = f"RL_{algorithm}"
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy = policy

        self.model = None
        self.env = None
        self.is_fitted = False

        logger.info(f"Initialized RL agent with algorithm={algorithm}")

    def fit(
        self,
        df: pd.DataFrame,
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5
    ):
        """
        Train the RL agent.

        Args:
            df: Training data with OHLC and features
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
        """
        logger.info(f"Starting RL training with {self.algorithm}...")

        # Create environment
        self.env = TradingEnvironment(df)

        # Wrap environment
        vec_env = DummyVecEnv([lambda: Monitor(self.env)])

        # Create agent based on algorithm
        if self.algorithm == "DQN":
            self.model = DQN(
                self.policy,
                vec_env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                learning_starts=self.learning_starts,
                batch_size=self.batch_size,
                gamma=self.gamma,
                verbose=1,
                tensorboard_log="./tensorboard_logs/"
            )

        elif self.algorithm == "PPO":
            self.model = PPO(
                self.policy,
                vec_env,
                learning_rate=self.learning_rate,
                n_steps=2048,
                batch_size=self.batch_size,
                gamma=self.gamma,
                verbose=1,
                tensorboard_log="./tensorboard_logs/"
            )

        elif self.algorithm == "A2C":
            self.model = A2C(
                self.policy,
                vec_env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=1,
                tensorboard_log="./tensorboard_logs/"
            )

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Setup evaluation callback
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path=f"./models/rl_{self.algorithm.lower()}_best/",
            log_path=f"./logs/rl_{self.algorithm.lower()}/",
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )

        # Train
        logger.info(f"Training {self.algorithm} for {total_timesteps} steps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )

        self.is_fitted = True
        logger.success(f"RL training completed for {self.algorithm}!")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained RL agent.

        Args:
            df: Input features

        Returns:
            Action predictions (0=SELL, 1=HOLD, 2=BUY)
        """
        if not self.is_fitted or self.model is None:
            logger.warning("RL model not fitted. Returning HOLD actions.")
            return np.ones(len(df), dtype=int)

        # Create environment for prediction
        env = TradingEnvironment(df)
        obs, _ = env.reset()

        predictions = []

        for _ in range(len(df)):
            action, _ = self.model.predict(obs, deterministic=True)
            predictions.append(action)

            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        return np.array(predictions)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get action probabilities.

        Args:
            df: Input features

        Returns:
            Probability array with shape (n_samples, 3) for [SELL, HOLD, BUY]
        """
        if not self.is_fitted or self.model is None:
            n_samples = len(df)
            return np.full((n_samples, 3), 1/3)

        # Get predictions
        predictions = self.predict(df)

        # Convert to probabilities (one-hot encoded with softening)
        probas = np.zeros((len(predictions), 3))

        for i, pred in enumerate(predictions):
            if pred == 2:  # BUY
                probas[i] = [0.1, 0.2, 0.7]
            elif pred == 0:  # SELL
                probas[i] = [0.7, 0.2, 0.1]
            else:  # HOLD
                probas[i] = [0.2, 0.6, 0.2]

        return probas

    def evaluate(self, df: pd.DataFrame, n_episodes: int = 10) -> Dict:
        """
        Evaluate agent performance.

        Args:
            df: Evaluation data
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, cannot evaluate")
            return {}

        env = TradingEnvironment(df)

        total_returns = []
        total_rewards = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_return = env.get_total_return()
            total_returns.append(total_return)
            total_rewards.append(episode_reward)

        metrics = {
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_return': np.min(total_returns),
            'max_return': np.max(total_returns)
        }

        logger.info(f"RL Evaluation: Mean Return={metrics['mean_return']:.2f}%, "
                   f"Std={metrics['std_return']:.2f}%")

        return metrics

    def save(self, path: str):
        """Save RL agent to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save stable-baselines3 model
        model_file = str(model_path).replace('.pkl', '.zip')
        self.model.save(model_file)

        # Save metadata
        metadata = {
            'is_fitted': self.is_fitted,
            'algorithm': self.algorithm,
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            'learning_starts': self.learning_starts,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'policy': self.policy
        }

        with open(path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"RL agent saved to {model_file} and {path}")

    def load(self, path: str):
        """Load RL agent from disk."""
        # Load metadata
        with open(path, 'rb') as f:
            metadata = pickle.load(f)

        self.is_fitted = metadata['is_fitted']
        self.algorithm = metadata['algorithm']
        self.learning_rate = metadata['learning_rate']
        self.buffer_size = metadata['buffer_size']
        self.learning_starts = metadata['learning_starts']
        self.batch_size = metadata['batch_size']
        self.gamma = metadata['gamma']
        self.policy = metadata['policy']

        # Load stable-baselines3 model
        model_file = str(path).replace('.pkl', '.zip')

        if self.algorithm == "DQN":
            self.model = DQN.load(model_file)
        elif self.algorithm == "PPO":
            self.model = PPO.load(model_file)
        elif self.algorithm == "A2C":
            self.model = A2C.load(model_file)

        logger.info(f"RL agent loaded from {model_file}")
