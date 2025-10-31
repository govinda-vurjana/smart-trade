from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

# Create the Trading environment
env = TradingEnv(live_data)

# Create the DQN model
model = DQN(MlpPolicy, env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)
